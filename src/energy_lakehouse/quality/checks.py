"""Declarative data-quality checks over silver and gold tables.

Each check is a function ``DataFrame -> int`` returning the number of *failing* rows (or
groups). Results are appended to ``ops_data_quality_results``; any failing check with
severity ``error`` aborts the job so a broken run never looks green.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.io import table_exists, write_append

Severity = str  # "error" | "warn"
Predicate = Callable[[DataFrame], int]


class DataQualityError(RuntimeError):
    """Raised when at least one ``error``-severity check fails."""


@dataclass(frozen=True)
class Check:
    name: str
    layer: str
    table: str
    description: str
    failing_rows: Predicate
    severity: Severity = "error"


@dataclass(frozen=True)
class CheckResult:
    check: Check
    failing_rows: int
    table_exists: bool

    @property
    def passed(self) -> bool:
        return self.table_exists and self.failing_rows == 0


# --- reusable predicate factories -----------------------------------------------------------


def rows_where(condition: str) -> Predicate:
    return lambda df: df.filter(F.expr(condition)).count()


def duplicates_on(*keys: str) -> Predicate:
    return lambda df: df.groupBy(*keys).count().filter(F.col("count") > 1).count()


def not_null(*cols: str) -> Predicate:
    cond = " OR ".join(f"`{c}` IS NULL" for c in cols)
    return rows_where(cond)


def one_current_version_per_site(df: DataFrame) -> int:
    per_site = df.groupBy("site_id").agg(F.sum(F.col("is_current").cast("int")).alias("n_current"))
    return per_site.filter(F.col("n_current") != 1).count()


def contiguous_scd2_versions(df: DataFrame) -> int:
    """Every closed version must end exactly where the next one starts (window-based check)."""
    w = Window.partitionBy("site_id").orderBy("valid_from")
    return (
        df.withColumn("_next_from", F.lead("valid_from").over(w))
        .filter(F.col("_next_from").isNotNull() & (F.col("valid_to") != F.col("_next_from")))
        .count()
    )


def region_shares_sum_to_100(df: DataFrame) -> int:
    totals = df.groupBy("month_start", "region").agg(F.sum("share_of_region_pct").alias("s"))
    return totals.filter(F.abs(F.col("s") - 100) > 0.5).count()


def first_row_only_has_null_lag(lag_col: str, order_col: str, partition_col: str = "site_id") -> Predicate:
    def _pred(df: DataFrame) -> int:
        w = Window.partitionBy(partition_col).orderBy(order_col)
        return (
            df.withColumn("_seq", F.row_number().over(w)).filter((F.col("_seq") > 1) & F.col(lag_col).isNull()).count()
        )

    return _pred


CHECKS: tuple[Check, ...] = (
    Check(
        "bronze_readings_not_empty",
        "bronze",
        "meter_readings",
        "raw readings landed",
        lambda df: int(df.limit(1).count() == 0),
        "warn",
    ),
    Check(
        "silver_readings_unique_key",
        "silver",
        "meter_readings",
        "exactly one row per site and hour",
        duplicates_on("site_id", "reading_hour"),
    ),
    Check(
        "silver_readings_no_null_keys",
        "silver",
        "meter_readings",
        "site_id and reading_hour populated",
        not_null("site_id", "reading_hour"),
    ),
    Check(
        "silver_readings_non_negative",
        "silver",
        "meter_readings",
        "filled consumption is never negative",
        rows_where("kwh_filled < 0"),
    ),
    Check(
        "silver_readings_hour_aligned",
        "silver",
        "meter_readings",
        "reading_hour has no minutes/seconds",
        rows_where("minute(reading_hour) <> 0 OR second(reading_hour) <> 0"),
    ),
    Check(
        "silver_readings_imputed_are_filled",
        "silver",
        "meter_readings",
        "forward-fill left no nulls",
        rows_where("kwh_filled IS NULL"),
    ),
    Check(
        "silver_sites_one_current",
        "silver",
        "sites",
        "each site has exactly one current version",
        one_current_version_per_site,
    ),
    Check(
        "silver_sites_contiguous",
        "silver",
        "sites",
        "SCD2 versions do not overlap or leave holes",
        contiguous_scd2_versions,
    ),
    Check(
        "silver_sites_unique_version",
        "silver",
        "sites",
        "one row per site and valid_from",
        duplicates_on("site_id", "valid_from"),
    ),
    Check(
        "silver_prices_rank_starts_at_one",
        "silver",
        "prices",
        "intraday price rank is 1-based",
        rows_where("price_rank_in_day < 1"),
    ),
    Check(
        "silver_generation_share_bounds",
        "silver",
        "generation_mix",
        "source share within 0..100",
        rows_where("share_pct < 0 OR share_pct > 100.01"),
    ),
    Check(
        "gold_daily_mtd_ge_daily",
        "gold",
        "site_daily_consumption",
        "month-to-date total never below the day's total",
        rows_where("kwh_mtd_cumulative < kwh_total - 0.001"),
    ),
    Check(
        "gold_daily_lag_only_null_on_first",
        "gold",
        "site_daily_consumption",
        "kwh_prev_day is null only for a site's first day",
        first_row_only_has_null_lag("kwh_prev_day", "reading_date"),
    ),
    Check(
        "gold_daily_ma7_present",
        "gold",
        "site_daily_consumption",
        "7-day moving average computed for every row",
        rows_where("kwh_ma_7d IS NULL"),
    ),
    Check(
        "gold_monthly_percent_rank_bounds",
        "gold",
        "site_monthly_kpis",
        "percent_rank in [0, 1]",
        rows_where("percent_rank_overall < 0 OR percent_rank_overall > 1"),
    ),
    Check(
        "gold_monthly_cume_dist_bounds",
        "gold",
        "site_monthly_kpis",
        "cume_dist in (0, 1]",
        rows_where("cume_dist_overall <= 0 OR cume_dist_overall > 1"),
    ),
    Check(
        "gold_monthly_quartile_bounds",
        "gold",
        "site_monthly_kpis",
        "ntile(4) yields 1..4",
        rows_where("consumption_quartile NOT BETWEEN 1 AND 4"),
    ),
    Check(
        "gold_monthly_region_share_sums",
        "gold",
        "site_monthly_kpis",
        "share_of_region_pct sums to 100 per region-month",
        region_shares_sum_to_100,
    ),
    Check(
        "gold_peaks_rank_le_row_number",
        "gold",
        "peak_demand_hours",
        "rank never exceeds row_number",
        rows_where("rank_in_day > row_number_in_day"),
    ),
    Check(
        "gold_outages_positive_duration",
        "gold",
        "outage_streaks",
        "streaks last at least one hour and end after they start",
        rows_where("streak_hours < 1 OR streak_end < streak_start"),
    ),
    Check(
        "gold_cost_no_missing_prices",
        "gold",
        "energy_cost_daily",
        "every consumption hour found a market price",
        rows_where("hours_without_price > 0"),
        "warn",
    ),
    Check(
        "gold_distribution_decile_bounds",
        "gold",
        "consumption_distribution_monthly",
        "ntile(10) yields 1..10",
        rows_where("decile_all_sites NOT BETWEEN 1 AND 10"),
    ),
    Check(
        "gold_rollup_month_share_100",
        "gold",
        "consumption_rollup_monthly",
        "month grain rows are 100% of themselves",
        rows_where("grain = 'month' AND ABS(share_of_month_pct - 100) > 0.01"),
    ),
)


def evaluate(spark: SparkSession, cfg: PipelineConfig, checks: Sequence[Check] = CHECKS) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        name = cfg.table(check.layer, check.table)
        if not table_exists(spark, name):
            results.append(CheckResult(check, failing_rows=-1, table_exists=False))
            continue
        results.append(CheckResult(check, check.failing_rows(spark.table(name)), table_exists=True))
    return results


def results_frame(spark: SparkSession, cfg: PipelineConfig, results: Sequence[CheckResult]) -> DataFrame:
    checked_at = datetime.now(tz=UTC)
    rows = [
        (
            r.check.name,
            r.check.layer,
            r.check.table,
            r.check.description,
            r.check.severity,
            r.table_exists,
            r.failing_rows,
            r.passed,
            cfg.run_id,
            checked_at,
        )
        for r in results
    ]
    return spark.createDataFrame(
        rows,
        "check_name string, layer string, table_name string, description string, severity string, "
        "table_exists boolean, failing_rows long, passed boolean, run_id string, checked_at timestamp",
    )


def run_quality(
    spark: SparkSession,
    cfg: PipelineConfig,
    checks: Sequence[Check] = CHECKS,
    raise_on_error: bool = True,
) -> list[CheckResult]:
    results = evaluate(spark, cfg, checks)
    write_append(results_frame(spark, cfg, results), cfg.table("ops", "data_quality_results"), cfg)
    hard_failures = [r for r in results if not r.passed and r.check.severity == "error"]
    if hard_failures and raise_on_error:
        detail = ", ".join(f"{r.check.name}({r.failing_rows})" for r in hard_failures)
        msg = f"{len(hard_failures)} data-quality check(s) failed: {detail}"
        raise DataQualityError(msg)
    return results
