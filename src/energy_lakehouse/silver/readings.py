"""Silver meter readings: typed, validated, hour-aligned, de-duplicated, gap-aware.

Window functions used here
--------------------------
* ``row_number()`` over ``(site_id, reading_hour)`` — keep the latest restatement
* ``last(..., ignorenulls=True)`` with ``rowsBetween(unboundedPreceding, currentRow)`` —
  forward-fill missing consumption
* ``lag()`` — previous hour's timestamp/consumption for gap detection and deltas
* ``date_trunc('hour', ...)`` — align readings that arrive a few minutes past the hour

All casts use ``try_*`` variants: under ANSI mode (default on Spark 4 / Databricks) a plain
``CAST`` or ``to_timestamp`` raises on malformed input instead of yielding NULL.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from energy_lakehouse.silver.casting import try_double

READING_KEYS = ("site_id", "reading_hour")


def cast_readings(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        F.col("reading_id"),
        F.trim("site_id").alias("site_id"),
        F.try_to_timestamp(F.col("reading_ts")).alias("reading_ts"),
        try_double("kwh_consumed").alias("kwh_consumed"),
        F.coalesce(try_double("kwh_exported"), F.lit(0.0)).alias("kwh_exported"),
        try_double("voltage_v").alias("voltage_v"),
        F.upper(F.trim("quality_flag")).alias("quality_flag"),
        "_source_file",
        "_ingested_at",
        "_batch_id",
    )


def split_valid_invalid(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return ``(valid, rejected)``; rejected rows carry a ``reject_reason``."""
    reason = (
        F.when(F.col("site_id").isNull() | (F.col("site_id") == ""), "missing_site_id")
        .when(F.col("reading_ts").isNull(), "unparseable_timestamp")
        .when(F.col("kwh_consumed") < 0, "negative_consumption")
    )
    tagged = df.withColumn("reject_reason", reason)
    return (
        tagged.filter(F.col("reject_reason").isNull()).drop("reject_reason"),
        tagged.filter(F.col("reject_reason").isNotNull()),
    )


def align_and_dedupe(df: DataFrame) -> DataFrame:
    """Truncate to the hour, then keep one row per ``(site_id, reading_hour)``.

    Precedence: most recently ingested file wins, then the highest ``reading_id``
    (source sequence numbers are monotonic, so a restatement always outranks the original
    even when both files were ingested in the same batch).
    """
    latest_first = Window.partitionBy(*READING_KEYS).orderBy(F.col("_ingested_at").desc(), F.col("reading_id").desc())
    return (
        df.withColumn("reading_hour", F.date_trunc("hour", F.col("reading_ts")))
        .withColumn("_rn", F.row_number().over(latest_first))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def enrich_time_series(df: DataFrame) -> DataFrame:
    """Forward-fill nulls and derive per-site lag features."""
    series = Window.partitionBy("site_id").orderBy("reading_hour")
    history = series.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    hours_between = (F.unix_timestamp("reading_hour") - F.unix_timestamp(F.lag("reading_hour").over(series))) / 3600.0
    return (
        df.withColumn(
            "kwh_filled",
            F.coalesce(
                F.col("kwh_consumed"),
                F.last("kwh_consumed", ignorenulls=True).over(history),
                F.lit(0.0),
            ),
        )
        .withColumn("is_imputed", F.col("kwh_consumed").isNull())
        .withColumn("gap_hours_before", F.greatest(hours_between - 1, F.lit(0.0)))
        .withColumn("kwh_prev_hour", F.lag("kwh_filled").over(series))
        .withColumn("kwh_delta_vs_prev", F.col("kwh_filled") - F.col("kwh_prev_hour"))
        .withColumn("reading_date", F.to_date("reading_hour"))
        .withColumn("hour_of_day", F.hour("reading_hour"))
        .withColumn("is_weekend", F.dayofweek("reading_hour").isin(1, 7))
    )


def build_silver_readings(bronze: DataFrame) -> DataFrame:
    valid, _rejected = split_valid_invalid(cast_readings(bronze))
    return enrich_time_series(align_and_dedupe(valid))


def rejected_readings(bronze: DataFrame) -> DataFrame:
    """Rows that failed validation, for the ops/quarantine table."""
    _valid, rejected = split_valid_invalid(cast_readings(bronze))
    return rejected
