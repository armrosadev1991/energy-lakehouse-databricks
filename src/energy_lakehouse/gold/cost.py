"""gold_energy_cost_daily — consumption priced at the day-ahead market, per site-day.

Window functions: ``lag`` (cost day-over-day), ``sum ... rowsBetween(unboundedPreceding,
currentRow)`` (month-to-date spend), ``rank`` (most expensive days per month),
``ntile(4)`` (cost quartile among sites in the same region-day).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.gold.common import SilverInputs, month_start, pct_change


def build_energy_cost_daily(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return energy_cost_daily(inputs.readings_enriched, inputs.prices)


def price_readings(readings: DataFrame, prices: DataFrame) -> DataFrame:
    p = prices.filter(F.col("market") == "day_ahead").select(
        F.col("region").alias("p_region"),
        F.col("price_hour").alias("p_hour"),
        "price_eur_mwh",
        "is_peak_price_hour",
    )
    return readings.join(
        p,
        (F.col("region") == F.col("p_region")) & (F.col("reading_hour") == F.col("p_hour")),
        "left",
    ).withColumn("cost_eur", F.col("kwh_filled") / 1000.0 * F.col("price_eur_mwh"))


def energy_cost_daily(readings: DataFrame, prices: DataFrame) -> DataFrame:
    priced = price_readings(readings, prices)
    dims = ["site_id", "region", "customer_segment", "tariff_plan", "reading_date"]
    daily = priced.groupBy(*dims).agg(
        F.round(F.sum("cost_eur"), 4).alias("cost_eur"),
        F.round(F.sum("kwh_filled"), 3).alias("kwh_total"),
        F.round(F.avg("price_eur_mwh"), 2).alias("avg_price_eur_mwh"),
        F.round(
            F.sum(F.col("kwh_filled") * F.col("price_eur_mwh")) / F.nullif(F.sum("kwh_filled"), F.lit(0.0)),
            2,
        ).alias("kwh_weighted_price_eur_mwh"),
        F.round(F.sum(F.when(F.col("is_peak_price_hour"), F.col("kwh_filled")).otherwise(0.0)), 3).alias(
            "kwh_in_peak_price_hours"
        ),
        F.sum(F.col("price_eur_mwh").isNull().cast("int")).alias("hours_without_price"),
    )
    series = Window.partitionBy("site_id").orderBy("reading_date")
    mtd = (
        Window.partitionBy("site_id", month_start("reading_date"))
        .orderBy("reading_date")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    costliest = Window.partitionBy("site_id", month_start("reading_date")).orderBy(F.col("cost_eur").desc())
    region_day = Window.partitionBy("region", "reading_date").orderBy("cost_eur")
    prev = F.lag("cost_eur").over(series)
    return (
        daily.withColumn(
            "peak_price_kwh_share_pct",
            F.when(
                F.col("kwh_total") > 0,
                F.round(F.col("kwh_in_peak_price_hours") / F.col("kwh_total") * 100, 2),
            ),
        )
        .withColumn("cost_prev_day", prev)
        .withColumn("cost_dod_change_pct", pct_change(F.col("cost_eur"), prev))
        .withColumn("cost_mtd_eur", F.round(F.sum("cost_eur").over(mtd), 4))
        .withColumn("cost_rank_in_month", F.rank().over(costliest))
        .withColumn("cost_quartile_in_region_day", F.ntile(4).over(region_day))
    )
