"""gold_site_daily_consumption — one row per site per day.

Window functions
----------------
* ``lag(1)`` / ``lag(7)`` / ``lead(1)`` — day-over-day, week-over-week, next-day
* ``avg() ... rowsBetween(-6, 0)`` — 7-row moving average
* ``avg() ... rangeBetween(-29, 0)`` on a day-number column — true 30-*calendar*-day
  average that is not fooled by missing days (contrast with the ROWS frame above)
* ``sum() ... rowsBetween(unboundedPreceding, currentRow)`` — month-to-date running total
* ``row_number()`` — pick the peak hour of each day
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.gold.common import SilverInputs, month_start, pct_change

EPOCH = "1970-01-01"


def build_site_daily_consumption(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return site_daily_consumption(inputs.readings_enriched)


def site_daily_consumption(readings: DataFrame) -> DataFrame:
    dims = ["site_id", "region", "customer_segment", "tariff_plan", "reading_date"]
    daily = readings.groupBy(*dims).agg(
        F.round(F.sum("kwh_filled"), 3).alias("kwh_total"),
        F.round(F.max("kwh_filled"), 3).alias("kwh_peak_hour"),
        F.round(F.avg("kwh_filled"), 3).alias("kwh_avg_hour"),
        F.round(F.min("kwh_filled"), 3).alias("kwh_min_hour"),
        F.round(F.sum("kwh_exported"), 3).alias("kwh_exported_total"),
        F.count("*").alias("hours_observed"),
        F.sum(F.col("is_imputed").cast("int")).alias("hours_imputed"),
        F.sum("gap_hours_before").alias("hours_missing"),
    )

    peak_pick = Window.partitionBy("site_id", "reading_date").orderBy(F.col("kwh_filled").desc(), F.col("reading_hour"))
    peak_hours = (
        readings.withColumn("_rn", F.row_number().over(peak_pick))
        .filter(F.col("_rn") == 1)
        .select("site_id", "reading_date", F.col("hour_of_day").alias("peak_hour_of_day"))
    )

    series = Window.partitionBy("site_id").orderBy("reading_date")
    day_num = F.datediff(F.col("reading_date"), F.lit(EPOCH))
    calendar_30d = Window.partitionBy("site_id").orderBy(day_num).rangeBetween(-29, 0)
    month_to_date = (
        Window.partitionBy("site_id", month_start("reading_date"))
        .orderBy("reading_date")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    prev_day = F.lag("kwh_total", 1).over(series)
    prev_week = F.lag("kwh_total", 7).over(series)
    return (
        daily.join(peak_hours, ["site_id", "reading_date"], "left")
        .withColumn("load_factor", F.round(F.try_divide(F.col("kwh_avg_hour"), F.col("kwh_peak_hour")), 4))
        .withColumn("kwh_prev_day", prev_day)
        .withColumn("dod_change_pct", pct_change(F.col("kwh_total"), prev_day))
        .withColumn("kwh_same_day_prev_week", prev_week)
        .withColumn("wow_change_pct", pct_change(F.col("kwh_total"), prev_week))
        .withColumn("kwh_next_day", F.lead("kwh_total", 1).over(series))
        .withColumn("kwh_ma_7d", F.round(F.avg("kwh_total").over(series.rowsBetween(-6, 0)), 3))
        .withColumn("kwh_ma_30d_calendar", F.round(F.avg("kwh_total").over(calendar_30d), 3))
        .withColumn("kwh_mtd_cumulative", F.round(F.sum("kwh_total").over(month_to_date), 3))
        .withColumn("day_in_month_seq", F.row_number().over(month_to_date))
    )
