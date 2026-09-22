"""gold_consumption_anomalies — hours whose consumption is far from the trailing baseline.

Windows: ``avg`` / ``stddev_samp`` / ``count`` over ``rowsBetween(-168, -1)`` (the previous
7 days of hourly rows, excluding the current one), ``lead`` for the following hour, ``rank``
of anomalies within each site-day by severity.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.gold.common import SilverInputs

BASELINE_HOURS = 168
MIN_BASELINE_ROWS = 24
Z_THRESHOLD = 3.0


def build_consumption_anomalies(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return consumption_anomalies(inputs.readings_enriched)


def score_readings(readings: DataFrame, baseline_hours: int = BASELINE_HOURS) -> DataFrame:
    series = Window.partitionBy("site_id").orderBy("reading_hour")
    trailing = series.rowsBetween(-baseline_hours, -1)
    return (
        readings.withColumn("baseline_mean_kwh", F.avg("kwh_filled").over(trailing))
        .withColumn("baseline_std_kwh", F.stddev_samp("kwh_filled").over(trailing))
        .withColumn("baseline_rows", F.count("kwh_filled").over(trailing))
        .withColumn(
            "z_score",
            F.when(
                (F.col("baseline_rows") >= MIN_BASELINE_ROWS) & (F.col("baseline_std_kwh") > 0),
                F.round(
                    (F.col("kwh_filled") - F.col("baseline_mean_kwh")) / F.col("baseline_std_kwh"),
                    3,
                ),
            ),
        )
        .withColumn("kwh_next_hour", F.lead("kwh_filled").over(series))
    )


def consumption_anomalies(readings: DataFrame, z_threshold: float = Z_THRESHOLD) -> DataFrame:
    scored = score_readings(readings).filter(F.abs(F.col("z_score")) > z_threshold)
    severity = Window.partitionBy("site_id", "reading_date").orderBy(F.abs(F.col("z_score")).desc())
    return (
        scored.withColumn("anomaly_kind", F.when(F.col("z_score") > 0, "SPIKE").otherwise("DROP"))
        .withColumn("anomaly_rank_in_day", F.rank().over(severity))
        .withColumn("anomalies_in_day", F.count("*").over(Window.partitionBy("site_id", "reading_date")))
        .select(
            "site_id",
            "region",
            "customer_segment",
            "reading_date",
            "reading_hour",
            "hour_of_day",
            "kwh_filled",
            "kwh_prev_hour",
            "kwh_next_hour",
            "baseline_mean_kwh",
            "baseline_std_kwh",
            "baseline_rows",
            "z_score",
            "anomaly_kind",
            "anomaly_rank_in_day",
            "anomalies_in_day",
            "quality_flag",
        )
    )
