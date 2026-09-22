"""gold_generation_kpis_hourly — regional grid mix KPIs with a pivot to wide format.

Windows: ``avg ... rowsBetween(-23, 0)`` (rolling 24 h), ``lag(24)`` (same hour yesterday),
``ntile(10)`` and ``cume_dist`` of renewable share within region-month, ``max_by`` aggregate
for the dominant source, ``pivot`` to one column per source.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.datagen.generator import SOURCES
from energy_lakehouse.gold.common import SilverInputs, month_start


def build_generation_kpis(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return generation_kpis_hourly(inputs.generation)


def generation_kpis_hourly(generation: DataFrame) -> DataFrame:
    hourly = generation.groupBy("region", "generation_hour").agg(
        F.round(F.sum("generation_mw"), 1).alias("total_mw"),
        F.round(F.sum(F.when(F.col("is_renewable"), F.col("generation_mw")).otherwise(0.0)), 1).alias("renewable_mw"),
        F.max_by("source", "generation_mw").alias("dominant_source"),
        F.min_by("source", "generation_mw").alias("smallest_source"),
    )
    wide = (
        generation.groupBy("region", "generation_hour")
        .pivot("source", list(SOURCES))
        .agg(F.round(F.sum("generation_mw"), 1))
    )
    for s in SOURCES:
        wide = wide.withColumnRenamed(s, f"{s}_mw")

    series = Window.partitionBy("region").orderBy("generation_hour")
    rolling_24h = series.rowsBetween(-23, 0)
    region_month = Window.partitionBy("region", month_start("generation_hour")).orderBy("renewable_share_pct")
    share = F.round(F.col("renewable_mw") / F.nullif(F.col("total_mw"), F.lit(0.0)) * 100, 2)
    return (
        hourly.join(wide, ["region", "generation_hour"], "left")
        .withColumn("renewable_share_pct", share)
        .withColumn("renewable_share_ma_24h", F.round(F.avg("renewable_share_pct").over(rolling_24h), 2))
        .withColumn("renewable_share_same_hour_prev_day", F.lag("renewable_share_pct", 24).over(series))
        .withColumn(
            "renewable_share_dod_pts",
            F.round(F.col("renewable_share_pct") - F.col("renewable_share_same_hour_prev_day"), 2),
        )
        .withColumn("renewable_share_decile_in_month", F.ntile(10).over(region_month))
        .withColumn("renewable_share_cume_dist", F.round(F.cume_dist().over(region_month), 4))
        .withColumn("generation_date", F.to_date("generation_hour"))
    )
