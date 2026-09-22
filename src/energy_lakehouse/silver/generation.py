"""Silver grid generation mix, long format, with each source's share of the regional total.

Window functions: ``row_number`` (dedupe), ``sum() over (partition by region, hour)`` —
an *aggregate as a window* so every row keeps its detail while seeing the group total.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from energy_lakehouse.silver.casting import try_double

GENERATION_KEYS = ("region", "generation_hour", "source")
RENEWABLE_SOURCES = ("solar", "wind", "hydro")


def build_silver_generation(bronze: DataFrame) -> DataFrame:
    typed = bronze.select(
        F.lower(F.trim("region")).alias("region"),
        F.date_trunc("hour", F.try_to_timestamp(F.col("generation_ts"))).alias("generation_hour"),
        F.lower(F.trim("source")).alias("source"),
        try_double("generation_mw").alias("generation_mw"),
        "_ingested_at",
    ).filter(F.col("generation_hour").isNotNull() & F.col("generation_mw").isNotNull())

    latest = Window.partitionBy(*GENERATION_KEYS).orderBy(F.col("_ingested_at").desc())
    hour_total = Window.partitionBy("region", "generation_hour")
    return (
        typed.withColumn("_rn", F.row_number().over(latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_ingested_at")
        .withColumn("region_total_mw", F.sum("generation_mw").over(hour_total))
        .withColumn(
            "share_pct",
            F.when(
                F.col("region_total_mw") > 0,
                F.round(F.col("generation_mw") / F.col("region_total_mw") * 100, 2),
            ),
        )
        .withColumn("is_renewable", F.col("source").isin(*RENEWABLE_SOURCES))
        .withColumn("generation_date", F.to_date("generation_hour"))
    )
