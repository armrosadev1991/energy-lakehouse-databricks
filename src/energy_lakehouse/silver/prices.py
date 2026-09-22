"""Silver day-ahead prices with hour-over-hour change and intraday price ranking.

Window functions: ``row_number`` (dedupe), ``lag`` (previous hour), ``rank`` over the
day ordered by price (most expensive hours first).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from energy_lakehouse.silver.casting import try_double

PRICE_KEYS = ("region", "market", "price_hour")
PEAK_PRICE_HOURS_PER_DAY = 4


def build_silver_prices(bronze: DataFrame) -> DataFrame:
    typed = bronze.select(
        F.lower(F.trim("region")).alias("region"),
        F.lower(F.trim("market")).alias("market"),
        F.date_trunc("hour", F.try_to_timestamp(F.col("price_ts"))).alias("price_hour"),
        try_double("price_eur_mwh").alias("price_eur_mwh"),
        "_ingested_at",
    ).filter(F.col("price_hour").isNotNull() & F.col("price_eur_mwh").isNotNull())

    latest = Window.partitionBy(*PRICE_KEYS).orderBy(F.col("_ingested_at").desc())
    series = Window.partitionBy("region", "market").orderBy("price_hour")
    intraday = Window.partitionBy("region", "market", F.to_date("price_hour")).orderBy(F.col("price_eur_mwh").desc())
    prev = F.lag("price_eur_mwh").over(series)
    return (
        typed.withColumn("_rn", F.row_number().over(latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_ingested_at")
        .withColumn("price_date", F.to_date("price_hour"))
        .withColumn("price_prev_hour", prev)
        .withColumn(
            "price_change_pct",
            F.when(prev != 0, F.round((F.col("price_eur_mwh") - prev) / F.abs(prev) * 100, 2)),
        )
        .withColumn("price_rank_in_day", F.rank().over(intraday))
        .withColumn("is_peak_price_hour", F.col("price_rank_in_day") <= PEAK_PRICE_HOURS_PER_DAY)
    )
