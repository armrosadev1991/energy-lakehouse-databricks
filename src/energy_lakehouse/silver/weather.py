"""Silver weather: hourly, de-duplicated, forward-filled, with degree-hour features.

Window functions: ``row_number`` (dedupe), ``last(ignorenulls)`` (forward-fill),
``lag(24)`` (same hour yesterday), ``date_trunc('hour')``.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from energy_lakehouse.silver.casting import try_double

WEATHER_KEYS = ("region", "observed_hour")
HEATING_BASE_C = 18.0
COOLING_BASE_C = 24.0


def build_silver_weather(bronze: DataFrame) -> DataFrame:
    typed = bronze.select(
        F.lower(F.trim("region")).alias("region"),
        F.date_trunc("hour", F.try_to_timestamp(F.col("observed_ts"))).alias("observed_hour"),
        try_double("temperature_c").alias("temperature_c"),
        try_double("wind_speed_ms").alias("wind_speed_ms"),
        try_double("solar_irradiance_wm2").alias("solar_irradiance_wm2"),
        try_double("humidity_pct").alias("humidity_pct"),
        "_ingested_at",
    ).filter(F.col("region").isNotNull() & F.col("observed_hour").isNotNull())

    latest = Window.partitionBy(*WEATHER_KEYS).orderBy(F.col("_ingested_at").desc())
    series = Window.partitionBy("region").orderBy("observed_hour")
    history = series.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return (
        typed.withColumn("_rn", F.row_number().over(latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_ingested_at")
        .withColumn("temperature_c", F.last("temperature_c", ignorenulls=True).over(history))
        .withColumn("temperature_prev_day_c", F.lag("temperature_c", 24).over(series))
        .withColumn(
            "heating_degree_hours",
            F.greatest(F.lit(HEATING_BASE_C) - F.col("temperature_c"), F.lit(0.0)),
        )
        .withColumn(
            "cooling_degree_hours",
            F.greatest(F.col("temperature_c") - F.lit(COOLING_BASE_C), F.lit(0.0)),
        )
        .withColumn("observed_date", F.to_date("observed_hour"))
    )
