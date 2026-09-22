"""gold_weather_sensitivity_monthly — how strongly each site's load follows the weather.

Aggregates: Pearson correlation and the OLS slope/intercept of kWh on temperature, built
from ``covar_samp`` / ``var_samp`` / ``stddev_samp`` with ``try_divide``. (Spark's built-in
``corr``/``regr_slope`` raise DIVIDE_BY_ZERO under ANSI mode when one series is constant,
which a non-solar site's export column is.) Windows: ``lag`` on the monthly correlation,
``rank`` within segment-month.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from energy_lakehouse.gold.common import SilverInputs, month_start


def pearson(y: str, x: str) -> Column:
    """ANSI-safe Pearson correlation: null (not an error) when either series is constant."""
    return F.try_divide(F.covar_samp(y, x), F.stddev_samp(y) * F.stddev_samp(x))


def ols_slope(y: str, x: str) -> Column:
    return F.try_divide(F.covar_samp(y, x), F.var_samp(x))


def ols_intercept(y: str, x: str) -> Column:
    return F.avg(y) - ols_slope(y, x) * F.avg(x)


def build_weather_sensitivity(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return weather_sensitivity_monthly(inputs.readings_enriched, inputs.weather)


def weather_sensitivity_monthly(readings: DataFrame, weather: DataFrame) -> DataFrame:
    w = weather.select(
        F.col("region").alias("w_region"),
        F.col("observed_hour").alias("w_hour"),
        "temperature_c",
        "solar_irradiance_wm2",
        "heating_degree_hours",
        "cooling_degree_hours",
    )
    joined = readings.join(
        w,
        (F.col("region") == F.col("w_region")) & (F.col("reading_hour") == F.col("w_hour")),
        "inner",
    ).withColumn("month_start", month_start("reading_hour"))

    monthly = joined.groupBy("site_id", "region", "customer_segment", "month_start").agg(
        F.round(pearson("kwh_filled", "temperature_c"), 4).alias("temp_correlation"),
        F.round(F.covar_samp("kwh_filled", "temperature_c"), 4).alias("temp_covariance"),
        F.round(ols_slope("kwh_filled", "temperature_c"), 5).alias("kwh_per_degree_c"),
        F.round(ols_intercept("kwh_filled", "temperature_c"), 4).alias("kwh_at_zero_c"),
        F.round(pearson("kwh_exported", "solar_irradiance_wm2"), 4).alias("export_irradiance_correlation"),
        F.round(F.sum("heating_degree_hours"), 2).alias("heating_degree_hours"),
        F.round(F.sum("cooling_degree_hours"), 2).alias("cooling_degree_hours"),
        F.round(F.sum(F.when(F.col("heating_degree_hours") > 0, F.col("kwh_filled")).otherwise(0.0)), 3).alias(
            "kwh_during_heating_hours"
        ),
        F.round(F.avg("temperature_c"), 2).alias("avg_temperature_c"),
        F.count("*").alias("hours_matched"),
    )
    site_ts = Window.partitionBy("site_id").orderBy("month_start")
    most_sensitive = Window.partitionBy("customer_segment", "month_start").orderBy(
        F.abs(F.col("temp_correlation")).desc_nulls_last()
    )
    return (
        monthly.withColumn(
            "kwh_per_heating_degree_hour",
            F.when(
                F.col("heating_degree_hours") > 0,
                F.round(F.col("kwh_during_heating_hours") / F.col("heating_degree_hours"), 4),
            ),
        )
        .withColumn("temp_correlation_prev_month", F.lag("temp_correlation").over(site_ts))
        .withColumn(
            "temp_correlation_shift",
            F.round(F.col("temp_correlation") - F.col("temp_correlation_prev_month"), 4),
        )
        .withColumn("sensitivity_rank_in_segment", F.rank().over(most_sensitive))
    )
