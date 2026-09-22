"""gold_consumption_distribution_monthly — where each site sits in the population.

Windows: ``cume_dist``, ``percent_rank``, ``ntile(10)`` across all sites in a month, plus
``percentile_approx`` per segment (joined back) so every site can be compared to its
segment's median and 90th percentile.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.gold.common import SilverInputs, month_start


def build_consumption_distribution(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return consumption_distribution_monthly(inputs.readings_enriched)


def consumption_distribution_monthly(readings: DataFrame) -> DataFrame:
    site_month = (
        readings.withColumn("month_start", month_start("reading_hour"))
        .groupBy("site_id", "region", "customer_segment", "month_start")
        .agg(F.round(F.sum("kwh_filled"), 3).alias("kwh_total"))
    )
    segment_stats = site_month.groupBy("customer_segment", "month_start").agg(
        F.percentile_approx("kwh_total", 0.5).alias("segment_p50_kwh"),
        F.percentile_approx("kwh_total", 0.9).alias("segment_p90_kwh"),
        F.count("*").alias("segment_site_count"),
    )
    population = Window.partitionBy("month_start").orderBy("kwh_total")
    within_segment = Window.partitionBy("customer_segment", "month_start").orderBy("kwh_total")
    return (
        site_month.join(segment_stats, ["customer_segment", "month_start"], "left")
        .withColumn("cume_dist_all_sites", F.round(F.cume_dist().over(population), 4))
        .withColumn("percent_rank_all_sites", F.round(F.percent_rank().over(population), 4))
        .withColumn("decile_all_sites", F.ntile(10).over(population))
        .withColumn("cume_dist_in_segment", F.round(F.cume_dist().over(within_segment), 4))
        .withColumn("rank_in_segment_desc", F.rank().over(within_segment.orderBy(F.col("kwh_total").desc())))
        .withColumn(
            "pct_of_segment_median",
            F.when(
                F.col("segment_p50_kwh") > 0,
                F.round(F.col("kwh_total") / F.col("segment_p50_kwh") * 100, 2),
            ),
        )
        .withColumn("is_above_segment_p90", F.col("kwh_total") > F.col("segment_p90_kwh"))
        .withColumn("is_top_decile", F.col("decile_all_sites") == 10)
    )
