"""gold_peak_demand_hours — the top-3 consumption hours of every site-day.

Window functions: ``rank`` vs ``dense_rank`` vs ``row_number`` on the same ordering (ties
behave differently), ``first_value`` / ``last_value`` over an explicit
``rowsBetween(unboundedPreceding, unboundedFollowing)`` frame, ``max`` as a window.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.gold.common import SilverInputs

TOP_N_HOURS = 3


def build_peak_demand_hours(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return peak_demand_hours(inputs.readings_enriched)


def peak_demand_hours(readings: DataFrame, top_n: int = TOP_N_HOURS) -> DataFrame:
    by_kwh_desc = Window.partitionBy("site_id", "reading_date").orderBy(
        F.col("kwh_filled").desc(), F.col("reading_hour")
    )
    by_kwh_desc_ties = Window.partitionBy("site_id", "reading_date").orderBy(F.col("kwh_filled").desc())
    whole_day = (
        Window.partitionBy("site_id", "reading_date")
        .orderBy("reading_hour")
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )
    return (
        readings.withColumn("rank_in_day", F.rank().over(by_kwh_desc_ties))
        .withColumn("dense_rank_in_day", F.dense_rank().over(by_kwh_desc_ties))
        .withColumn("row_number_in_day", F.row_number().over(by_kwh_desc))
        .withColumn("day_peak_kwh", F.max("kwh_filled").over(whole_day))
        .withColumn("first_hour_kwh", F.first_value("kwh_filled").over(whole_day))
        .withColumn("last_hour_kwh", F.last_value("kwh_filled").over(whole_day))
        .withColumn(
            "pct_of_day_peak",
            F.when(
                F.col("day_peak_kwh") > 0,
                F.round(F.col("kwh_filled") / F.col("day_peak_kwh") * 100, 2),
            ),
        )
        .withColumn(
            "contract_utilisation_pct",
            F.when(
                F.col("contracted_kw") > 0,
                F.round(F.col("kwh_filled") / F.col("contracted_kw") * 100, 2),
            ),
        )
        .withColumn("is_over_contract", F.col("kwh_filled") > F.col("contracted_kw"))
        .filter(F.col("row_number_in_day") <= top_n)
        .select(
            "site_id",
            "region",
            "customer_segment",
            "reading_date",
            "reading_hour",
            "hour_of_day",
            "is_weekend",
            "kwh_filled",
            "rank_in_day",
            "dense_rank_in_day",
            "row_number_in_day",
            "day_peak_kwh",
            "pct_of_day_peak",
            "first_hour_kwh",
            "last_hour_kwh",
            "contracted_kw",
            "contract_utilisation_pct",
            "is_over_contract",
        )
    )
