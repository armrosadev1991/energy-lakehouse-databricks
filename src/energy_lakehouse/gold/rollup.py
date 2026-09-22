"""gold_consumption_rollup_monthly — hierarchical totals with GROUPING SETS.

One query returns month totals at four grain levels (month; month+region; month+region+
segment; month+region+segment+tariff). ``grouping_id()`` (bit per grouping column, 1 when the
column is rolled up: 0 = tariff grain, 1 = segment, 3 = region, 7 = month) tells each row's level, and a
``sum`` window turns every row into a share of its month's grand total.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from energy_lakehouse.gold.common import SilverInputs

VIEW = "_gold_input_rollup"

ROLLUP_SQL = f"""
WITH base AS (
    SELECT
        CAST(date_trunc('month', reading_hour) AS DATE) AS month_start,
        region,
        customer_segment,
        tariff_plan,
        site_id,
        kwh_filled,
        kwh_exported
    FROM {VIEW}
),
grouped AS (
    SELECT
        month_start,
        region,
        customer_segment,
        tariff_plan,
        GROUPING_ID(month_start, region, customer_segment, tariff_plan) AS grouping_level,
        ROUND(SUM(kwh_filled), 3)                         AS kwh_total,
        ROUND(SUM(kwh_exported), 3)                       AS kwh_exported_total,
        COUNT(DISTINCT site_id)                           AS site_count,
        ROUND(AVG(kwh_filled), 4)                         AS kwh_avg_hour
    FROM base
    GROUP BY month_start, GROUPING SETS (
        (region, customer_segment, tariff_plan),
        (region, customer_segment),
        (region),
        ()
    )
)
SELECT
    *,
    CASE grouping_level
        WHEN 0 THEN 'tariff'
        WHEN 1 THEN 'segment'
        WHEN 3 THEN 'region'
        WHEN 7 THEN 'month'
    END                                                                  AS grain,
    ROUND(try_divide(kwh_total, MAX(CASE WHEN grouping_level = 7 THEN kwh_total END)
                        OVER (PARTITION BY month_start)) * 100, 2)        AS share_of_month_pct,
    RANK() OVER (PARTITION BY month_start, grouping_level ORDER BY kwh_total DESC)
                                                                         AS rank_within_grain
FROM grouped
"""


def build_consumption_rollup(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return consumption_rollup_monthly(spark, inputs.readings_enriched)


def consumption_rollup_monthly(spark: SparkSession, readings_enriched: DataFrame) -> DataFrame:
    readings_enriched.createOrReplaceTempView(VIEW)
    return spark.sql(ROLLUP_SQL)
