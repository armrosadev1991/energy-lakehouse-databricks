"""gold_site_monthly_kpis — written in Spark SQL to show the same ideas in SQL syntax.

Window functions: LAG, SUM ... ROWS UNBOUNDED PRECEDING (year-to-date), RANK, DENSE_RANK,
ROW_NUMBER, PERCENT_RANK, CUME_DIST, NTILE, SUM as a window (share of region), FIRST_VALUE,
LAST_VALUE with an explicit full frame, AVG as a window, plus a named WINDOW clause.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from energy_lakehouse.gold.common import SilverInputs

VIEW = "_gold_input_readings_enriched"

MONTHLY_KPI_SQL = f"""
WITH monthly AS (
    SELECT
        site_id,
        region,
        customer_segment,
        CAST(date_trunc('month', reading_hour) AS DATE)      AS month_start,
        ROUND(SUM(kwh_filled), 3)                             AS kwh_total,
        ROUND(AVG(kwh_filled), 3)                             AS kwh_avg_hour,
        ROUND(MAX(kwh_filled), 3)                             AS kwh_peak_hour,
        ROUND(SUM(kwh_exported), 3)                           AS kwh_exported_total,
        COUNT(*)                                              AS hours_observed
    FROM {VIEW}
    GROUP BY site_id, region, customer_segment, date_trunc('month', reading_hour)
)
SELECT
    *,
    LAG(kwh_total) OVER site_ts                                              AS kwh_prev_month,
    ROUND((kwh_total - LAG(kwh_total) OVER site_ts)
          / NULLIF(ABS(LAG(kwh_total) OVER site_ts), 0) * 100, 2)            AS mom_change_pct,
    LEAD(kwh_total) OVER site_ts                                             AS kwh_next_month,
    ROUND(SUM(kwh_total) OVER (
        PARTITION BY site_id, YEAR(month_start)
        ORDER BY month_start
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 3)                AS kwh_ytd,
    RANK()         OVER region_desc                                          AS rank_in_region,
    DENSE_RANK()   OVER (PARTITION BY month_start ORDER BY kwh_total DESC)   AS dense_rank_overall,
    ROW_NUMBER()   OVER (PARTITION BY month_start
                         ORDER BY kwh_total DESC, site_id)                   AS row_num_overall,
    ROUND(PERCENT_RANK() OVER month_asc, 4)                                  AS percent_rank_overall,
    ROUND(CUME_DIST()    OVER month_asc, 4)                                  AS cume_dist_overall,
    NTILE(4)       OVER month_asc                                            AS consumption_quartile,
    ROUND(try_divide(kwh_total, SUM(kwh_total) OVER (PARTITION BY month_start, region)) * 100, 2)
                                                                             AS share_of_region_pct,
    FIRST_VALUE(site_id) OVER region_desc                                    AS region_top_site,
    LAST_VALUE(site_id)  OVER (PARTITION BY month_start, region
                               ORDER BY kwh_total DESC
                               ROWS BETWEEN UNBOUNDED PRECEDING
                                        AND UNBOUNDED FOLLOWING)             AS region_bottom_site,
    ROUND(kwh_total - AVG(kwh_total) OVER (PARTITION BY month_start, customer_segment), 3)
                                                                             AS kwh_vs_segment_avg
FROM monthly
WINDOW
    site_ts     AS (PARTITION BY site_id ORDER BY month_start),
    region_desc AS (PARTITION BY month_start, region ORDER BY kwh_total DESC),
    month_asc   AS (PARTITION BY month_start ORDER BY kwh_total)
"""


def build_site_monthly_kpis(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return site_monthly_kpis(spark, inputs.readings_enriched)


def site_monthly_kpis(spark: SparkSession, readings_enriched: DataFrame) -> DataFrame:
    readings_enriched.createOrReplaceTempView(VIEW)
    return spark.sql(MONTHLY_KPI_SQL)
