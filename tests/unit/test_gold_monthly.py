from datetime import date

import pytest
from pyspark.sql import SparkSession

from energy_lakehouse.gold.monthly import site_monthly_kpis
from tests.conftest import by_key, ts

COLS = (
    "site_id string, region string, customer_segment string, reading_hour timestamp, "
    "kwh_filled double, kwh_exported double"
)


def _one_hour(site: str, region: str, day: str, kwh: float) -> tuple:  # type: ignore[type-arg]
    return (site, region, "residential", ts(f"{day}T12:00"), kwh, 0.0)


def test_ranking_and_distribution_functions(spark: SparkSession) -> None:
    rows = [
        _one_hour("A", "north", "2025-01-01", 10.0),
        _one_hour("B", "north", "2025-01-01", 20.0),
        _one_hour("C", "north", "2025-01-01", 20.0),
        _one_hour("D", "north", "2025-01-01", 40.0),
        _one_hour("E", "south", "2025-01-01", 5.0),
        _one_hour("A", "north", "2025-02-01", 15.0),
    ]
    kpis = by_key(site_monthly_kpis(spark, spark.createDataFrame(rows, COLS)), "site_id", "month_start")
    jan = date(2025, 1, 1)
    a, b, c, d, e = (kpis[(s, jan)] for s in "ABCDE")

    # RANK leaves a hole after the tie, DENSE_RANK does not, ROW_NUMBER is unique
    assert [r["rank_in_region"] for r in (d, b, c, a)] == [1, 2, 2, 4]
    assert e["rank_in_region"] == 1  # partitioned by region
    assert [r["dense_rank_overall"] for r in (d, b, c, a, e)] == [1, 2, 2, 3, 4]
    assert sorted(r["row_num_overall"] for r in (a, b, c, d, e)) == [1, 2, 3, 4, 5]

    # PERCENT_RANK = (rank - 1) / (n - 1), CUME_DIST = rows <= current / n, over 5 sites asc
    assert e["percent_rank_overall"] == 0.0 and a["percent_rank_overall"] == 0.25
    assert b["percent_rank_overall"] == c["percent_rank_overall"] == 0.5
    assert d["percent_rank_overall"] == 1.0
    assert e["cume_dist_overall"] == 0.2 and a["cume_dist_overall"] == 0.4
    assert b["cume_dist_overall"] == c["cume_dist_overall"] == 0.8 and d["cume_dist_overall"] == 1.0

    # NTILE(4) over 5 rows -> buckets of size 2,1,1,1 in ascending order (tie order is arbitrary)
    assert e["consumption_quartile"] == 1 and d["consumption_quartile"] == 4
    assert {b["consumption_quartile"], c["consumption_quartile"]} <= {2, 3}

    # aggregate-as-window: share of the region, FIRST_VALUE / LAST_VALUE with a full frame
    assert d["share_of_region_pct"] == pytest.approx(44.44) and a["share_of_region_pct"] == pytest.approx(11.11)
    assert all(r["region_top_site"] == "D" and r["region_bottom_site"] == "A" for r in (a, b, c, d))
    assert e["region_top_site"] == e["region_bottom_site"] == "E"
    assert d["kwh_vs_segment_avg"] == 21.0  # 40 - avg(10,20,20,40,5)=19

    # LAG / LEAD / running total across months for site A
    a_feb = kpis[("A", date(2025, 2, 1))]
    assert a["kwh_prev_month"] is None and a["kwh_next_month"] == 15.0
    assert a_feb["kwh_prev_month"] == 10.0 and a_feb["mom_change_pct"] == 50.0
    assert a["kwh_ytd"] == 10.0 and a_feb["kwh_ytd"] == 25.0
