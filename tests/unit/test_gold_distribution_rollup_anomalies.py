from datetime import date, timedelta

from pyspark.sql import SparkSession

from energy_lakehouse.gold.anomalies import consumption_anomalies
from energy_lakehouse.gold.distribution import consumption_distribution_monthly
from energy_lakehouse.gold.rollup import consumption_rollup_monthly
from tests.conftest import by_key, rows_to_dicts, ts

COLS = (
    "site_id string, region string, customer_segment string, tariff_plan string, reading_date date, "
    "reading_hour timestamp, hour_of_day int, kwh_filled double, kwh_exported double, kwh_prev_hour double, "
    "quality_flag string"
)


def _row(site: str, region: str, segment: str, tariff: str, hour: str, kwh: float) -> tuple:  # type: ignore[type-arg]
    t = ts(hour)
    return (site, region, segment, tariff, t.date(), t, t.hour, kwh, 0.0, None, "OK")


def test_distribution_positions(spark: SparkSession) -> None:
    rows = [_row(f"S{i}", "north", "residential", "flat", "2025-01-01T00:00", float(i)) for i in range(1, 11)]
    dist = by_key(consumption_distribution_monthly(spark.createDataFrame(rows, COLS)), "site_id")
    for i in range(1, 11):
        r = dist[(f"S{i}",)]
        assert r["decile_all_sites"] == i
        assert r["cume_dist_all_sites"] == round(i / 10, 4)
        assert r["percent_rank_all_sites"] == round((i - 1) / 9, 4)
        assert r["cume_dist_in_segment"] == r["cume_dist_all_sites"]
        assert r["rank_in_segment_desc"] == 11 - i
    assert dist[("S10",)]["is_top_decile"] is True and dist[("S9",)]["is_top_decile"] is False
    assert dist[("S1",)]["segment_site_count"] == 10
    assert 5.0 <= dist[("S1",)]["segment_p50_kwh"] <= 6.0
    assert sum(r["is_above_segment_p90"] for r in dist.values()) <= 1


def test_grouping_sets_rollup(spark: SparkSession) -> None:
    rows = [
        _row("A", "north", "residential", "flat", "2025-01-01T00:00", 10.0),
        _row("B", "north", "commercial", "flat", "2025-01-01T00:00", 20.0),
        _row("C", "south", "residential", "time_of_use", "2025-01-01T00:00", 30.0),
        _row("D", "south", "commercial", "time_of_use", "2025-01-01T00:00", 40.0),
    ]
    out = rows_to_dicts(consumption_rollup_monthly(spark, spark.createDataFrame(rows, COLS)))
    assert len(out) == 4 + 4 + 2 + 1
    by_grain = {}
    for r in out:
        by_grain.setdefault(r["grain"], []).append(r)
    (month,) = by_grain["month"]
    assert month["kwh_total"] == 100.0 and month["share_of_month_pct"] == 100.0 and month["site_count"] == 4
    assert month["region"] is None and month["grouping_level"] == 7
    regions = {r["region"]: r for r in by_grain["region"]}
    assert regions["north"]["kwh_total"] == 30.0 and regions["north"]["share_of_month_pct"] == 30.0
    assert regions["south"]["rank_within_grain"] == 1 and regions["north"]["rank_within_grain"] == 2
    segs = {(r["region"], r["customer_segment"]): r["kwh_total"] for r in by_grain["segment"]}
    assert segs[("north", "residential")] == 10.0 and segs[("south", "commercial")] == 40.0
    assert all(r["grouping_level"] == 0 for r in by_grain["tariff"])


def test_anomaly_detection_against_trailing_baseline(spark: SparkSession) -> None:
    start = ts("2025-01-01T00:00")
    rows = []
    for h in range(220):
        kwh = 9.0 if h % 2 else 11.0  # mean 10, small variance
        if h == 150:
            kwh = 0.0  # DROP
        if h == 190:
            kwh = 100.0  # SPIKE
        t = start + timedelta(hours=h)
        rows.append(("S1", "north", "residential", "flat", t.date(), t, t.hour, kwh, 0.0, None, "OK"))
    found = rows_to_dicts(consumption_anomalies(spark.createDataFrame(rows, COLS)), "reading_hour")
    assert [(a["reading_hour"], a["anomaly_kind"]) for a in found] == [
        (start + timedelta(hours=150), "DROP"),
        (start + timedelta(hours=190), "SPIKE"),
    ]
    spike = found[1]
    assert spike["z_score"] > 3 and spike["baseline_rows"] == 168 and spike["anomaly_rank_in_day"] == 1
    assert spike["kwh_next_hour"] == 9.0 and spike["reading_date"] == date(2025, 1, 8)
    assert found[0]["z_score"] < -3
