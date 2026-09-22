from datetime import date

from pyspark.sql import SparkSession

from energy_lakehouse.gold.outages import outage_streaks
from energy_lakehouse.gold.peaks import peak_demand_hours
from tests.conftest import by_key, rows_to_dicts, ts

PEAK_COLS = (
    "site_id string, region string, customer_segment string, reading_date date, "
    "reading_hour timestamp, hour_of_day int, is_weekend boolean, kwh_filled double, contracted_kw double"
)
OUTAGE_COLS = "site_id string, reading_hour timestamp, kwh_filled double, quality_flag string, gap_hours_before double"


def test_rank_vs_dense_rank_vs_row_number(spark: SparkSession) -> None:
    kwh = {0: 5.0, 1: 9.0, 2: 9.0, 3: 2.0, 4: 7.0}
    rows = [
        (
            "S1",
            "north",
            "residential",
            date(2025, 1, 1),
            ts(f"2025-01-01T0{h}:00"),
            h,
            False,
            v,
            8.0,
        )
        for h, v in kwh.items()
    ]
    top = by_key(peak_demand_hours(spark.createDataFrame(rows, PEAK_COLS)), "site_id", "hour_of_day")

    assert sorted(h for _, h in top) == [1, 2, 4]  # top-3 by row_number
    h1, h2, h4 = top[("S1", 1)], top[("S1", 2)], top[("S1", 4)]
    assert (h1["rank_in_day"], h2["rank_in_day"], h4["rank_in_day"]) == (1, 1, 3)
    assert (h1["dense_rank_in_day"], h2["dense_rank_in_day"], h4["dense_rank_in_day"]) == (1, 1, 2)
    assert (h1["row_number_in_day"], h2["row_number_in_day"], h4["row_number_in_day"]) == (1, 2, 3)
    assert h4["first_hour_kwh"] == 5.0 and h4["last_hour_kwh"] == 7.0 and h4["day_peak_kwh"] == 9.0
    assert h4["pct_of_day_peak"] == 77.78
    assert h1["contract_utilisation_pct"] == 112.5 and h1["is_over_contract"] is True
    assert h4["is_over_contract"] is False


def test_gaps_and_islands(spark: SparkSession) -> None:
    kwh = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # hours 0..9: zero runs at 1-3 and 6
    rows = [("S1", ts(f"2025-01-01T0{h}:00"), float(v), "OK", 0.0) for h, v in enumerate(kwh)]
    rows.append(("S1", ts("2025-01-01T12:00"), 1.0, "OK", 2.0))  # hours 10-11 missing
    rows.append(("S1", ts("2025-01-01T20:00"), 0.0, "OUTAGE", 7.0))  # 13-19 missing, 20 is zero
    rows.append(("S1", ts("2025-01-01T23:00"), 0.0, "OK", 2.0))  # 21-22 missing, 23 is zero
    rows.append(("S2", ts("2025-01-01T00:00"), 0.0, "OK", 0.0))
    streaks = rows_to_dicts(outage_streaks(spark.createDataFrame(rows, OUTAGE_COLS)), "site_id", "streak_start")
    s1 = [s for s in streaks if s["site_id"] == "S1"]

    assert [(s["streak_kind"], s["streak_start"].hour, s["streak_end"].hour, s["streak_hours"]) for s in s1] == [
        ("ZERO_CONSUMPTION", 1, 3, 3),
        ("ZERO_CONSUMPTION", 6, 6, 1),
        ("MISSING_DATA", 10, 11, 2),
        ("MISSING_DATA", 13, 19, 7),
        (
            "ZERO_CONSUMPTION",
            20,
            20,
            1,
        ),  # separated from hour 23 by missing rows -> distinct islands
        ("MISSING_DATA", 21, 22, 2),
        ("ZERO_CONSUMPTION", 23, 23, 1),
    ]
    assert [s["streak_seq"] for s in s1] == list(range(1, 8))
    ranks = {(s["streak_start"].hour): s["streak_rank_by_duration"] for s in s1}
    assert ranks[13] == 1 and ranks[1] == 2 and ranks[10] == ranks[21] == 3 and ranks[6] == 4
    assert [s["is_longest_streak"] for s in s1].count(True) == 1
    first = s1[0]
    assert first["hours_since_prev_streak"] is None and first["hours_until_next_streak"] == 2.0  # hours 4,5
    assert s1[1]["hours_since_prev_streak"] == 2.0
    s2 = [s for s in streaks if s["site_id"] == "S2"]
    assert len(s2) == 1 and s2[0]["streak_kind"] == "ZERO_CONSUMPTION" and s2[0]["is_longest_streak"] is True
