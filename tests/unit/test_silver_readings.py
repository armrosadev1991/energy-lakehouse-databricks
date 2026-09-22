from pyspark.sql import SparkSession

from energy_lakehouse.schemas import FEEDS
from energy_lakehouse.silver.readings import build_silver_readings, rejected_readings
from tests.conftest import bronze_like, by_key, ts

COLS = FEEDS["meter_readings"]


def _bronze(spark: SparkSession):  # type: ignore[no-untyped-def]
    ok = {"kwh_exported": "0", "voltage_v": "230", "quality_flag": "OK"}
    return bronze_like(
        spark,
        COLS,
        [
            {
                **ok,
                "reading_id": "R1",
                "site_id": "S1",
                "reading_ts": "2025-01-01T00:00:00",
                "kwh_consumed": "1.0",
            },
            # arrives 3 minutes late -> aligned to 01:00
            {
                **ok,
                "reading_id": "R2",
                "site_id": "S1",
                "reading_ts": "2025-01-01T01:03:00",
                "kwh_consumed": "2.0",
            },
            # restatement of 01:00 in the same batch: higher reading_id wins
            {
                **ok,
                "reading_id": "R3",
                "site_id": " S1 ",
                "reading_ts": "2025-01-01T01:00:00",
                "kwh_consumed": "2.5",
                "quality_flag": "restated",
            },
            # missing value -> forward-filled from 01:00
            {
                **ok,
                "reading_id": "R4",
                "site_id": "S1",
                "reading_ts": "2025-01-01T02:00:00",
                "kwh_consumed": None,
                "quality_flag": "MISSING",
            },
            # hours 03 and 04 never arrive -> gap of 2 hours before 05:00
            {
                **ok,
                "reading_id": "R5",
                "site_id": "S1",
                "reading_ts": "2025-01-01T05:00:00",
                "kwh_consumed": "4.0",
            },
            # rejected rows
            {
                **ok,
                "reading_id": "R6",
                "site_id": "S1",
                "reading_ts": "2025-01-01T06:00:00",
                "kwh_consumed": "-3",
            },
            {
                **ok,
                "reading_id": "R7",
                "site_id": None,
                "reading_ts": "2025-01-01T06:00:00",
                "kwh_consumed": "1",
            },
            {
                **ok,
                "reading_id": "R8",
                "site_id": "S2",
                "reading_ts": "not a date",
                "kwh_consumed": "1",
            },
            # a second site, first row null -> falls back to 0.0
            {
                **ok,
                "reading_id": "R9",
                "site_id": "S2",
                "reading_ts": "2025-01-01T00:00:00",
                "kwh_consumed": None,
            },
        ],
    )


def test_dedupe_truncate_forward_fill_and_gaps(spark: SparkSession) -> None:
    silver = build_silver_readings(_bronze(spark))
    rows = by_key(silver, "site_id", "reading_hour")

    s1_hours = sorted(h for s, h in rows if s == "S1")
    assert s1_hours == [
        ts("2025-01-01T00:00"),
        ts("2025-01-01T01:00"),
        ts("2025-01-01T02:00"),
        ts("2025-01-01T05:00"),
    ]

    h1 = rows[("S1", ts("2025-01-01T01:00"))]
    assert h1["reading_id"] == "R3" and h1["kwh_consumed"] == 2.5 and h1["quality_flag"] == "RESTATED"

    h2 = rows[("S1", ts("2025-01-01T02:00"))]
    assert h2["kwh_consumed"] is None and h2["kwh_filled"] == 2.5 and h2["is_imputed"] is True

    h5 = rows[("S1", ts("2025-01-01T05:00"))]
    assert h5["gap_hours_before"] == 2.0
    assert h5["kwh_prev_hour"] == 2.5 and h5["kwh_delta_vs_prev"] == 1.5

    h0 = rows[("S1", ts("2025-01-01T00:00"))]
    assert h0["gap_hours_before"] == 0.0 and h0["kwh_prev_hour"] is None and h0["is_imputed"] is False
    assert h0["hour_of_day"] == 0 and h0["is_weekend"] is False  # 2025-01-01 is a Wednesday

    s2 = rows[("S2", ts("2025-01-01T00:00"))]
    assert s2["kwh_filled"] == 0.0 and s2["is_imputed"] is True


def test_rejected_rows_carry_reasons(spark: SparkSession) -> None:
    rejected = {r["reading_id"]: r["reject_reason"] for r in rejected_readings(_bronze(spark)).collect()}
    assert rejected == {
        "R6": "negative_consumption",
        "R7": "missing_site_id",
        "R8": "unparseable_timestamp",
    }
