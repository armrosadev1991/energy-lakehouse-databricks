from datetime import date

from pyspark.sql import SparkSession

from energy_lakehouse.gold.daily import site_daily_consumption
from tests.conftest import by_key, ts

COLS = (
    "site_id string, region string, customer_segment string, tariff_plan string, "
    "reading_date date, reading_hour timestamp, hour_of_day int, kwh_filled double, "
    "kwh_exported double, is_imputed boolean, gap_hours_before double"
)


def _readings(spark: SparkSession, days: list[date], per_day: dict[date, float]):  # type: ignore[no-untyped-def]
    rows = []
    for d in days:
        total = per_day[d]
        # two readings a day: 40% at 01:00 and 60% at 02:00 -> peak hour is always 2
        for hour, share in ((1, 0.4), (2, 0.6)):
            rows.append(
                (
                    "S1",
                    "north",
                    "residential",
                    "flat",
                    d,
                    ts(f"{d.isoformat()}T0{hour}:00"),
                    hour,
                    total * share,
                    0.0,
                    False,
                    0.0,
                )
            )
    return spark.createDataFrame(rows, COLS)


def test_lags_moving_averages_and_running_totals(spark: SparkSession) -> None:
    days = [date(2025, 1, d) for d in range(1, 10)]
    totals = {d: 10.0 * d.day for d in days}  # 10, 20, ..., 90
    daily = by_key(site_daily_consumption(_readings(spark, days, totals)), "site_id", "reading_date")

    d1, d2, d7, d8, d9 = (daily[("S1", date(2025, 1, n))] for n in (1, 2, 7, 8, 9))
    assert d1["kwh_total"] == 10.0 and d1["peak_hour_of_day"] == 2
    assert d1["kwh_peak_hour"] == 6.0 and d1["kwh_avg_hour"] == 5.0 and d1["load_factor"] == 0.8333
    assert d1["kwh_prev_day"] is None and d1["dod_change_pct"] is None
    assert d2["kwh_prev_day"] == 10.0 and d2["dod_change_pct"] == 100.0
    assert d1["kwh_next_day"] == 20.0 and d9["kwh_next_day"] is None
    assert d8["kwh_same_day_prev_week"] == 10.0 and d8["wow_change_pct"] == 700.0
    assert d7["wow_change_pct"] is None
    assert d1["kwh_ma_7d"] == 10.0 and d7["kwh_ma_7d"] == 40.0 and d9["kwh_ma_7d"] == 60.0
    assert d9["kwh_ma_30d_calendar"] == 50.0  # all nine days are inside the calendar window
    assert d2["kwh_mtd_cumulative"] == 30.0 and d9["kwh_mtd_cumulative"] == 450.0
    assert d9["day_in_month_seq"] == 9


def test_rows_frame_vs_calendar_range_frame(spark: SparkSession) -> None:
    """A gap of 37 days: ROWS looks back 6 *rows*, RANGE looks back 29 *days*."""
    days = [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3), date(2025, 2, 9)]
    totals = {days[0]: 10.0, days[1]: 20.0, days[2]: 30.0, days[3]: 100.0}
    daily = by_key(site_daily_consumption(_readings(spark, days, totals)), "site_id", "reading_date")
    feb9 = daily[("S1", date(2025, 2, 9))]
    assert feb9["kwh_ma_7d"] == 40.0  # (10+20+30+100)/4 — ROWS frame ignores the calendar
    assert feb9["kwh_ma_30d_calendar"] == 100.0  # nothing else within the previous 29 days
    assert feb9["kwh_mtd_cumulative"] == 100.0  # month-to-date restarted in February
    assert feb9["kwh_prev_day"] == 30.0  # lag() is row-based, so the "previous day" is Jan 3
    assert feb9["day_in_month_seq"] == 1


def test_missing_hours_and_imputations_are_counted(spark: SparkSession) -> None:
    rows = [
        (
            "S1",
            "north",
            "residential",
            "flat",
            date(2025, 1, 1),
            ts("2025-01-01T00:00"),
            0,
            1.0,
            0.0,
            False,
            0.0,
        ),
        (
            "S1",
            "north",
            "residential",
            "flat",
            date(2025, 1, 1),
            ts("2025-01-01T03:00"),
            3,
            1.0,
            0.5,
            True,
            2.0,
        ),
    ]
    daily = by_key(site_daily_consumption(spark.createDataFrame(rows, COLS)), "site_id", "reading_date")
    d = daily[("S1", date(2025, 1, 1))]
    assert d["hours_observed"] == 2 and d["hours_imputed"] == 1 and d["hours_missing"] == 2.0
    assert d["kwh_exported_total"] == 0.5 and d["peak_hour_of_day"] == 0  # tie -> earliest hour
