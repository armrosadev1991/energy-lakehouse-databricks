from datetime import date

import pytest
from pyspark.sql import SparkSession

from energy_lakehouse.gold.cost import energy_cost_daily
from energy_lakehouse.gold.generation import generation_kpis_hourly
from energy_lakehouse.gold.weather import weather_sensitivity_monthly
from tests.conftest import by_key, ts

READ_COLS = (
    "site_id string, region string, customer_segment string, tariff_plan string, reading_date date, "
    "reading_hour timestamp, kwh_filled double, kwh_exported double"
)
PRICE_COLS = "region string, market string, price_hour timestamp, price_eur_mwh double, is_peak_price_hour boolean"


def test_energy_cost_daily(spark: SparkSession) -> None:
    readings = spark.createDataFrame(
        [
            (
                "S1",
                "north",
                "residential",
                "flat",
                date(2025, 1, 1),
                ts("2025-01-01T00:00"),
                1000.0,
                0.0,
            ),
            (
                "S1",
                "north",
                "residential",
                "flat",
                date(2025, 1, 1),
                ts("2025-01-01T01:00"),
                3000.0,
                0.0,
            ),
            (
                "S1",
                "north",
                "residential",
                "flat",
                date(2025, 1, 2),
                ts("2025-01-02T00:00"),
                1000.0,
                0.0,
            ),
            (
                "S1",
                "north",
                "residential",
                "flat",
                date(2025, 1, 2),
                ts("2025-01-02T01:00"),
                500.0,
                0.0,
            ),  # no price
        ],
        READ_COLS,
    )
    prices = spark.createDataFrame(
        [
            ("north", "day_ahead", ts("2025-01-01T00:00"), 50.0, False),
            ("north", "day_ahead", ts("2025-01-01T01:00"), 100.0, True),
            ("north", "day_ahead", ts("2025-01-02T00:00"), 70.0, False),
            ("north", "intraday", ts("2025-01-02T01:00"), 999.0, False),  # other market is ignored
        ],
        PRICE_COLS,
    )
    daily = by_key(energy_cost_daily(readings, prices), "site_id", "reading_date")
    d1, d2 = daily[("S1", date(2025, 1, 1))], daily[("S1", date(2025, 1, 2))]
    assert d1["cost_eur"] == 350.0 and d1["kwh_weighted_price_eur_mwh"] == 87.5 and d1["avg_price_eur_mwh"] == 75.0
    assert d1["peak_price_kwh_share_pct"] == 75.0 and d1["hours_without_price"] == 0
    assert d2["cost_eur"] == 70.0 and d2["hours_without_price"] == 1
    assert d2["cost_prev_day"] == 350.0 and d2["cost_dod_change_pct"] == -80.0
    assert d1["cost_mtd_eur"] == 350.0 and d2["cost_mtd_eur"] == 420.0
    assert d1["cost_rank_in_month"] == 1 and d2["cost_rank_in_month"] == 2
    assert d1["cost_quartile_in_region_day"] == 1


def test_weather_sensitivity_recovers_a_linear_relationship(spark: SparkSession) -> None:
    temps = list(range(10))  # kwh = 2 * temp + 1 exactly
    readings = spark.createDataFrame(
        [
            (
                "S1",
                "north",
                "residential",
                "flat",
                date(2025, 1, 1),
                ts(f"2025-01-01T{t:02d}:00"),
                2.0 * t + 1,
                0.0,
            )
            for t in temps
        ],
        READ_COLS,
    )
    weather = spark.createDataFrame(
        [("north", ts(f"2025-01-01T{t:02d}:00"), float(t), 0.0, max(18.0 - t, 0.0), 0.0) for t in temps],
        "region string, observed_hour timestamp, temperature_c double, solar_irradiance_wm2 double, "
        "heating_degree_hours double, cooling_degree_hours double",
    )
    row = weather_sensitivity_monthly(readings, weather).collect()[0]
    assert row["temp_correlation"] == 1.0
    assert row["kwh_per_degree_c"] == 2.0 and row["kwh_at_zero_c"] == 1.0
    assert row["export_irradiance_correlation"] is None  # both constant -> null, not an ANSI error
    assert row["heating_degree_hours"] == 135.0 and row["kwh_during_heating_hours"] == 100.0
    assert row["kwh_per_heating_degree_hour"] == pytest.approx(100 / 135, abs=1e-4)
    assert row["avg_temperature_c"] == 4.5 and row["hours_matched"] == 10
    assert row["sensitivity_rank_in_segment"] == 1 and row["temp_correlation_prev_month"] is None


def test_generation_kpis(spark: SparkSession) -> None:
    rows = [
        ("north", ts("2025-01-01T00:00"), "solar", 100.0, True),
        ("north", ts("2025-01-01T00:00"), "wind", 100.0, True),
        ("north", ts("2025-01-01T00:00"), "gas", 200.0, False),
        ("north", ts("2025-01-01T01:00"), "solar", 0.0, True),
        ("north", ts("2025-01-01T01:00"), "wind", 300.0, True),
        ("north", ts("2025-01-01T01:00"), "gas", 100.0, False),
    ]
    gen = spark.createDataFrame(
        rows,
        "region string, generation_hour timestamp, source string, generation_mw double, is_renewable boolean",
    )
    kpis = by_key(generation_kpis_hourly(gen), "region", "generation_hour")
    h0, h1 = kpis[("north", ts("2025-01-01T00:00"))], kpis[("north", ts("2025-01-01T01:00"))]
    assert h0["renewable_share_pct"] == 50.0 and h1["renewable_share_pct"] == 75.0
    assert h0["dominant_source"] == "gas" and h1["dominant_source"] == "wind" and h1["smallest_source"] == "solar"
    assert h1["renewable_share_ma_24h"] == 62.5 and h1["renewable_share_same_hour_prev_day"] is None
    assert (h0["solar_mw"], h0["wind_mw"], h0["gas_mw"], h0["hydro_mw"], h0["nuclear_mw"]) == (
        100.0,
        100.0,
        200.0,
        None,
        None,
    )
    assert (h0["renewable_share_decile_in_month"], h1["renewable_share_decile_in_month"]) == (1, 2)
    assert (h0["renewable_share_cume_dist"], h1["renewable_share_cume_dist"]) == (0.5, 1.0)
