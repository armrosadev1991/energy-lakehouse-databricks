from pyspark.sql import SparkSession

from energy_lakehouse.schemas import FEEDS
from energy_lakehouse.silver.generation import build_silver_generation
from energy_lakehouse.silver.prices import build_silver_prices
from energy_lakehouse.silver.weather import build_silver_weather
from tests.conftest import bronze_like, by_key, ts


def test_weather_forward_fill_and_degree_hours(spark: SparkSession) -> None:
    base = {
        "region": "north",
        "wind_speed_ms": "3",
        "solar_irradiance_wm2": "0",
        "humidity_pct": "50",
    }
    bronze = bronze_like(
        spark,
        FEEDS["weather"],
        [
            {**base, "observed_ts": "2025-01-01T00:00:00", "temperature_c": "10"},
            {**base, "observed_ts": "2025-01-01T01:00:00", "temperature_c": None},
            {**base, "observed_ts": "2025-01-01T02:00:00", "temperature_c": "30"},
            {
                **base,
                "observed_ts": "2025-01-01T02:00:00",
                "temperature_c": "31",
                "_ingested_at": ts("2025-03-01T00:00"),
            },
            {**base, "observed_ts": "2025-01-02T02:00:00", "temperature_c": "20"},
        ],
    )
    rows = by_key(build_silver_weather(bronze), "region", "observed_hour")
    assert rows[("north", ts("2025-01-01T01:00"))]["temperature_c"] == 10.0  # forward-filled
    assert rows[("north", ts("2025-01-01T02:00"))]["temperature_c"] == 31.0  # latest ingestion wins
    assert rows[("north", ts("2025-01-01T00:00"))]["heating_degree_hours"] == 8.0
    assert rows[("north", ts("2025-01-01T02:00"))]["cooling_degree_hours"] == 7.0
    # lag(24) rows: only 4 rows per region so the "previous day" lag is null
    assert rows[("north", ts("2025-01-02T02:00"))]["temperature_prev_day_c"] is None


def test_prices_lag_and_intraday_rank(spark: SparkSession) -> None:
    base = {"region": "south", "market": "day_ahead"}
    bronze = bronze_like(
        spark,
        FEEDS["prices"],
        [
            {**base, "price_ts": "2025-01-01T00:00:00", "price_eur_mwh": "50"},
            {**base, "price_ts": "2025-01-01T01:00:00", "price_eur_mwh": "100"},
            {**base, "price_ts": "2025-01-01T02:00:00", "price_eur_mwh": "100"},
            {**base, "price_ts": "2025-01-01T03:00:00", "price_eur_mwh": "80"},
            {**base, "price_ts": "2025-01-01T04:00:00", "price_eur_mwh": "70"},
            {**base, "price_ts": "2025-01-01T05:00:00", "price_eur_mwh": "60"},
        ],
    )
    rows = by_key(build_silver_prices(bronze), "region", "price_hour")
    h1 = rows[("south", ts("2025-01-01T01:00"))]
    assert h1["price_prev_hour"] == 50.0 and h1["price_change_pct"] == 100.0
    ranks = {h: rows[("south", ts(f"2025-01-01T0{h}:00"))]["price_rank_in_day"] for h in range(6)}
    assert ranks == {0: 6, 1: 1, 2: 1, 3: 3, 4: 4, 5: 5}  # rank() leaves a hole after ties
    peak = {h for h in range(6) if rows[("south", ts(f"2025-01-01T0{h}:00"))]["is_peak_price_hour"]}
    assert peak == {1, 2, 3, 4}


def test_generation_share_of_regional_total(spark: SparkSession) -> None:
    bronze = bronze_like(
        spark,
        FEEDS["generation_mix"],
        [
            {
                "region": "north",
                "generation_ts": "2025-01-01T00:00:00",
                "source": "solar",
                "generation_mw": "100",
            },
            {
                "region": "north",
                "generation_ts": "2025-01-01T00:00:00",
                "source": "gas",
                "generation_mw": "300",
            },
            {
                "region": "south",
                "generation_ts": "2025-01-01T00:00:00",
                "source": "wind",
                "generation_mw": "50",
            },
        ],
    )
    rows = by_key(build_silver_generation(bronze), "region", "source")
    assert rows[("north", "solar")]["share_pct"] == 25.0 and rows[("north", "solar")]["is_renewable"] is True
    assert rows[("north", "gas")]["share_pct"] == 75.0 and rows[("north", "gas")]["is_renewable"] is False
    assert rows[("south", "wind")]["share_pct"] == 100.0
