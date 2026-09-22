"""Column contracts for the raw feeds landing in the bronze layer.

Bronze keeps every column as a string (schema-on-read); the silver layer casts and validates.
The column order here is also the CSV column order produced by ``energy_lakehouse.datagen``.
"""

from __future__ import annotations

from pyspark.sql.types import StringType, StructField, StructType

FEEDS: dict[str, list[str]] = {
    "meter_readings": [
        "reading_id",
        "site_id",
        "reading_ts",
        "kwh_consumed",
        "kwh_exported",
        "voltage_v",
        "quality_flag",
    ],
    "sites": [
        "site_id",
        "site_name",
        "region",
        "customer_segment",
        "tariff_plan",
        "contracted_kw",
        "effective_from",
    ],
    "weather": [
        "region",
        "observed_ts",
        "temperature_c",
        "wind_speed_ms",
        "solar_irradiance_wm2",
        "humidity_pct",
    ],
    "prices": [
        "region",
        "price_ts",
        "price_eur_mwh",
        "market",
    ],
    "generation_mix": [
        "region",
        "generation_ts",
        "source",
        "generation_mw",
    ],
}

# Feeds that arrive as one file per day (bronze skips files it has already ingested).
DAILY_FEEDS = frozenset({"meter_readings", "weather", "prices", "generation_mix"})


def raw_schema(feed: str) -> StructType:
    """All-string StructType for a feed, used when reading landed CSV files."""
    return StructType([StructField(c, StringType(), nullable=True) for c in FEEDS[feed]])
