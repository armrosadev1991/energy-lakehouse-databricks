"""Silver layer: typed, validated, de-duplicated, conformed tables."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import SparkSession

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.io import merge_upsert, read_table
from energy_lakehouse.silver.generation import GENERATION_KEYS, build_silver_generation
from energy_lakehouse.silver.prices import PRICE_KEYS, build_silver_prices
from energy_lakehouse.silver.readings import READING_KEYS, build_silver_readings
from energy_lakehouse.silver.sites import SITE_KEYS, build_silver_sites
from energy_lakehouse.silver.weather import WEATHER_KEYS, build_silver_weather


@dataclass(frozen=True)
class SilverResult:
    table: str
    rows: int


def run_silver(spark: SparkSession, cfg: PipelineConfig) -> list[SilverResult]:
    """Rebuild every silver table from bronze and upsert it (MERGE) into place.

    Silver is recomputed from the full bronze history on each run: the time-series
    features (forward-fill, lags, gaps) depend on neighbouring rows, so a partial
    recompute would be wrong at the batch boundaries. MERGE keeps the operation idempotent.
    """
    plan = [
        ("meter_readings", build_silver_readings, READING_KEYS, ["reading_date"]),
        ("sites", build_silver_sites, SITE_KEYS, None),
        ("weather", build_silver_weather, WEATHER_KEYS, None),
        ("prices", build_silver_prices, PRICE_KEYS, None),
        ("generation_mix", build_silver_generation, GENERATION_KEYS, None),
    ]
    results: list[SilverResult] = []
    for name, build, keys, partition_by in plan:
        source = read_table(spark, cfg.table("bronze", name))
        target = cfg.table("silver", name)
        df = build(source)
        merge_upsert(spark, df, target, keys, cfg, partition_by)
        results.append(SilverResult(target, spark.table(target).count()))
    return results


__all__ = [
    "SilverResult",
    "build_silver_generation",
    "build_silver_prices",
    "build_silver_readings",
    "build_silver_sites",
    "build_silver_weather",
    "run_silver",
]
