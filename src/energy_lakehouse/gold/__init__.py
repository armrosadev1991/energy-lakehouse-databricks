"""Gold layer: analytics-ready tables built with Spark window functions.

Every builder is a pure function ``(spark, SilverInputs) -> DataFrame`` so it can be
unit-tested on tiny hand-built frames. ``GOLD_TABLES`` is the ordered registry the job runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.gold.anomalies import build_consumption_anomalies
from energy_lakehouse.gold.common import SilverInputs
from energy_lakehouse.gold.cost import build_energy_cost_daily
from energy_lakehouse.gold.daily import build_site_daily_consumption
from energy_lakehouse.gold.distribution import build_consumption_distribution
from energy_lakehouse.gold.generation import build_generation_kpis
from energy_lakehouse.gold.monthly import build_site_monthly_kpis
from energy_lakehouse.gold.outages import build_outage_streaks
from energy_lakehouse.gold.peaks import build_peak_demand_hours
from energy_lakehouse.gold.rollup import build_consumption_rollup
from energy_lakehouse.gold.weather import build_weather_sensitivity
from energy_lakehouse.io import read_table, write_overwrite
from energy_lakehouse.silver.sites import join_site_as_of

GoldBuilder = Callable[[SparkSession, SilverInputs], DataFrame]


@dataclass(frozen=True)
class GoldTable:
    name: str
    build: GoldBuilder
    partition_by: tuple[str, ...] = ()


GOLD_TABLES: tuple[GoldTable, ...] = (
    GoldTable("site_daily_consumption", build_site_daily_consumption),
    GoldTable("site_monthly_kpis", build_site_monthly_kpis),
    GoldTable("peak_demand_hours", build_peak_demand_hours),
    GoldTable("outage_streaks", build_outage_streaks),
    GoldTable("energy_cost_daily", build_energy_cost_daily),
    GoldTable("weather_sensitivity_monthly", build_weather_sensitivity),
    GoldTable("generation_kpis_hourly", build_generation_kpis),
    GoldTable("consumption_rollup_monthly", build_consumption_rollup),
    GoldTable("consumption_distribution_monthly", build_consumption_distribution),
    GoldTable("consumption_anomalies", build_consumption_anomalies),
)


@dataclass(frozen=True)
class GoldResult:
    table: str
    rows: int


def load_silver_inputs(spark: SparkSession, cfg: PipelineConfig) -> SilverInputs:
    readings = read_table(spark, cfg.table("silver", "meter_readings"))
    sites = read_table(spark, cfg.table("silver", "sites"))
    return SilverInputs(
        readings=readings,
        readings_enriched=join_site_as_of(readings, sites),
        sites=sites,
        weather=read_table(spark, cfg.table("silver", "weather")),
        prices=read_table(spark, cfg.table("silver", "prices")),
        generation=read_table(spark, cfg.table("silver", "generation_mix")),
    )


def run_gold(spark: SparkSession, cfg: PipelineConfig) -> list[GoldResult]:
    """Full recompute of every gold table (they are small aggregates; overwrite is simplest)."""
    inputs = load_silver_inputs(spark, cfg)
    results: list[GoldResult] = []
    for gold in GOLD_TABLES:
        target = cfg.table("gold", gold.name)
        df = gold.build(spark, inputs)
        write_overwrite(df, target, cfg, gold.partition_by or None)
        results.append(GoldResult(target, spark.table(target).count()))
    return results


__all__ = [
    "GOLD_TABLES",
    "GoldResult",
    "GoldTable",
    "SilverInputs",
    "load_silver_inputs",
    "run_gold",
]
