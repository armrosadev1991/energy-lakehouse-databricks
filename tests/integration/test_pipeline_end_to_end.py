"""Run the whole medallion pipeline on local Spark + Delta with the synthetic generator."""

from __future__ import annotations

from datetime import date

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.datagen import GeneratorConfig, generate
from energy_lakehouse.datagen.generator import row_count
from energy_lakehouse.gold import GOLD_TABLES
from energy_lakehouse.pipeline import STAGES, GenerateOptions, run_stages
from energy_lakehouse.quality import CHECKS
from energy_lakehouse.schemas import FEEDS

pytestmark = pytest.mark.integration

GEN = GenerateOptions(n_sites=6, n_days=8, start_date=date(2025, 1, 1), seed=11)


def test_medallion_pipeline_end_to_end(spark: SparkSession, cfg: PipelineConfig) -> None:
    run_stages(spark, cfg, STAGES, GEN)
    dataset = generate(GeneratorConfig(GEN.n_sites, GEN.n_days, GEN.start_date, GEN.seed))

    # bronze: exactly what landed, nothing more
    for feed in FEEDS:
        assert spark.table(cfg.table("bronze", feed)).count() == row_count(dataset, feed), feed

    # silver: the key is unique, restatements won, gaps/outages survived as features
    readings = spark.table(cfg.table("silver", "meter_readings"))
    assert readings.groupBy("site_id", "reading_hour").count().filter("count > 1").count() == 0
    assert readings.filter("quality_flag = 'RESTATED'").count() > 0
    assert readings.filter("is_imputed").count() > 0
    assert readings.filter("gap_hours_before > 0").count() > 0
    assert readings.filter("kwh_filled < 0").count() == 0
    sites = spark.table(cfg.table("silver", "sites"))
    assert sites.filter("is_current").count() == GEN.n_sites
    assert sites.count() > GEN.n_sites  # tariff changes created extra SCD2 versions
    assert sites.filter("version = 2").count() == 2  # sites 1 and 4 (every 3rd) changed tariff

    # gold: every table materialised with rows, the interesting ones found what was injected
    for gold in GOLD_TABLES:
        assert spark.table(cfg.table("gold", gold.name)).count() > 0, gold.name
    streak_kinds = {
        r[0] for r in spark.table(cfg.table("gold", "outage_streaks")).select("streak_kind").distinct().collect()
    }
    assert streak_kinds == {"ZERO_CONSUMPTION", "MISSING_DATA"}
    anomalies = spark.table(cfg.table("gold", "consumption_anomalies"))
    assert anomalies.filter("anomaly_kind = 'SPIKE' AND site_id = 'S002'").count() > 0
    daily = spark.table(cfg.table("gold", "site_daily_consumption"))
    assert daily.count() == GEN.n_sites * GEN.n_days
    assert daily.filter("kwh_ma_7d IS NULL OR kwh_mtd_cumulative IS NULL").count() == 0
    monthly = spark.table(cfg.table("gold", "site_monthly_kpis"))
    assert monthly.count() == GEN.n_sites
    assert monthly.agg(F.max("cume_dist_overall")).collect()[0][0] == 1.0

    # quality: every built-in check ran and passed (warnings included)
    dq = spark.table(cfg.table("ops", "data_quality_results"))
    assert dq.count() == len(CHECKS)
    failed = [r["check_name"] for r in dq.filter("NOT passed").collect()]
    assert failed == []


def test_rerun_is_idempotent(spark: SparkSession, cfg: PipelineConfig) -> None:
    run_stages(spark, cfg, STAGES, GEN)
    before = {
        t: spark.table(cfg.table(*t)).count()
        for t in (
            ("bronze", "meter_readings"),
            ("silver", "meter_readings"),
            ("silver", "sites"),
            ("gold", "site_daily_consumption"),
        )
    }
    second = PipelineConfig(cfg.catalog, cfg.schema, cfg.landing_path, run_id="run_two")
    run_stages(spark, second, ("bronze", "silver", "gold", "quality"))
    after = {t: spark.table(cfg.table(*t)).count() for t in before}
    assert after == before
    dq = spark.table(cfg.table("ops", "data_quality_results"))
    assert dq.select("run_id").distinct().count() == 2 and dq.count() == 2 * len(CHECKS)
