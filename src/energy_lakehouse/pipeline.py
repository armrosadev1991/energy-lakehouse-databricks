"""Stage orchestration used by both the CLI and the tests."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from pyspark.sql import SparkSession

from energy_lakehouse.bronze import ingest_all
from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.datagen import GeneratorConfig, generate, write_csv
from energy_lakehouse.gold import run_gold
from energy_lakehouse.io import ensure_schema
from energy_lakehouse.quality import run_quality
from energy_lakehouse.silver import run_silver

log = logging.getLogger("energy_lakehouse")

STAGES: tuple[str, ...] = ("generate", "bronze", "silver", "gold", "quality")


@dataclass(frozen=True)
class GenerateOptions:
    n_sites: int = 12
    n_days: int = 60
    start_date: date = date(2025, 1, 1)
    seed: int = 42


def run_generate(cfg: PipelineConfig, opts: GenerateOptions) -> int:
    """Write the synthetic feeds into the landing area (a UC volume on Databricks)."""
    dataset = generate(
        GeneratorConfig(n_sites=opts.n_sites, n_days=opts.n_days, start_date=opts.start_date, seed=opts.seed)
    )
    written = write_csv(dataset, cfg.landing_path)
    log.info("generated %d files under %s", len(written), cfg.landing_path)
    return len(written)


def run_stage(
    spark: SparkSession,
    cfg: PipelineConfig,
    stage: str,
    gen: GenerateOptions | None = None,
) -> None:
    if stage not in STAGES:
        msg = f"unknown stage {stage!r}; expected one of {STAGES}"
        raise ValueError(msg)
    ensure_schema(spark, cfg)
    if stage == "generate":
        run_generate(cfg, gen or GenerateOptions())
    elif stage == "bronze":
        for r in ingest_all(spark, cfg):
            log.info(
                "bronze %-16s rows=%-7d new_files=%d skipped_files=%d",
                r.feed,
                r.rows_ingested,
                r.files_ingested,
                r.files_skipped,
            )
    elif stage == "silver":
        for s in run_silver(spark, cfg):
            log.info("silver %s rows=%d", s.table, s.rows)
    elif stage == "gold":
        for g in run_gold(spark, cfg):
            log.info("gold %s rows=%d", g.table, g.rows)
    elif stage == "quality":
        for q in run_quality(spark, cfg):
            log.info(
                "quality %-40s %s failing=%d",
                q.check.name,
                "PASS" if q.passed else "FAIL",
                q.failing_rows,
            )


def run_stages(
    spark: SparkSession,
    cfg: PipelineConfig,
    stages: Iterable[str],
    gen: GenerateOptions | None = None,
) -> None:
    for stage in stages:
        log.info("=== stage %s (run_id=%s) ===", stage, cfg.run_id)
        run_stage(spark, cfg, stage, gen)
