"""Shared fixtures: a local Delta-enabled SparkSession and a throw-away schema per test."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.io import ensure_schema

LINEAGE_DDL = "_source_file string, _ingested_at timestamp, _batch_id string"


@pytest.fixture(scope="session")
def spark(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SparkSession]:
    root = tmp_path_factory.mktemp("spark")
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("energy-lakehouse-tests")
        .config("spark.sql.warehouse.dir", str(root / "warehouse"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={root / 'derby'}")
    )
    session = configure_spark_with_delta_pip(builder).getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def cfg(spark: SparkSession, tmp_path: Path) -> Iterator[PipelineConfig]:
    """A PipelineConfig pointing at a fresh schema and landing directory; dropped afterwards."""
    config = PipelineConfig(
        catalog="spark_catalog",
        schema=f"t_{uuid.uuid4().hex[:8]}",
        landing_path=str(tmp_path / "landing"),
        run_id="run_test",
    )
    ensure_schema(spark, config)
    yield config
    spark.sql(f"DROP SCHEMA IF EXISTS {config.qualified_schema} CASCADE")


def ts(value: str) -> datetime:
    """'2025-01-01T08:00' -> datetime (UTC-naive; the session runs in UTC)."""
    return datetime.fromisoformat(value)


def rows_to_dicts(df: DataFrame, *order_by: str) -> list[dict[str, Any]]:
    out = df.orderBy(*order_by) if order_by else df
    return [r.asDict() for r in out.collect()]


def by_key(df: DataFrame, *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(r[k] for k in keys): r for r in rows_to_dicts(df)}


def bronze_like(spark: SparkSession, feed_columns: list[str], rows: list[dict[str, Any]]) -> DataFrame:
    """Build a bronze-shaped DataFrame (all strings + lineage) from partial dicts."""
    ddl = ", ".join(f"{c} string" for c in feed_columns) + ", " + LINEAGE_DDL
    full = []
    for r in rows:
        lineage = (
            r.get("_source_file", "file_1.csv"),
            r.get("_ingested_at", ts("2025-02-01T00:00")),
            r.get("_batch_id", "b1"),
        )
        full.append((*[r.get(c) for c in feed_columns], *lineage))
    return spark.createDataFrame(full, ddl)
