"""Land raw CSV feeds into bronze tables.

Bronze rules:
* schema-on-read: every column is kept as a string exactly as it arrived
* append-only, with ``_source_file``, ``_ingested_at``, ``_batch_id`` lineage columns
* idempotent: files already present in the table are skipped, so re-running a batch
  never duplicates bronze rows (silver still de-duplicates *within* a file).
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.io import table_exists, write_append
from energy_lakehouse.schemas import FEEDS, raw_schema

LINEAGE_COLUMNS = ("_source_file", "_ingested_at", "_batch_id")


@dataclass(frozen=True)
class IngestResult:
    feed: str
    table: str
    rows_ingested: int
    files_ingested: int
    files_skipped: int


def read_landed_csv(spark: SparkSession, path: str, feed: str) -> DataFrame:
    """Read every CSV under ``path`` for ``feed`` with lineage columns attached."""
    return (
        spark.read.option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(raw_schema(feed))
        .csv(path)
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


def with_lineage(df: DataFrame, cfg: PipelineConfig) -> DataFrame:
    return df.withColumn("_ingested_at", F.current_timestamp()).withColumn("_batch_id", F.lit(cfg.run_id))


def new_files_only(spark: SparkSession, df: DataFrame, target: str) -> DataFrame:
    """Drop rows whose ``_source_file`` was already ingested into ``target``."""
    if not table_exists(spark, target):
        return df
    seen = spark.table(target).select("_source_file").distinct()
    return df.join(seen, on="_source_file", how="left_anti")


def ingest_feed(spark: SparkSession, cfg: PipelineConfig, feed: str) -> IngestResult:
    if feed not in FEEDS:
        msg = f"unknown feed {feed!r}; expected one of {sorted(FEEDS)}"
        raise ValueError(msg)
    target = cfg.table("bronze", feed)
    try:
        landed = read_landed_csv(spark, cfg.feed_path(feed), feed)
        all_files = landed.select("_source_file").distinct().count()
    except AnalysisException as exc:  # nothing landed yet for this feed
        if "PATH_NOT_FOUND" not in str(exc):
            raise
        return IngestResult(feed, target, 0, 0, 0)

    fresh = with_lineage(new_files_only(spark, landed, target), cfg)
    new_files = fresh.select("_source_file").distinct().count()
    rows = fresh.count()
    if rows:
        write_append(fresh, target, cfg)
    return IngestResult(feed, target, rows, new_files, all_files - new_files)


def ingest_all(spark: SparkSession, cfg: PipelineConfig) -> list[IngestResult]:
    return [ingest_feed(spark, cfg, feed) for feed in FEEDS]
