"""Table read/write helpers. All Delta interaction goes through SQL so it works identically
on classic clusters, serverless (Spark Connect) and local delta-spark."""

from __future__ import annotations

from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession

from energy_lakehouse.config import PipelineConfig


def ensure_schema(spark: SparkSession, cfg: PipelineConfig) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.qualified_schema}")


def table_exists(spark: SparkSession, name: str) -> bool:
    return spark.catalog.tableExists(name.replace("`", ""))


def read_table(spark: SparkSession, name: str) -> DataFrame:
    return spark.table(name)


def write_append(df: DataFrame, name: str, cfg: PipelineConfig) -> None:
    df.write.format(cfg.table_format).mode("append").saveAsTable(name)


def write_overwrite(
    df: DataFrame,
    name: str,
    cfg: PipelineConfig,
    partition_by: Sequence[str] | None = None,
) -> None:
    writer = df.write.format(cfg.table_format).mode("overwrite").option("overwriteSchema", "true")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.saveAsTable(name)


def merge_upsert(
    spark: SparkSession,
    df: DataFrame,
    name: str,
    keys: Sequence[str],
    cfg: PipelineConfig,
    partition_by: Sequence[str] | None = None,
) -> None:
    """Upsert ``df`` into ``name`` on ``keys`` (creates the table on first run).

    Uses ``MERGE INTO`` rather than the ``DeltaTable`` Python API so the same code path runs
    on Spark Connect / serverless where the JVM-backed API is not available.
    """
    if not table_exists(spark, name):
        write_overwrite(df, name, cfg, partition_by)
        return
    view = f"_src_{abs(hash(name)) % 10_000_000}"
    df.createOrReplaceTempView(view)
    on_clause = " AND ".join(f"t.`{k}` <=> s.`{k}`" for k in keys)
    spark.sql(
        f"""
        MERGE INTO {name} AS t
        USING {view} AS s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    spark.catalog.dropTempView(view)
