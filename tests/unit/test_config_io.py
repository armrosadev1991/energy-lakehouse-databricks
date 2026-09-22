import pytest
from pyspark.sql import SparkSession

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.io import merge_upsert, table_exists, write_overwrite


def test_table_names() -> None:
    cfg = PipelineConfig.for_databricks("main", "energy")
    assert cfg.table("silver", "meter_readings") == "`main`.`energy`.`silver_meter_readings`"
    assert cfg.landing_path == "/Volumes/main/energy/landing"
    assert cfg.feed_path("sites") == "/Volumes/main/energy/landing/sites"
    with pytest.raises(ValueError, match="unknown layer"):
        cfg.table("platinum", "x")


def test_merge_upsert_creates_then_updates(spark: SparkSession, cfg: PipelineConfig) -> None:
    target = cfg.table("silver", "t")
    first = spark.createDataFrame([(1, "a"), (2, "b")], "id int, v string")
    merge_upsert(spark, first, target, ["id"], cfg)
    assert table_exists(spark, target)
    second = spark.createDataFrame([(2, "B"), (3, "c")], "id int, v string")
    merge_upsert(spark, second, target, ["id"], cfg)
    got = {r["id"]: r["v"] for r in spark.table(target).collect()}
    assert got == {1: "a", 2: "B", 3: "c"}


def test_overwrite_replaces_schema(spark: SparkSession, cfg: PipelineConfig) -> None:
    target = cfg.table("gold", "t")
    write_overwrite(spark.createDataFrame([(1,)], "a int"), target, cfg)
    write_overwrite(spark.createDataFrame([("x", 2)], "b string, c int"), target, cfg)
    assert spark.table(target).columns == ["b", "c"]
