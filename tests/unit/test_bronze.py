from datetime import date

from pyspark.sql import SparkSession

from energy_lakehouse.bronze import ingest_all, ingest_feed
from energy_lakehouse.bronze.ingest import LINEAGE_COLUMNS
from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.datagen import GeneratorConfig, generate, write_csv
from energy_lakehouse.datagen.generator import row_count
from energy_lakehouse.schemas import FEEDS

SMALL = GeneratorConfig(n_sites=3, n_days=2, start_date=date(2025, 1, 1), seed=1)


def test_ingest_is_idempotent_per_file(spark: SparkSession, cfg: PipelineConfig) -> None:
    ds = generate(SMALL)
    write_csv(ds, cfg.landing_path)

    first = ingest_feed(spark, cfg, "weather")
    assert first.rows_ingested == row_count(ds, "weather")
    assert first.files_ingested == SMALL.n_days
    assert first.files_skipped == 0
    table = spark.table(first.table)
    assert set(LINEAGE_COLUMNS) <= set(table.columns)
    assert set(FEEDS["weather"]) <= set(table.columns)
    assert table.select("_batch_id").distinct().collect()[0][0] == cfg.run_id

    again = ingest_feed(spark, cfg, "weather")
    assert again.rows_ingested == 0
    assert again.files_skipped == SMALL.n_days
    assert spark.table(first.table).count() == first.rows_ingested

    # a brand-new file is picked up on the next run, older ones are still skipped
    extra = {"weather": {"2025-01-03.csv": ds["weather"]["2025-01-01.csv"][:5]}}
    write_csv(extra, cfg.landing_path)
    third = ingest_feed(spark, cfg, "weather")
    assert (third.rows_ingested, third.files_ingested, third.files_skipped) == (5, 1, SMALL.n_days)


def test_missing_feed_directory_is_not_an_error(spark: SparkSession, cfg: PipelineConfig) -> None:
    result = ingest_feed(spark, cfg, "prices")
    assert result.rows_ingested == 0


def test_ingest_all_covers_every_feed(spark: SparkSession, cfg: PipelineConfig) -> None:
    write_csv(generate(SMALL), cfg.landing_path)
    results = ingest_all(spark, cfg)
    assert [r.feed for r in results] == list(FEEDS)
    assert all(r.rows_ingested > 0 for r in results)
