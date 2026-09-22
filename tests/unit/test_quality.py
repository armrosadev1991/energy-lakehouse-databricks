from datetime import date

import pytest
from pyspark.sql import SparkSession

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.io import write_overwrite
from energy_lakehouse.quality import CHECKS, Check, DataQualityError, run_quality
from energy_lakehouse.quality.checks import (
    contiguous_scd2_versions,
    duplicates_on,
    evaluate,
    one_current_version_per_site,
    rows_where,
)

SITES_DDL = "site_id string, valid_from date, valid_to date, is_current boolean"


def test_scd2_predicates(spark: SparkSession) -> None:
    good = spark.createDataFrame(
        [
            ("S1", date(2024, 1, 1), date(2025, 1, 1), False),
            ("S1", date(2025, 1, 1), None, True),
            ("S2", date(2024, 1, 1), None, True),
        ],
        SITES_DDL,
    )
    assert one_current_version_per_site(good) == 0 and contiguous_scd2_versions(good) == 0

    bad = spark.createDataFrame(
        [
            ("S1", date(2024, 1, 1), date(2024, 12, 1), False),
            ("S1", date(2025, 1, 1), None, True),
            ("S2", date(2024, 1, 1), None, False),
        ],
        SITES_DDL,
    )
    assert one_current_version_per_site(bad) == 1  # S2 has no current version
    assert contiguous_scd2_versions(bad) == 1  # hole between 2024-12-01 and 2025-01-01
    assert duplicates_on("site_id")(bad) == 1 and rows_where("is_current")(bad) == 1


def test_run_quality_records_results_and_raises(spark: SparkSession, cfg: PipelineConfig) -> None:
    write_overwrite(
        spark.createDataFrame([("S1", date(2024, 1, 1), None, True), ("S1", date(2025, 1, 1), None, True)], SITES_DDL),
        cfg.table("silver", "sites"),
        cfg,
    )
    checks = [
        Check("one_current", "silver", "sites", "one current per site", one_current_version_per_site),
        Check("never_fails", "silver", "sites", "always ok", lambda df: 0),
        Check("soft", "silver", "sites", "a warning only", lambda df: 5, "warn"),
        Check("absent_table", "gold", "nope", "table missing", lambda df: 0, "warn"),
    ]
    with pytest.raises(DataQualityError, match=r"1 data-quality check\(s\) failed: one_current\(1\)"):
        run_quality(spark, cfg, checks)

    results = {r["check_name"]: r for r in spark.table(cfg.table("ops", "data_quality_results")).collect()}
    assert results["one_current"]["passed"] is False and results["one_current"]["failing_rows"] == 1
    assert results["never_fails"]["passed"] is True
    assert results["soft"]["passed"] is False and results["soft"]["severity"] == "warn"
    assert results["absent_table"]["table_exists"] is False and results["absent_table"]["failing_rows"] == -1
    assert results["one_current"]["run_id"] == cfg.run_id

    soft_only = run_quality(spark, cfg, checks[1:], raise_on_error=True)  # warns never raise
    assert [r.passed for r in soft_only] == [True, False, False]


def test_builtin_checks_reference_known_tables() -> None:
    names = [c.name for c in CHECKS]
    assert len(names) == len(set(names))
    assert {c.layer for c in CHECKS} <= {"bronze", "silver", "gold"}


def test_evaluate_marks_missing_tables(spark: SparkSession, cfg: PipelineConfig) -> None:
    results = evaluate(spark, cfg, CHECKS[:3])
    assert all(not r.table_exists and not r.passed for r in results)
