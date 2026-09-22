from datetime import date

from pyspark.sql import SparkSession

from energy_lakehouse.schemas import FEEDS
from energy_lakehouse.silver.sites import build_silver_sites, join_site_as_of
from tests.conftest import bronze_like, rows_to_dicts, ts

COLS = FEEDS["sites"]
BASE = {
    "site_id": "S1",
    "site_name": "Site 1",
    "region": "North",
    "customer_segment": "residential",
    "contracted_kw": "6.9",
}


def _sites(spark: SparkSession):  # type: ignore[no-untyped-def]
    return bronze_like(
        spark,
        COLS,
        [
            {**BASE, "tariff_plan": "flat", "effective_from": "2024-01-01"},
            # same attributes re-snapshotted later: must NOT open a new version
            {**BASE, "tariff_plan": "flat", "effective_from": "2024-06-01"},
            # two rows for the same effective_from: the later ingestion wins
            {
                **BASE,
                "tariff_plan": "time_of_use",
                "effective_from": "2025-01-15",
                "_ingested_at": ts("2025-01-15T01:00"),
            },
            {
                **BASE,
                "tariff_plan": "dynamic",
                "effective_from": "2025-01-15",
                "_ingested_at": ts("2025-01-16T01:00"),
            },
            {
                "site_id": "S2",
                "site_name": "Site 2",
                "region": "south",
                "customer_segment": "commercial",
                "contracted_kw": "41.4",
                "tariff_plan": "flat",
                "effective_from": "2024-03-01",
            },
        ],
    )


def test_scd2_versions(spark: SparkSession) -> None:
    dim = build_silver_sites(_sites(spark))
    s1 = [r for r in rows_to_dicts(dim, "site_id", "valid_from") if r["site_id"] == "S1"]
    assert [(r["version"], r["tariff_plan"], r["valid_from"], r["valid_to"], r["is_current"]) for r in s1] == [
        (1, "flat", date(2024, 1, 1), date(2025, 1, 15), False),
        (2, "dynamic", date(2025, 1, 15), None, True),
    ]
    assert s1[0]["region"] == "north"  # normalised
    s2 = [r for r in rows_to_dicts(dim) if r["site_id"] == "S2"]
    assert len(s2) == 1 and s2[0]["is_current"] is True and s2[0]["version"] == 1


def test_point_in_time_join(spark: SparkSession) -> None:
    dim = build_silver_sites(_sites(spark))
    facts = spark.createDataFrame(
        [
            ("S1", ts("2025-01-10T12:00")),
            ("S1", ts("2025-01-15T00:00")),
            ("S1", ts("2023-12-31T23:00")),
            ("S9", ts("2025-01-10T00:00")),
        ],
        "site_id string, reading_hour timestamp",
    )
    got = {
        (r["site_id"], r["reading_hour"]): (r["tariff_plan"], r["site_version"])
        for r in join_site_as_of(facts, dim).collect()
    }
    assert got[("S1", ts("2025-01-10T12:00"))] == ("flat", 1)
    assert got[("S1", ts("2025-01-15T00:00"))] == ("dynamic", 2)
    assert got[("S1", ts("2023-12-31T23:00"))] == (None, None)  # before the first version
    assert got[("S9", ts("2025-01-10T00:00"))] == (
        None,
        None,
    )  # unknown site keeps the fact (left join)
