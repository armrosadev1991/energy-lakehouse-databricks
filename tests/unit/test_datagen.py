from datetime import date

from energy_lakehouse.datagen import GeneratorConfig, generate, write_csv
from energy_lakehouse.datagen.generator import row_count
from energy_lakehouse.schemas import FEEDS

SMALL = GeneratorConfig(n_sites=4, n_days=5, start_date=date(2025, 3, 1), seed=7)


def test_generation_is_deterministic() -> None:
    assert generate(SMALL) == generate(SMALL)
    assert generate(SMALL) != generate(GeneratorConfig(n_sites=4, n_days=5, seed=8))


def test_every_feed_has_exact_columns() -> None:
    ds = generate(SMALL)
    assert set(ds) == set(FEEDS)
    for feed, files in ds.items():
        for rows in files.values():
            for row in rows:
                assert list(row) == FEEDS[feed], feed


def test_shapes() -> None:
    ds = generate(SMALL)
    assert len(ds["weather"]) == SMALL.n_days
    assert row_count(ds, "weather") == SMALL.n_days * 24 * len(SMALL.regions)
    assert row_count(ds, "prices") == SMALL.n_days * 24 * len(SMALL.regions)
    assert row_count(ds, "generation_mix") == SMALL.n_days * 24 * len(SMALL.regions) * 5
    # one clean reading per site-hour, minus the "missing" windows, plus injected extras
    assert row_count(ds, "meter_readings") > SMALL.n_sites * SMALL.n_days * 24 * 0.9
    site_ids = {r["site_id"] for rows in ds["sites"].values() for r in rows}
    assert site_ids == {f"S{i:03d}" for i in range(1, 5)}


def test_anomalies_are_injected() -> None:
    ds = generate(GeneratorConfig(n_sites=6, n_days=8, seed=3))
    readings = [r for rows in ds["meter_readings"].values() for r in rows]
    flags = {r["quality_flag"] for r in readings}
    assert {"OK", "OUTAGE", "MISSING", "ERROR", "RESTATED"} <= flags
    ids = [r["reading_id"] for r in readings]
    assert len(ids) > len(set(ids)), "exact duplicates expected"
    assert any(r["reading_ts"][-5:-3] != "00" for r in readings), "off-the-hour timestamps expected"
    assert any(r["kwh_consumed"] == "" for r in readings)
    assert any(r["kwh_consumed"].startswith("-") for r in readings)
    site_rows = [r for rows in ds["sites"].values() for r in rows]
    assert len(site_rows) > 6, "SCD2 change snapshots expected"


def test_clean_mode_has_no_anomalies() -> None:
    ds = generate(GeneratorConfig(n_sites=3, n_days=3, inject_anomalies=False))
    readings = [r for rows in ds["meter_readings"].values() for r in rows]
    assert {r["quality_flag"] for r in readings} == {"OK"}
    assert len(readings) == 3 * 3 * 24
    assert len({r["reading_id"] for r in readings}) == len(readings)


def test_write_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    written = write_csv(generate(SMALL), tmp_path)
    assert (tmp_path / "sites" / "sites_snapshot.csv").exists()
    assert len(written) == sum(len(f) for f in generate(SMALL).values())
    header = (tmp_path / "meter_readings" / "2025-03-01.csv").read_text().splitlines()[0]
    assert header == ",".join(FEEDS["meter_readings"])
