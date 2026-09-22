"""Synthetic but realistic energy data.

Everything is driven by ``random.Random(seed)`` so a given ``GeneratorConfig`` always yields
byte-identical files. The generator intentionally injects the messiness a real smart-meter
feed has, so the silver/gold layers have something to clean and analyse:

* exact duplicate rows and *restated* readings that arrive in the next day's file
* timestamps a few minutes off the hour (``date_trunc`` in silver aligns them)
* null and negative ``kwh_consumed`` values with matching quality flags
* multi-hour outages (zero consumption) and windows where rows are missing entirely
* one consumption spike per dataset for anomaly detection
* site attribute changes over time (tariff plan) to exercise SCD Type 2
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from random import Random

from energy_lakehouse.schemas import FEEDS

REGIONS: tuple[str, ...] = ("north", "centre", "south")
SEGMENTS: tuple[str, ...] = ("residential", "commercial", "industrial")
TARIFFS: tuple[str, ...] = ("flat", "time_of_use", "dynamic")
SOURCES: tuple[str, ...] = ("solar", "wind", "hydro", "gas", "nuclear")

Row = dict[str, str]
Files = dict[str, list[Row]]  # filename -> rows
SyntheticDataset = dict[str, Files]  # feed -> files


@dataclass(frozen=True)
class GeneratorConfig:
    n_sites: int = 12
    n_days: int = 60
    start_date: date = date(2025, 1, 1)
    seed: int = 42
    inject_anomalies: bool = True
    regions: tuple[str, ...] = REGIONS


@dataclass
class _Site:
    site_id: str
    name: str
    region: str
    segment: str
    tariff: str
    contracted_kw: float
    has_solar: bool
    base_kw: float
    outage: tuple[datetime, int] | None = None
    missing: tuple[datetime, int] | None = None
    spike: tuple[datetime, int] | None = None
    restated: set[datetime] = field(default_factory=set)


def _fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _day_of_year_temp(region: str, ts: datetime, rng: Random) -> float:
    """Seasonal + diurnal temperature with region offsets (Portugal-like climate)."""
    offset = {"north": -2.0, "centre": 0.0, "south": 3.0}.get(region, 0.0)
    season = 15.0 + 8.0 * math.sin(2 * math.pi * (ts.timetuple().tm_yday - 110) / 365.0)
    diurnal = 4.5 * math.sin(2 * math.pi * (ts.hour - 9) / 24.0)
    return season + diurnal + offset + rng.gauss(0, 0.8)


def _daylight(ts: datetime) -> float:
    """0..1 solar elevation proxy, zero at night."""
    return max(0.0, math.sin(math.pi * (ts.hour - 6) / 12.0)) if 6 <= ts.hour <= 18 else 0.0


def _profile(site: _Site, ts: datetime) -> float:
    """Hour-of-day / day-of-week load multiplier per customer segment."""
    h, weekend = ts.hour, ts.weekday() >= 5
    if site.segment == "residential":
        morning = math.exp(-((h - 8) ** 2) / 4.0)
        evening = math.exp(-((h - 19) ** 2) / 6.0)
        return 0.45 + 0.9 * morning + 1.3 * evening
    if site.segment == "commercial":
        working = 1.0 if 8 <= h <= 18 else 0.25
        return working * (0.4 if weekend else 1.0)
    return 0.85 + 0.15 * math.cos(2 * math.pi * (h - 14) / 24.0)


def _build_sites(cfg: GeneratorConfig, rng: Random) -> list[_Site]:
    sites: list[_Site] = []
    for i in range(1, cfg.n_sites + 1):
        segment = SEGMENTS[(i - 1) % len(SEGMENTS)]
        base = {"residential": 0.9, "commercial": 6.0, "industrial": 24.0}[segment]
        base *= rng.uniform(0.7, 1.3)
        contracted = {"residential": 6.9, "commercial": 41.4, "industrial": 250.0}[segment]
        sites.append(
            _Site(
                site_id=f"S{i:03d}",
                name=f"{segment.title()} site {i:03d}",
                region=cfg.regions[(i - 1) % len(cfg.regions)],
                segment=segment,
                tariff=rng.choice(TARIFFS),
                contracted_kw=contracted,
                has_solar=(i % 4 == 0),
                base_kw=base,
            )
        )
    return sites


def _schedule_anomalies(cfg: GeneratorConfig, sites: list[_Site], rng: Random) -> None:
    start = datetime.combine(cfg.start_date, datetime.min.time())
    total_hours = cfg.n_days * 24
    for idx, site in enumerate(sites):
        if total_hours > 48:
            o_start = rng.randrange(12, total_hours - 12)
            site.outage = (start + timedelta(hours=o_start), rng.randint(3, 8))
            m_start = rng.randrange(12, total_hours - 12)
            site.missing = (start + timedelta(hours=m_start), rng.randint(2, 5))
        if idx == 1 and total_hours > 72:
            site.spike = (start + timedelta(hours=rng.randrange(24, total_hours - 24)), 4)


def _sites_feed(cfg: GeneratorConfig, sites: list[_Site]) -> Files:
    """Site master snapshots, including tariff changes mid-period (SCD2 material)."""
    rows: list[Row] = []
    commissioned = cfg.start_date - timedelta(days=365)
    change_day = cfg.start_date + timedelta(days=max(1, cfg.n_days // 2))
    for i, s in enumerate(sites):
        base_row = {
            "site_id": s.site_id,
            "site_name": s.name,
            "region": s.region,
            "customer_segment": s.segment,
            "tariff_plan": s.tariff,
            "contracted_kw": _fmt(s.contracted_kw, 1),
            "effective_from": commissioned.isoformat(),
        }
        rows.append(base_row)
        if i % 3 == 0:  # tariff change -> new SCD2 version
            new_tariff = TARIFFS[(TARIFFS.index(s.tariff) + 1) % len(TARIFFS)]
            rows.append({**base_row, "tariff_plan": new_tariff, "effective_from": change_day.isoformat()})
            s.tariff = new_tariff
        if i == 0:  # unchanged re-snapshot: must be collapsed, not versioned
            rows.append(
                {
                    **base_row,
                    "effective_from": (cfg.start_date + timedelta(days=10)).isoformat(),
                }
            )
    return {"sites_snapshot.csv": rows}


def _weather_for(cfg: GeneratorConfig, rng: Random) -> dict[tuple[str, datetime], dict[str, float]]:
    out: dict[tuple[str, datetime], dict[str, float]] = {}
    start = datetime.combine(cfg.start_date, datetime.min.time())
    for region in cfg.regions:
        cloud = 1.0
        for h in range(cfg.n_days * 24):
            ts = start + timedelta(hours=h)
            if ts.hour == 0:
                cloud = rng.uniform(0.35, 1.0)
            out[(region, ts)] = {
                "temperature_c": _day_of_year_temp(region, ts, rng),
                "wind_speed_ms": max(0.0, rng.gauss(5.5, 2.5)),
                "solar_irradiance_wm2": _daylight(ts) * 920.0 * cloud * rng.uniform(0.9, 1.0),
                "humidity_pct": min(100.0, max(20.0, rng.gauss(68, 12))),
            }
    return out


def generate(cfg: GeneratorConfig = GeneratorConfig()) -> SyntheticDataset:  # noqa: B008
    """Produce every feed as ``{feed: {filename: [row, ...]}}`` with all values as strings."""
    rng = Random(cfg.seed)
    sites = _build_sites(cfg, rng)
    if cfg.inject_anomalies:
        _schedule_anomalies(cfg, sites, rng)
    weather = _weather_for(cfg, rng)
    start = datetime.combine(cfg.start_date, datetime.min.time())

    readings_files: Files = {}
    weather_files: Files = {}
    price_files: Files = {}
    gen_files: Files = {}
    reading_seq = 0
    carry_over: list[Row] = []  # restated readings landing in the next daily file

    for d in range(cfg.n_days):
        day = cfg.start_date + timedelta(days=d)
        fname = f"{day.isoformat()}.csv"
        day_rows: list[Row] = list(carry_over)
        carry_over = []
        w_rows: list[Row] = []
        p_rows: list[Row] = []
        g_rows: list[Row] = []

        for hh in range(24):
            ts = start + timedelta(days=d, hours=hh)
            for region in cfg.regions:
                wx = weather[(region, ts)]
                w_rows.append(
                    {
                        "region": region,
                        "observed_ts": _fmt_ts(ts),
                        "temperature_c": _fmt(wx["temperature_c"], 2),
                        "wind_speed_ms": _fmt(wx["wind_speed_ms"], 2),
                        "solar_irradiance_wm2": _fmt(wx["solar_irradiance_wm2"], 1),
                        "humidity_pct": _fmt(wx["humidity_pct"], 1),
                    }
                )
                # Day-ahead price: base + demand shape + wind discount + rare spike
                shape = 1.0 + 0.35 * math.sin(2 * math.pi * (ts.hour - 6) / 24.0) ** 2
                price = 62.0 * shape - 1.6 * wx["wind_speed_ms"] + rng.gauss(0, 4)
                if rng.random() < 0.004:
                    price *= 3.0
                p_rows.append(
                    {
                        "region": region,
                        "price_ts": _fmt_ts(ts),
                        "price_eur_mwh": _fmt(max(price, -5.0), 2),
                        "market": "day_ahead",
                    }
                )
                # Generation mix (MW) per source
                solar = wx["solar_irradiance_wm2"] / 920.0 * 1400.0
                wind = min(1.0, wx["wind_speed_ms"] / 12.0) ** 3 * 2200.0
                hydro = 600.0 + 120.0 * math.sin(2 * math.pi * d / 30.0)
                nuclear = 1000.0 if region == "north" else 0.0
                demand = 3800.0 * shape
                gas = max(0.0, demand - solar - wind - hydro - nuclear)
                for source, mw in zip(SOURCES, (solar, wind, hydro, gas, nuclear), strict=True):
                    g_rows.append(
                        {
                            "region": region,
                            "generation_ts": _fmt_ts(ts),
                            "source": source,
                            "generation_mw": _fmt(mw * rng.uniform(0.97, 1.03), 1),
                        }
                    )

            for site in sites:
                if site.missing and site.missing[0] <= ts < site.missing[0] + timedelta(hours=site.missing[1]):
                    continue  # rows simply never arrive -> gap
                wx = weather[(site.region, ts)]
                temp = wx["temperature_c"]
                kw = site.base_kw * _profile(site, ts)
                if temp < 15:
                    kw *= 1 + 0.03 * (15 - temp)
                elif temp > 24:
                    kw *= 1 + 0.04 * (temp - 24)
                kw *= 1 + rng.gauss(0, 0.08)
                flag = "OK"
                if site.outage and site.outage[0] <= ts < site.outage[0] + timedelta(hours=site.outage[1]):
                    kw, flag = 0.0, "OUTAGE"
                if site.spike and site.spike[0] <= ts < site.spike[0] + timedelta(hours=site.spike[1]):
                    kw *= 5.0
                exported = 0.0
                if site.has_solar:
                    exported = max(0.0, wx["solar_irradiance_wm2"] / 1000.0 * 3.2 - kw * 0.3)

                reading_seq += 1
                ts_out = ts
                if cfg.inject_anomalies and rng.random() < 0.02:
                    ts_out = ts + timedelta(minutes=rng.randint(1, 5))
                kwh_str = _fmt(max(kw, 0.0))
                if cfg.inject_anomalies:
                    r = rng.random()
                    if r < 0.005:
                        kwh_str, flag = "", "MISSING"
                    elif r < 0.007:
                        kwh_str, flag = _fmt(-abs(kw) - 1.0), "ERROR"
                row = {
                    "reading_id": f"R{reading_seq:09d}",
                    "site_id": site.site_id,
                    "reading_ts": _fmt_ts(ts_out),
                    "kwh_consumed": kwh_str,
                    "kwh_exported": _fmt(exported),
                    "voltage_v": _fmt(rng.gauss(230, 2.5), 1),
                    "quality_flag": flag,
                }
                day_rows.append(row)
                if cfg.inject_anomalies:
                    r = rng.random()
                    if r < 0.01:
                        day_rows.append(dict(row))  # exact duplicate in the same file
                    elif r < 0.015 and flag == "OK":
                        reading_seq += 1
                        carry_over.append(  # restated value, lands tomorrow
                            {
                                **row,
                                "reading_id": f"R{reading_seq:09d}",
                                "kwh_consumed": _fmt(max(kw, 0.0) * 1.05),
                                "quality_flag": "RESTATED",
                            }
                        )
                        site.restated.add(ts)

        readings_files[fname] = day_rows
        weather_files[fname] = w_rows
        price_files[fname] = p_rows
        gen_files[fname] = g_rows

    if carry_over:  # flush restatements for the final day into an extra file
        readings_files[f"{(cfg.start_date + timedelta(days=cfg.n_days)).isoformat()}.csv"] = carry_over

    return {
        "meter_readings": readings_files,
        "sites": _sites_feed(cfg, sites),
        "weather": weather_files,
        "prices": price_files,
        "generation_mix": gen_files,
    }


def write_csv(dataset: SyntheticDataset, root: str | Path) -> list[Path]:
    """Write ``dataset`` as ``<root>/<feed>/<file>.csv`` and return the written paths."""
    root = Path(root)
    written: list[Path] = []
    for feed, files in dataset.items():
        feed_dir = root / feed
        feed_dir.mkdir(parents=True, exist_ok=True)
        for fname, rows in files.items():
            path = feed_dir / fname
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FEEDS[feed])
                writer.writeheader()
                writer.writerows(rows)
            written.append(path)
    return written


def row_count(dataset: SyntheticDataset, feed: str) -> int:
    return sum(len(rows) for rows in dataset[feed].values())
