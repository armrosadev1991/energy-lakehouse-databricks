"""Console entry point: ``main`` is referenced by ``python_wheel_task.entry_point``.

Examples (locally or as job task parameters)::

    main --catalog main --schema energy --stage bronze
    main --catalog main --schema energy --stage all --n-sites 20 --n-days 90
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date

from energy_lakehouse.config import PipelineConfig
from energy_lakehouse.pipeline import STAGES, GenerateOptions, run_stages
from energy_lakehouse.spark_session import get_spark


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="energy-lakehouse", description=__doc__)
    p.add_argument("--catalog", required=True, help="Unity Catalog catalog name")
    p.add_argument("--schema", required=True, help="schema holding bronze/silver/gold tables")
    p.add_argument(
        "--landing-path",
        default=None,
        help="raw file landing area (default: /Volumes/<catalog>/<schema>/landing)",
    )
    p.add_argument(
        "--stage",
        choices=[*STAGES, "all"],
        default="all",
        help="which layer to run (default: all)",
    )
    p.add_argument("--n-sites", type=int, default=12, help="[generate] number of meters")
    p.add_argument("--n-days", type=int, default=60, help="[generate] number of days")
    p.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date(2025, 1, 1),
        help="[generate] first day",
    )
    p.add_argument("--seed", type=int, default=42, help="[generate] random seed")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = PipelineConfig.for_databricks(args.catalog, args.schema, args.landing_path)
    gen = GenerateOptions(args.n_sites, args.n_days, args.start_date, args.seed)
    stages = STAGES if args.stage == "all" else (args.stage,)
    run_stages(get_spark(), cfg, stages, gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
