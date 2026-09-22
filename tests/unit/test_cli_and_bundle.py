import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from energy_lakehouse.cli import build_parser
from energy_lakehouse.pipeline import STAGES

ROOT = Path(__file__).resolve().parents[2]


def test_cli_defaults() -> None:
    args = build_parser().parse_args(["--catalog", "main", "--schema", "energy"])
    assert args.stage == "all" and args.landing_path is None and args.n_sites == 12
    args = build_parser().parse_args(
        ["--catalog", "c", "--schema", "s", "--stage", "gold", "--start-date", "2024-06-01"]
    )
    assert args.stage == "gold" and args.start_date.isoformat() == "2024-06-01"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--catalog", "c", "--schema", "s", "--stage", "platinum"])


def _bundle_docs() -> tuple[dict, list[dict]]:  # type: ignore[type-arg]
    root = yaml.safe_load((ROOT / "databricks.yml").read_text())
    resources = [yaml.safe_load(p.read_text()) for pattern in root["include"] for p in sorted(ROOT.glob(pattern))]
    return root, resources


def test_bundle_wiring_matches_the_package() -> None:
    root, resources = _bundle_docs()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    jobs = {k: v for doc in resources for k, v in doc.get("resources", {}).get("jobs", {}).items()}
    job = jobs["energy_medallion_pipeline"]

    env_keys = {e["environment_key"] for e in job["environments"]}
    stages_seen = []
    for task in job["tasks"]:
        assert task["environment_key"] in env_keys, "serverless wheel tasks need an environment_key"
        wheel = task["python_wheel_task"]
        assert wheel["package_name"] == pyproject["project"]["name"]
        assert wheel["entry_point"] in pyproject["project"]["scripts"]
        params = wheel["parameters"]
        stages_seen.append(params[params.index("--stage") + 1])
    assert stages_seen == list(STAGES), "job tasks must run the stages in medallion order"

    # a strict chain: every task except the first depends on exactly the previous one
    keys = [t["task_key"] for t in job["tasks"]]
    for prev, task in zip(keys, job["tasks"][1:], strict=False):
        assert [d["task_key"] for d in task["depends_on"]] == [prev]

    assert job["environments"][0]["spec"]["dependencies"] == ["../dist/*.whl"]
    assert root["artifacts"]["energy_lakehouse"]["type"] == "whl"
    assert {"dev", "prod"} <= set(root["targets"])
    assert root["targets"]["dev"]["mode"] == "development" and root["targets"]["prod"]["mode"] == "production"
    assert "run_as" in root["targets"]["prod"], "production deployments should run as a service principal"


@pytest.mark.skipif(shutil.which("databricks") is None, reason="databricks CLI not installed")
def test_bundle_config_against_cli_json_schema() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/check_bundle_schema.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
