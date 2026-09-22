"""Validate every bundle YAML file against the JSON schema shipped with the Databricks CLI.

This catches typos in resource keys without needing workspace credentials (full
``databricks bundle validate`` also resolves ``${workspace.*}`` lookups, which does).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent.parent


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_bundle_config() -> dict[str, Any]:
    merged: dict[str, Any] = yaml.safe_load((ROOT / "databricks.yml").read_text())
    for pattern in merged.get("include", []):
        for path in sorted(ROOT.glob(pattern)):
            merged = deep_merge(merged, yaml.safe_load(path.read_text()) or {})
    return merged


def _python_regex(pattern: str) -> str:
    """Translate the Go/ECMA unicode classes the CLI schema uses into ``re``-compatible ones."""
    return (
        pattern.replace(r"[\p{L}\p{N}]", r"[^\W_]")  # letters or digits
        .replace(r"\p{L}", r"[^\W\d_]")  # letters
        .replace(r"\p{N}", r"\d")  # digits
    )


def portable_schema(node: Any) -> Any:
    r"""Rewrite the schema's ``${...}`` interpolation regexes so Python's ``re`` can compile them.

    The CLI's JSON schema is written for Go's regexp (``\p{L}``); nothing else changes, so
    ``oneOf`` branches, enums, key names and value types are all still checked exactly.
    """
    if isinstance(node, dict):
        return {
            k: (_python_regex(v) if k == "pattern" and isinstance(v, str) else portable_schema(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [portable_schema(v) for v in node]
    return node


def cli_schema() -> dict[str, Any]:
    if shutil.which("databricks") is None:
        msg = "databricks CLI not found on PATH"
        raise RuntimeError(msg)
    out = subprocess.run(["databricks", "bundle", "schema"], check=True, capture_output=True, text=True)
    return portable_schema(json.loads(out.stdout))  # type: ignore[no-any-return]


def leaf_errors(err: jsonschema.ValidationError) -> list[jsonschema.ValidationError]:
    """Descend through ``oneOf`` contexts to the deepest errors.

    Every object in the CLI schema is ``oneOf: [<object>, <"${...}" string>]``, so a typo deep
    inside a job is reported by jsonschema as "resources is not valid under any of the given
    schemas". The most deeply nested errors are the ones a human wants to read.
    """
    if not err.context:
        return [err]
    leaves = [leaf for child in err.context for leaf in leaf_errors(child)]
    # Prefer errors that point *inside* this node (a typo in a nested key), then errors from the
    # object branch itself (missing required key, unknown key); the interpolation branch only ever
    # says "not a string" / "does not match pattern", which is noise. If everything failed on the
    # value itself (e.g. a string where an int is expected), the summary line is enough.
    deeper = [leaf for leaf in leaves if len(leaf.absolute_path) > len(err.absolute_path)]
    if deeper:
        return deeper
    informative = [leaf for leaf in leaves if leaf.validator not in {"type", "pattern"}]
    return informative or [err]


def report(config: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = jsonschema.validators.validator_for(schema)(schema)
    seen: dict[tuple[str, str], None] = {}
    for err in validator.iter_errors(config):
        for leaf in leaf_errors(err):
            path = "/".join(str(p) for p in leaf.absolute_path) or "<root>"
            seen.setdefault((path, leaf.message[:200]), None)
    return [f"{path}: {message}" for path, message in seen]


def main() -> int:
    config = load_bundle_config()
    errors = report(config, cli_schema())
    if errors:
        for line in errors:
            print(f"✗ {line}")
        return 1
    n_jobs = len(config.get("resources", {}).get("jobs", {}))
    print(f"✓ bundle config valid against CLI schema ({n_jobs} job(s), targets: {sorted(config['targets'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
