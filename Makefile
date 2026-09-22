# Developer shortcuts. Everything CI runs is reproducible from here.
.PHONY: sync lint fmt typecheck test test-unit test-integration build validate deploy-dev run-dev check

sync:            ## install the locked dev environment
	uv sync --locked --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:             ## auto-format
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test-unit:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

test: test-unit test-integration

build:
	uv build --wheel

schema-check:    ## bundle YAML vs CLI JSON schema (offline)
	uv run python scripts/check_bundle_schema.py

validate:        ## full bundle validation against the dev workspace
	databricks bundle validate -t dev

deploy-dev:
	databricks bundle deploy -t dev

run-dev:
	databricks bundle run -t dev energy_medallion_pipeline

check: lint typecheck test schema-check
