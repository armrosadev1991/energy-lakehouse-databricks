"""Pipeline configuration shared by every layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

LAYERS = ("bronze", "silver", "gold", "ops")


@dataclass(frozen=True)
class PipelineConfig:
    """Where tables live and where raw files land.

    ``catalog`` / ``schema`` are Unity Catalog names on Databricks. In local tests the
    catalog is ``spark_catalog`` and the schema is a throw-away database.
    """

    catalog: str
    schema: str
    landing_path: str
    table_format: str = "delta"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @classmethod
    def for_databricks(cls, catalog: str, schema: str, landing_path: str | None = None) -> PipelineConfig:
        """Build a config that targets Unity Catalog with a managed landing volume."""
        return cls(
            catalog=catalog,
            schema=schema,
            landing_path=landing_path or f"/Volumes/{catalog}/{schema}/landing",
        )

    @property
    def qualified_schema(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`"

    def table(self, layer: str, name: str) -> str:
        """Fully qualified table name, e.g. ``main.energy.silver_meter_readings``."""
        if layer not in LAYERS:
            msg = f"unknown layer {layer!r}; expected one of {LAYERS}"
            raise ValueError(msg)
        return f"{self.qualified_schema}.`{layer}_{name}`"

    def feed_path(self, feed: str) -> str:
        return f"{self.landing_path.rstrip('/')}/{feed}"
