"""Obtain a SparkSession that works on Databricks (classic or serverless) and locally."""

from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    """Return the active session, a Databricks Connect session, or a plain local session.

    On Databricks serverless compute ``databricks-connect`` is pre-installed and
    ``DatabricksSession`` is the supported way to obtain a session inside a wheel task.
    Locally (tests, notebooks) we fall back to the classic builder; tests inject their own
    Delta-enabled session via ``SparkSession.builder.getOrCreate`` before calling this.
    """
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()  # type: ignore[no-any-return]
    except ImportError:
        return SparkSession.builder.getOrCreate()
