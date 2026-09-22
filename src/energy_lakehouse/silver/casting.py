"""ANSI-safe casting helpers for the bronze -> silver boundary.

Spark 4 and current Databricks runtimes enable ANSI SQL mode by default, where ``CAST``
and ``to_timestamp`` raise on malformed input. Bronze is schema-on-read, so bad values
are expected; these helpers turn them into NULLs that the validation step can reject.
"""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.column import Column


def try_double(name: str) -> Column:
    return F.expr(f"try_cast(`{name}` AS DOUBLE)")


def try_date(name: str) -> Column:
    return F.try_to_timestamp(F.col(name)).cast("date")
