"""Shared inputs and helpers for gold builders."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column


@dataclass(frozen=True)
class SilverInputs:
    readings: DataFrame
    readings_enriched: DataFrame  # readings + point-in-time site attributes (SCD2 lookup)
    sites: DataFrame
    weather: DataFrame
    prices: DataFrame
    generation: DataFrame


def pct_change(current: Column, previous: Column, scale: int = 2) -> Column:
    """``(current - previous) / |previous| * 100``; null when previous is null or zero."""
    return F.when(
        previous.isNotNull() & (previous != 0),
        F.round((current - previous) / F.abs(previous) * 100, scale),
    )


def month_start(ts: str | Column) -> Column:
    col = F.col(ts) if isinstance(ts, str) else ts
    return F.date_trunc("month", col).cast("date")
