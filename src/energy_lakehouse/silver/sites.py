"""Silver site dimension as a Slowly Changing Dimension Type 2.

Window functions used here
--------------------------
* ``row_number()`` — de-duplicate snapshots that share ``(site_id, effective_from)``
* ``lag(attr_hash)`` — collapse consecutive snapshots whose attributes did not change
* ``lead(effective_from)`` — close each version with the next version's start date
* ``row_number()`` again — version number per site
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from energy_lakehouse.silver.casting import try_date, try_double

SITE_KEYS = ("site_id", "valid_from")
TRACKED_ATTRIBUTES = ("site_name", "region", "customer_segment", "tariff_plan", "contracted_kw")
OPEN_ENDED = "9999-12-31"


def cast_sites(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        F.trim("site_id").alias("site_id"),
        F.trim("site_name").alias("site_name"),
        F.lower(F.trim("region")).alias("region"),
        F.lower(F.trim("customer_segment")).alias("customer_segment"),
        F.lower(F.trim("tariff_plan")).alias("tariff_plan"),
        try_double("contracted_kw").alias("contracted_kw"),
        try_date("effective_from").alias("effective_from"),
        "_ingested_at",
    ).filter(F.col("site_id").isNotNull() & F.col("effective_from").isNotNull())


def build_silver_sites(bronze: DataFrame) -> DataFrame:
    typed = cast_sites(bronze)
    attr_hash = F.sha2(
        F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in TRACKED_ATTRIBUTES]),
        256,
    )
    snapshot_latest = Window.partitionBy("site_id", "effective_from").orderBy(F.col("_ingested_at").desc())
    by_site = Window.partitionBy("site_id").orderBy("effective_from")

    deduped = (
        typed.withColumn("_rn", F.row_number().over(snapshot_latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_ingested_at")
        .withColumn("attr_hash", attr_hash)
    )
    # A snapshot only opens a new version when something tracked actually changed.
    changed = (
        deduped.withColumn("_prev_hash", F.lag("attr_hash").over(by_site))
        .filter(F.col("_prev_hash").isNull() | (F.col("_prev_hash") != F.col("attr_hash")))
        .drop("_prev_hash")
    )
    return (
        changed.withColumn("valid_from", F.col("effective_from"))
        .withColumn("valid_to", F.lead("effective_from").over(by_site))
        .withColumn("is_current", F.col("valid_to").isNull())
        .withColumn("version", F.row_number().over(by_site))
        .drop("effective_from")
    )


def as_of_condition(fact_ts: Column, dim_alias: str = "d") -> Column:
    """Point-in-time predicate: fact timestamp falls inside the dimension version."""
    day = F.to_date(fact_ts)
    return (day >= F.col(f"{dim_alias}.valid_from")) & (
        day < F.coalesce(F.col(f"{dim_alias}.valid_to"), F.lit(OPEN_ENDED).cast("date"))
    )


def join_site_as_of(facts: DataFrame, sites: DataFrame, ts_col: str = "reading_hour") -> DataFrame:
    """Attach the site version that was valid when each fact happened (SCD2 lookup)."""
    f, d = facts.alias("f"), sites.alias("d")
    return f.join(
        d,
        (F.col("f.site_id") == F.col("d.site_id")) & as_of_condition(F.col(f"f.{ts_col}")),
        "left",
    ).select(
        "f.*",
        F.col("d.region").alias("region"),
        F.col("d.customer_segment").alias("customer_segment"),
        F.col("d.tariff_plan").alias("tariff_plan"),
        F.col("d.contracted_kw").alias("contracted_kw"),
        F.col("d.version").alias("site_version"),
    )
