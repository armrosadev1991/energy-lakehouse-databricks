"""gold_outage_streaks — "gaps and islands" over the hourly series.

Two kinds of streak are detected and unioned:

* ``ZERO_CONSUMPTION`` — consecutive hours where the meter reported 0 kWh (or the source
  flagged an outage). Islands are found with the classic trick: ``hour_index - row_number()``
  over the flagged rows is constant for each run of consecutive hours.
* ``MISSING_DATA`` — hours where no row arrived at all, derived from the ``gap_hours_before``
  feature that silver computed with ``lag()``.

Then ``dense_rank`` ranks each site's streaks by duration and ``lead``/``lag`` measure the
spacing between consecutive streaks.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energy_lakehouse.gold.common import SilverInputs


def build_outage_streaks(spark: SparkSession, inputs: SilverInputs) -> DataFrame:
    return outage_streaks(inputs.readings)


def zero_consumption_islands(readings: DataFrame) -> DataFrame:
    flagged = readings.filter((F.col("kwh_filled") <= 0) | (F.col("quality_flag") == "OUTAGE")).withColumn(
        "_hour_index", (F.unix_timestamp("reading_hour") / 3600).cast("long")
    )
    ordered = Window.partitionBy("site_id").orderBy("reading_hour")
    return (
        flagged.withColumn("_island", F.col("_hour_index") - F.row_number().over(ordered))
        .groupBy("site_id", "_island")
        .agg(
            F.min("reading_hour").alias("streak_start"),
            F.max("reading_hour").alias("streak_end"),
            F.count("*").alias("streak_hours"),
        )
        .withColumn("streak_kind", F.lit("ZERO_CONSUMPTION"))
        .drop("_island")
    )


def missing_data_islands(readings: DataFrame) -> DataFrame:
    gap = F.col("gap_hours_before").cast("long")
    return readings.filter(F.col("gap_hours_before") > 0).select(
        "site_id",
        (F.col("reading_hour") - F.expr("make_interval(0, 0, 0, 0, gap_hours_before)")).alias("streak_start"),
        (F.col("reading_hour") - F.expr("INTERVAL 1 HOUR")).alias("streak_end"),
        gap.alias("streak_hours"),
        F.lit("MISSING_DATA").alias("streak_kind"),
    )


def outage_streaks(readings: DataFrame) -> DataFrame:
    streaks = zero_consumption_islands(readings).unionByName(missing_data_islands(readings))
    by_duration = Window.partitionBy("site_id").orderBy(F.col("streak_hours").desc())
    chronological = Window.partitionBy("site_id").orderBy("streak_start")
    next_start = F.lead("streak_start").over(chronological)
    prev_end = F.lag("streak_end").over(chronological)
    return (
        streaks.withColumn("streak_rank_by_duration", F.dense_rank().over(by_duration))
        .withColumn("is_longest_streak", F.col("streak_rank_by_duration") == 1)
        .withColumn("streak_seq", F.row_number().over(chronological))
        .withColumn(
            "hours_until_next_streak",
            (F.unix_timestamp(next_start) - F.unix_timestamp("streak_end")) / 3600 - 1,
        )
        .withColumn(
            "hours_since_prev_streak",
            (F.unix_timestamp("streak_start") - F.unix_timestamp(prev_end)) / 3600 - 1,
        )
        .withColumn("streak_date", F.to_date("streak_start"))
    )
