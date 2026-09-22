# Window functions used in this project — a guided tour

Every example below is real code from `src/energy_lakehouse`, with the input/expected output the
unit tests assert. Column names follow the tables described in the README.

## 1. `row_number()` — keep one row per key

`silver/readings.py`. Restatements and exact duplicates arrive for the same site-hour; the most
recently ingested file wins, then the highest source sequence number.

```python
latest_first = Window.partitionBy("site_id", "reading_hour").orderBy(
    F.col("_ingested_at").desc(), F.col("reading_id").desc()
)
df.withColumn("_rn", F.row_number().over(latest_first)).filter("_rn = 1")
```

## 2. `date_trunc` — align to the hour, roll up to the day/month

Readings can be stamped `08:03:00`; `F.date_trunc("hour", reading_ts)` gives the hour bucket that
becomes the silver key. Daily and monthly gold tables group by `date_trunc('day'|'month', ...)`.

## 3. `lag` / `lead` — previous and next rows

| Where | Expression | Meaning |
| --- | --- | --- |
| silver readings | `lag("reading_hour")` | hours since the previous reading → `gap_hours_before` |
| silver readings | `lag("kwh_filled")` | `kwh_prev_hour`, `kwh_delta_vs_prev` |
| silver prices | `lag("price_eur_mwh")` | hour-over-hour price change |
| silver weather | `lag("temperature_c", 24)` | same hour yesterday |
| gold daily | `lag(kwh_total, 1)`, `lag(kwh_total, 7)`, `lead(kwh_total, 1)` | DoD %, WoW %, next day |
| gold monthly (SQL) | `LAG(kwh_total) OVER (PARTITION BY site_id ORDER BY month_start)` | MoM % |
| silver sites | `lag(attr_hash)` / `lead(effective_from)` | SCD2 change detection / closing a version |

`lag(7)` is **row**-based: if a day is missing, "the same day last week" silently becomes eight days
ago. `test_rows_frame_vs_calendar_range_frame` demonstrates this and the RANGE alternative below.

## 4. Forward-fill with `last(..., ignorenulls=True)`

```python
history = (
    Window.partitionBy("site_id").orderBy("reading_hour").rowsBetween(Window.unboundedPreceding, Window.currentRow)
)
kwh_filled = F.coalesce(F.col("kwh_consumed"), F.last("kwh_consumed", ignorenulls=True).over(history), F.lit(0.0))
```

Input `1.0, 2.5, NULL, 4.0` → `1.0, 2.5, 2.5, 4.0` with `is_imputed = true` on the third row.

## 5. `rank` vs `dense_rank` vs `row_number` on ties

`gold/peaks.py`. Hourly kWh `5, 9, 9, 2, 7` ordered descending:

| hour | kWh | rank | dense_rank | row_number |
| --- | --- | --- | --- | --- |
| 1 | 9 | 1 | 1 | 1 |
| 2 | 9 | 1 | 1 | 2 |
| 4 | 7 | 3 | 2 | 3 |
| 0 | 5 | 4 | 3 | 4 |
| 3 | 2 | 5 | 4 | 5 |

`rank` leaves a hole after ties, `dense_rank` does not, `row_number` needs a tiebreaker
(`orderBy(kwh.desc(), reading_hour)`) to be deterministic.

## 6. `percent_rank`, `cume_dist`, `ntile`

`gold/monthly.py` (SQL) and `gold/distribution.py`. Five sites with monthly totals
`5, 10, 20, 20, 40` ordered ascending:

| kWh | percent_rank = (rank−1)/(n−1) | cume_dist = rows ≤ current / n | ntile(4) |
| --- | --- | --- | --- |
| 5 | 0.00 | 0.2 | 1 |
| 10 | 0.25 | 0.4 | 1 |
| 20 | 0.50 | 0.8 | 2 or 3 |
| 20 | 0.50 | 0.8 | 2 or 3 |
| 40 | 1.00 | 1.0 | 4 |

`ntile` splits rows into buckets as evenly as possible (sizes 2,1,1,1 here); which tied row lands in
which bucket is not defined, so tests only assert the set.

## 7. Aggregates as windows — share of a group

```sql
ROUND(try_divide(kwh_total, SUM(kwh_total) OVER (PARTITION BY month_start, region)) * 100, 2)
    AS share_of_region_pct
```

Every row keeps its detail and sees its group's total. `try_divide` (not `/`) because ANSI mode
turns a zero denominator into an error.

## 8. `first_value` / `last_value` and explicit frames

The default frame with an `ORDER BY` is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, so
`last_value` would just return the current row. Always widen it:

```sql
LAST_VALUE(site_id) OVER (PARTITION BY month_start, region ORDER BY kwh_total DESC
                          ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
```

## 9. Running totals — `ROWS UNBOUNDED PRECEDING`

```python
month_to_date = (
    Window.partitionBy("site_id", F.date_trunc("month", "reading_date"))
    .orderBy("reading_date")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)
F.sum("kwh_total").over(month_to_date)
```

The partition on the truncated month makes the total restart every month.

## 10. Moving averages — ROWS vs RANGE frames

```python
series = Window.partitionBy("site_id").orderBy("reading_date")
day_num = F.datediff("reading_date", F.lit("1970-01-01"))
calendar_30 = Window.partitionBy("site_id").orderBy(day_num).rangeBetween(-29, 0)

F.avg("kwh_total").over(series.rowsBetween(-6, 0))  # last 7 *rows*
F.avg("kwh_total").over(calendar_30)  # last 30 *days*, by value of day_num
```

With daily totals on Jan 1–3 and Feb 9: the ROWS average on Feb 9 is `(10+20+30+100)/4 = 40`; the
RANGE average is `100` because nothing else falls within the previous 29 days.

## 11. Trailing baseline that excludes the current row

`gold/anomalies.py`: `rowsBetween(-168, -1)` is the previous 168 hourly rows, *not* including the
row being scored, so a spike cannot inflate its own baseline.

```python
z = (kwh - avg(kwh).over(trailing)) / stddev_samp(kwh).over(trailing)  # flag |z| > 3
```

## 12. Gaps and islands

`gold/outages.py`. For rows flagged as zero consumption, `hour_index − row_number()` is constant
across a run of *consecutive* hours, so grouping by it yields one row per streak:

```python
flagged.withColumn("_island", F.col("_hour_index") - F.row_number().over(ordered))
       .groupBy("site_id", "_island").agg(min("reading_hour"), max("reading_hour"), count("*"))
```

Using the hour index (not a plain row counter) means two zero hours separated by *missing* rows
correctly form two islands. Missing-data holes come from silver's `gap_hours_before` (a `lag()`).

## 13. SCD Type 2 and point-in-time joins

`silver/sites.py`:

1. `sha2(concat_ws('||', tracked attributes))` → `attr_hash`
2. `lag(attr_hash)` → drop snapshots whose hash equals the previous one (no real change)
3. `lead(effective_from)` → `valid_to`; `is_current = valid_to IS NULL`; `row_number()` → `version`
4. Facts join on `site_id` **and** `valid_from <= to_date(ts) < coalesce(valid_to, '9999-12-31')`

## 14. `GROUPING SETS` + `grouping_id`

`gold/rollup.py` computes month / region / segment / tariff totals in one scan. `grouping_id`
has one bit per grouping column (1 = rolled up), so `0` is the finest grain and `7` the month total;
a `MAX(...) OVER (PARTITION BY month_start)` window then expresses every row as a share of its month.

## 15. `pivot`, `max_by`, `percentile_approx`

- `generation.groupBy("region", "generation_hour").pivot("source", SOURCES).agg(sum("generation_mw"))`
  → one `<source>_mw` column per source.
- `F.max_by("source", "generation_mw")` → the dominant source without a self-join.
- `F.percentile_approx("kwh_total", 0.5)` per segment-month, joined back so each site can be
  compared to its segment's median.
