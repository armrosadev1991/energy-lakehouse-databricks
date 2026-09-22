# fixtures

`sample/` is a tiny, anomaly-free output of `energy_lakehouse.datagen` (2 sites, 1 day) so the
shape of every feed can be read on GitHub. Tests do not read these files: they call the generator
directly (it is deterministic) and build small DataFrames by hand.

Regenerate with:

```bash
uv run python -c "from datetime import date; from energy_lakehouse.datagen import *; \
write_csv(generate(GeneratorConfig(n_sites=2, n_days=1, start_date=date(2025,1,1), seed=1, inject_anomalies=False)), 'fixtures/sample')"
```
