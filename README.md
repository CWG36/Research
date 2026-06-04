# panel-screen-dsl

A working recreation of the **point-in-time, factor-aware equity-screening
pipeline DSL** shown in
[@doodlestein's post](https://x.com/doodlestein/status/2062360758864785740).

That post showed Section 3 ("Operators") of a specification for a skill library
used by AI agents to build fundamental long/short equity screens: ~60 composable
primitives, organized into Source / Reshape / Windowed / Rolling / Trend &
Regression / Cross-Sectional / Predicate / Join / Aggregation / Column /
Composite. This repository turns that reference into a real, runnable Python
library and documents it faithfully in
[`docs/OPERATOR_REFERENCE.md`](docs/OPERATOR_REFERENCE.md).

## Design

A `Frame` wraps a **long-form panel** `DataFrame` whose contract is:

| column   | meaning                            |
| -------- | ---------------------------------- |
| `entity` | the security (e.g. a ticker)       |
| `date`   | the as-of / observation date       |
| values   | one or more metric columns         |

Properties that mirror the spec:

- **Immutable & chainable** — every operator returns a *new* `Frame`, so a
  screen is a deterministic DAG of method calls.
- **Point-in-time by construction** — `cross_section` takes the latest row at or
  before an as-of date; `lead` (the only forward-looking operator) refuses
  unless lookahead is explicitly acknowledged.
- **Refuse, don't coerce** — operators raise typed errors (`SourceError`,
  `SchemaError`, `ConversionError`, `LookaheadError`, `CoverageError`) with
  stable `code`s instead of silently inventing data.
- **Per-entity vs per-date** — windowed/rolling/trend ops run per entity in date
  order; cross-sectional ops run per date across entities.

## Quickstart

```python
import pandas as pd
from panel_screen_dsl import Frame, SourceRegistry

panel = pd.DataFrame({
    "entity": ["AAA", "AAA", "BBB", "BBB"],
    "date":   pd.to_datetime(["2023-09-30", "2023-12-31"] * 2),
    "metric_name": ["revenue"] * 4,
    "value":  [121, 133, 202, 210],
})

reg = SourceRegistry().declare("fundamentals", panel)

screen = (
    Frame.load(reg, "fundamentals")   # 3.1 Source
    .pivot()                           # 3.2 Reshape -> wide
    .pct_change(1, col="revenue")      # 3.3 Windowed (per entity)
    .z_score(col="revenue_pct1")       # 3.6 Cross-sectional (per date)
    .top_n(1, col="revenue_pct1_z")    # 3.7 Predicate
    .materialize("long_candidates")    # 3.1 Source
)
print(screen.df)
```

A fuller worked example — a momentum + value long/short screen with
sector-neutralization — lives in
[`examples/momentum_value_screen.py`](examples/momentum_value_screen.py):

```bash
PYTHONPATH=. python examples/momentum_value_screen.py
```

## Install & test

```bash
pip install -e ".[dev]"   # or: pip install pandas numpy pytest
pytest -q
```

## Layout

```
panel_screen_dsl/
  frame.py      # the Frame class — all ~60 operators, grouped by section
  sources.py    # SourceRegistry: declared, entitlement/coverage-aware sources
  errors.py     # typed refusals with stable codes
docs/
  OPERATOR_REFERENCE.md   # the recreated Section 3 spec
examples/
  momentum_value_screen.py
tests/
  test_operators.py       # coverage across every section
```

## Scope & caveats

This recreates the **public operator reference** from the screenshots, not the
proprietary data backend behind it. The data sources here are an in-memory
`SourceRegistry` rather than a real HFDT feed, and the regression operators use
straightforward NumPy least-squares rather than any production estimator. The
operator surface, semantics, and refusal behavior follow the spec.

## License

MIT
