"""A worked long/short screen built entirely from DSL operators.

Run with::

    python examples/momentum_value_screen.py

The screen demonstrates how the operators chain into a deterministic DAG:
ingest -> reshape -> per-entity windowed features -> cross-sectional
normalization -> factor neutralization -> composite score -> selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from panel_screen_dsl import Frame, SourceRegistry


def synthetic_universe(n_entities: int = 40, n_periods: int = 24, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic long-form panel of monthly price + book_yield + sector."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=n_periods, freq="ME")
    sectors = ["tech", "energy", "health", "fin"]
    rows = []
    for i in range(n_entities):
        ent = f"E{i:03d}"
        sector = sectors[i % len(sectors)]
        price = 100.0
        drift = rng.normal(0.01, 0.01)
        book_yield = abs(rng.normal(0.06, 0.03))
        for d in dates:
            price *= 1 + drift + rng.normal(0, 0.05)
            rows.append({"entity": ent, "date": d, "metric_name": "price", "value": price})
            rows.append({"entity": ent, "date": d, "metric_name": "book_yield", "value": book_yield})
            rows.append({"entity": ent, "date": d, "metric_name": "sector", "value": sector})
    return pd.DataFrame(rows)


def build_screen(registry: SourceRegistry) -> Frame:
    base = (
        Frame.load(registry, "market")          # 3.1 Source
        .pivot()                                  # 3.2 Reshape: long -> wide
    )

    # 3.3/3.4 Windowed + Rolling features (per entity, in date order)
    featured = (
        base
        .pct_change(12, col="price", out="mom_12m")          # 12-month momentum
        .rolling_std(6, col="price", out="vol_6m", min_periods=3)
    )

    # 3.6 Cross-Sectional normalization, factor-neutralized against sector & vol
    scored = (
        featured
        .z_score(col="mom_12m", out="mom_z", winsorize_pct=0.02)
        .z_score(col="book_yield", out="value_z", winsorize_pct=0.02)
        .neutralize_by_group("sector", col="mom_z", out="mom_neutral")
        # 3.11 Composite: deterministic weighted score
        .score_combine({"mom_neutral": 0.6, "value_z": 0.4}, out="alpha")
    )

    # 3.1 cross_section to a point-in-time candidate set, then 3.7 selection
    as_of = scored.df["date"].max()
    point_in_time = scored.cross_section(as_of)
    longs = point_in_time.top_n(5, col="alpha").materialize("long_book")
    shorts = point_in_time.bottom_n(5, col="alpha").materialize("short_book")
    return longs, shorts, as_of


def main() -> None:
    registry = SourceRegistry().declare("market", synthetic_universe())
    longs, shorts, as_of = build_screen(registry)
    print(f"As-of date: {as_of.date()}\n")
    cols = ["entity", "sector", "mom_z", "value_z", "alpha"]
    print("=== LONG book (highest alpha) ===")
    print(longs.df[cols].sort_values("alpha", ascending=False).to_string(index=False))
    print("\n=== SHORT book (lowest alpha) ===")
    print(shorts.df[cols].sort_values("alpha").to_string(index=False))


if __name__ == "__main__":
    main()
