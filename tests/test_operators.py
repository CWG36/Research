"""Tests exercising the panel-screen DSL operators across every section."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panel_screen_dsl import (
    ConversionError,
    Frame,
    LookaheadError,
    SourceError,
    SourceRegistry,
)


def make_long_panel() -> pd.DataFrame:
    """A small long-form panel: two entities, quarterly revenue + price."""
    dates = pd.to_datetime(
        ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]
    )
    rows = []
    rev = {"AAA": [100, 110, 121, 133, 146], "BBB": [200, 198, 202, 210, 205]}
    px = {"AAA": [10, 11, 12, 13, 14], "BBB": [50, 49, 51, 52, 50]}
    for ent in ("AAA", "BBB"):
        for d, r, p in zip(dates, rev[ent], px[ent]):
            rows.append({"entity": ent, "date": d, "metric_name": "revenue", "value": r})
            rows.append({"entity": ent, "date": d, "metric_name": "price", "value": p})
    return pd.DataFrame(rows)


@pytest.fixture
def registry() -> SourceRegistry:
    return SourceRegistry().declare("fundamentals", make_long_panel())


# ---------------------------------------------------------------- 3.1 Source
def test_load_and_refusals(registry):
    f = Frame.load(registry, "fundamentals")
    assert len(f) == 20
    with pytest.raises(SourceError):
        Frame.load(registry, "does_not_exist")
    reg2 = SourceRegistry().declare("empty", make_long_panel().iloc[:0])
    with pytest.raises(SourceError):
        Frame.load(reg2, "empty")
    reg3 = SourceRegistry().declare("locked", make_long_panel(), entitled=False)
    with pytest.raises(SourceError):
        Frame.load(reg3, "locked")


def test_cross_section_is_point_in_time(registry):
    cs = Frame.load(registry, "fundamentals").pivot().cross_section("2023-10-15")
    # one row per entity, taking the latest <= as_of (the 2023-09-30 row)
    assert len(cs) == 2
    assert set(cs.df["date"].unique()) == {pd.Timestamp("2023-09-30")}


def test_snapshot_narrows_range(registry):
    snap = Frame.load(registry, "fundamentals").snapshot("2023-06-01", "2023-09-30")
    assert snap.df["date"].min() == pd.Timestamp("2023-06-30")
    assert snap.df["date"].max() == pd.Timestamp("2023-09-30")


# --------------------------------------------------------------- 3.2 Reshape
def test_pivot_unpivot_roundtrip(registry):
    wide = Frame.load(registry, "fundamentals").pivot()
    assert {"revenue", "price"}.issubset(wide.df.columns)
    long = wide.unpivot(["revenue", "price"])
    assert len(long) == 20


def test_reindex_keeps_missing_observable(registry):
    cal = pd.to_datetime(["2023-03-31", "2099-01-01"])
    out = Frame.load(registry, "fundamentals").pivot().rename({"revenue": "value"}).reindex(cal)
    # 2 entities x 2 calendar dates; the invented future date is NaN, not dropped
    assert len(out) == 4
    assert out.df["value"].isna().sum() == 2


# -------------------------------------------------------------- 3.3 Windowed
def test_lag_pct_change_yoy(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    pct = f.pct_change(1, col="revenue")
    aaa = pct.df[pct.df["entity"] == "AAA"].sort_values("date")["revenue_pct1"].to_numpy()
    assert np.isnan(aaa[0])
    assert aaa[1] == pytest.approx(0.10)  # 100 -> 110
    yoy = f.yoy(col="revenue")  # periods=4 default
    aaa_yoy = yoy.df[yoy.df["entity"] == "AAA"].sort_values("date")["revenue_yoy"].to_numpy()
    assert aaa_yoy[4] == pytest.approx(146 / 100 - 1)


def test_lead_is_gated(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    with pytest.raises(LookaheadError):
        f.lead(1, col="revenue")
    ok = f.lead(1, col="revenue", acknowledge_lookahead=True)
    assert "revenue_lead1" in ok.df.columns


def test_log_return_refuses_nonpositive(registry):
    df = make_long_panel()
    df.loc[df["metric_name"] == "price", "value"] = -1.0
    reg = SourceRegistry().declare("neg", df)
    f = Frame.load(reg, "neg").pivot()
    with pytest.raises(ConversionError):
        f.log_return(col="price")


def test_trailing_avg_min_coverage(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    out = f.trailing_avg_with_min_coverage(3, 3, col="revenue")
    aaa = out.df[out.df["entity"] == "AAA"].sort_values("date")["revenue_tavg3"].to_numpy()
    assert np.isnan(aaa[0]) and np.isnan(aaa[1])
    assert aaa[2] == pytest.approx((100 + 110 + 121) / 3)


# --------------------------------------------------------------- 3.4 Rolling
def test_rolling_mean_and_ewma(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    rm = f.rolling_mean(2, col="price", min_periods=1)
    assert "price_roll_mean2" in rm.df.columns
    ew = f.ewma(2, col="price")
    assert "price_ewma2" in ew.df.columns and ew.df["price_ewma2"].notna().all()


# ----------------------------------------------- 3.5 Trend And Regression
def test_trend_slope_positive(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    sl = f.trend_slope(5, col="price")
    aaa = sl.df[sl.df["entity"] == "AAA"].sort_values("date")["price_slope5"].to_numpy()
    assert aaa[-1] == pytest.approx(1.0)  # AAA price rises by exactly 1 each period


def test_residualize_against_factors(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    out = f.residualize_against_factors(["price"], col="revenue")
    assert "revenue_resid" in out.df.columns


# ------------------------------------------------------ 3.6 Cross-Sectional
def test_z_score_per_date(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    z = f.z_score(col="revenue")
    # per date there are exactly 2 entities; z-scores are +/- equal magnitude
    g = z.df.groupby("date")["revenue_z"].sum()
    assert np.allclose(g.dropna().to_numpy(), 0.0, atol=1e-9)


def test_rank_and_winsorize(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    r = f.rank(col="revenue", ascending=False)
    assert r.df["revenue_rank"].max() == 2.0
    w = f.winsorize(0.1, col="revenue")
    assert "revenue_wins" in w.df.columns


def test_standardize_strategies(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    for strat in ("mean_std", "median_mad", "min_max"):
        out = f.standardize(strat, col="revenue")
        assert "revenue_std" in out.df.columns


# -------------------------------------------------------------- 3.7 Predicate
def test_predicates(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    filt = f.filter(lambda d: d["revenue"] > 150)
    assert (filt.df["revenue"] > 150).all()
    uni = f.where({"AAA"})
    assert set(uni.df["entity"].unique()) == {"AAA"}
    top = f.top_n(1, col="revenue")
    assert len(top) == 5  # one winner per date
    cmp = f.scalar_compare("ge", 200, col="revenue")
    assert cmp.df["revenue_ge"].dtype == bool


# ------------------------------------------------------------------ 3.8 Join
def test_factor_exposure_and_asof_join(registry):
    base = Frame.load(registry, "fundamentals").pivot()
    factors = base.with_column("mom", lambda d: d["price"] * 0.01)
    joined = base.factor_exposure_join(factors, ["mom"])
    assert "mom" in joined.df.columns and len(joined) == len(base)


def test_union_dedupes(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    u = f.union(f)
    assert len(u) == len(f)


# ----------------------------------------------------------- 3.9 Aggregation
def test_group_by_and_aggregate(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    g = f.group_by(["entity"], {"revenue": "mean"})
    assert len(g) == 2
    a = f.aggregate({"revenue": "sum"})
    assert len(a) == 1


def test_cross_sectional_aggregate_broadcast(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    out = f.cross_sectional_aggregate({"revenue": "mean"}, broadcast=True)
    assert "revenue_mean" in out.df.columns and len(out) == len(f)


# --------------------------------------------------------------- 3.10 Column
def test_column_ops(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    wc = f.with_column("rev_k", lambda d: d["revenue"] / 1000)
    assert "rev_k" in wc.df.columns
    cl = f.clamp(0, 120, col="revenue")
    assert cl.df["revenue_clamp"].max() <= 120
    sg = f.with_column("delta", lambda d: d["revenue"] - 150).sign(col="delta")
    assert set(np.unique(sg.df["delta_sign"])).issubset({-1.0, 0.0, 1.0})
    with pytest.raises(ConversionError):
        f.cast("entity", "float64")  # "AAA" cannot become a float


# ------------------------------------------------------------ 3.11 Composite
def test_composite_score_and_subpipeline(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    scored = (
        f.z_score(col="revenue")
        .z_score(col="price")
        .score_combine({"revenue_z": 0.7, "price_z": 0.3}, out="alpha")
    )
    assert "alpha" in scored.df.columns

    def recipe(frame: Frame) -> Frame:
        return frame.filter(lambda d: d["revenue"] > 100)

    sub = f.call_sub_pipeline(recipe)
    assert (sub.df["revenue"] > 100).all()


def test_bool_combine(registry):
    f = Frame.load(registry, "fundamentals").pivot()
    out = (
        f.scalar_compare("gt", 100, col="revenue", out="big")
        .scalar_compare("gt", 11, col="price", out="pricey")
        .bool_combine(["big", "pricey"], op="and", out="both")
    )
    assert out.df["both"].dtype == bool
