"""The :class:`Frame` -- an immutable, chainable panel of point-in-time evidence.

This module is a working recreation of Section 3 ("Operators") of the
factor-aware equity-screening pipeline DSL. A :class:`Frame` wraps a long-form
panel ``DataFrame`` whose contract is:

* one ``entity`` column (e.g. a ticker),
* one ``date`` column (the as-of / observation date),
* one or more value columns.

Every operator returns a *new* ``Frame`` (operators never mutate in place), so a
screen is just a chain of method calls that forms a deterministic DAG. Window
and rolling operators run per entity in date order; cross-sectional operators
run per date across entities. Operators *refuse* (raise a :class:`DslError`
subclass) on malformed input rather than silently coercing or inventing data.

The operators are grouped exactly as in the reference document:

    3.1 Source        3.5 Trend & Regression   3.9  Aggregation
    3.2 Reshape       3.6 Cross-Sectional      3.10 Column
    3.3 Windowed      3.7 Predicate            3.11 Composite
    3.4 Rolling       3.8 Join
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .errors import (
    ConversionError,
    CoverageError,
    DslError,
    LookaheadError,
    SchemaError,
)
from .sources import SourceRegistry


class Frame:
    """An immutable wrapper around a long-form panel ``DataFrame``."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        entity: str = "entity",
        date: str = "date",
        value: str = "value",
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._entity = entity
        self._date = date
        self._value = value

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    @property
    def df(self) -> pd.DataFrame:
        """A defensive copy of the underlying rows."""
        return self._df.copy()

    @property
    def value_col(self) -> str:
        return self._value

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Frame(rows={len(self._df)}, cols={list(self._df.columns)}, "
            f"entity={self._entity!r}, date={self._date!r}, value={self._value!r})"
        )

    def _spawn(self, df: pd.DataFrame, *, value: str | None = None) -> "Frame":
        """Build a sibling frame that inherits this frame's column contract."""
        return Frame(
            df,
            entity=self._entity,
            date=self._date,
            value=value if value is not None else self._value,
        )

    def _require(self, *cols: str) -> None:
        missing = [c for c in cols if c not in self._df.columns]
        if missing:
            raise SchemaError(
                f"missing required column(s): {missing}", code="schema_missing_column"
            )

    def _resolve_col(self, col: str | None) -> str:
        col = col if col is not None else self._value
        self._require(col)
        return col

    # ================================================================== #
    # 3.1 Source
    # ================================================================== #
    @classmethod
    def load(
        cls,
        registry: SourceRegistry,
        alias: str,
        *,
        entity: str = "entity",
        date: str = "date",
        value: str = "value",
    ) -> "Frame":
        """Read rows from a declared source alias.

        The zero-input entry point from already-ingested evidence into the
        screen DAG. Refuses with source-family codes when rows, entitlement,
        universe, or coverage conditions are unmet.
        """
        df = registry.resolve(alias)
        frame = cls(df, entity=entity, date=date, value=value)
        frame._require(entity, date)
        return frame

    def materialize(self, name: str | None = None) -> "Frame":
        """Mark the upstream frame as a final materialized output.

        Makes an intermediate panel visible to the planner / sink layer without
        altering row semantics. The optional ``name`` is recorded on the frame.
        """
        out = self._spawn(self._df)
        out.materialized_as = name  # type: ignore[attr-defined]
        return out

    def cross_section(self, as_of: object) -> "Frame":
        """Project the panel to a single cross-sectional date.

        The safe way to convert bitemporal long-form rows into a point-in-time
        candidate set: keeps the most recent row per entity at or before
        ``as_of``.
        """
        self._require(self._entity, self._date)
        as_of = pd.Timestamp(as_of)
        df = self._df[self._df[self._date] <= as_of]
        df = (
            df.sort_values([self._entity, self._date])
            .groupby(self._entity, sort=False)
            .tail(1)
        )
        return self._spawn(df)

    def snapshot(self, start: object, end: object) -> "Frame":
        """Project the panel to a ``[start, end]`` date range.

        Preserves panel shape while narrowing the period window used by
        downstream operators.
        """
        self._require(self._date)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        df = self._df[(self._df[self._date] >= start) & (self._df[self._date] <= end)]
        return self._spawn(df)

    # ================================================================== #
    # 3.2 Reshape
    # ================================================================== #
    def pivot(self, metric_col: str = "metric_name", value_col: str | None = None) -> "Frame":
        """Turn long-form rows into wide-form columns.

        Used when downstream operators need one metric per column rather than
        rows keyed by ``metric_name``.
        """
        value_col = self._resolve_col(value_col)
        self._require(self._entity, self._date, metric_col)
        wide = (
            self._df.pivot_table(
                index=[self._entity, self._date],
                columns=metric_col,
                values=value_col,
                aggfunc="last",
            )
            .reset_index()
        )
        wide.columns.name = None
        return self._spawn(wide)

    def unpivot(
        self,
        metric_cols: Sequence[str],
        metric_name: str = "metric_name",
        value_name: str | None = None,
    ) -> "Frame":
        """Melt wide-form columns back into long form.

        The escape hatch after a column-oriented calculation when the canonical
        long-form panel contract must be restored.
        """
        value_name = value_name or self._value
        self._require(self._entity, self._date, *metric_cols)
        long = self._df.melt(
            id_vars=[self._entity, self._date],
            value_vars=list(metric_cols),
            var_name=metric_name,
            value_name=value_name,
        )
        return self._spawn(long, value=value_name)

    def resample(self, freq: str, how: str = "last") -> "Frame":
        """Align time series to a regular calendar grid.

        Buckets each entity's rows to ``freq`` (e.g. ``"ME"``, ``"QE"``) using
        the ``how`` aggregation. Refuses on unknown aggregations rather than
        inventing a calendar.
        """
        self._require(self._entity, self._date, self._value)
        if not hasattr(pd.core.groupby.SeriesGroupBy, how) and how not in {
            "last", "first", "mean", "sum", "min", "max", "median",
        }:
            raise DslError(f"unknown resample aggregation: {how!r}", code="resample_how")
        out = (
            self._df.set_index(self._date)
            .groupby(self._entity, group_keys=True)[self._value]
            .resample(freq)
            .agg(how)
            .reset_index()
        )
        return self._spawn(out)

    def reindex(self, calendar: Sequence[object]) -> "Frame":
        """Project rows onto an explicit calendar without aggregation.

        Missing rows remain observable (as NaN values) rather than being filled
        or dropped.
        """
        self._require(self._entity, self._date, self._value)
        cal = pd.DatetimeIndex(pd.to_datetime(list(calendar))).sort_values()
        pieces = []
        for ent, g in self._df.groupby(self._entity, sort=False):
            s = g.set_index(self._date)[self._value].reindex(cal)
            piece = pd.DataFrame(
                {self._entity: ent, self._date: cal, self._value: s.to_numpy()}
            )
            pieces.append(piece)
        out = pd.concat(pieces, ignore_index=True) if pieces else self._df.iloc[:0]
        return self._spawn(out)

    def align_calendar(self, reference: "Frame") -> "Frame":
        """Align this frame to a reference frame's calendar.

        A two-input operator for joining panels that must share a date skeleton
        before later arithmetic.
        """
        cal = sorted(reference._df[reference._date].unique())
        return self.reindex(cal)

    # ================================================================== #
    # 3.3 Windowed (per entity, in date order)
    # ================================================================== #
    def _sorted(self) -> pd.DataFrame:
        return self._df.sort_values([self._entity, self._date]).reset_index(drop=True)

    def _grouped_value(self, df: pd.DataFrame, col: str):
        return df.groupby(self._entity, sort=False)[col]

    def lag(self, periods: int = 1, col: str | None = None, out: str | None = None) -> "Frame":
        """Shift a series forward in time within each entity.

        The standard way to compare a current value with an older value without
        lookahead.
        """
        col = self._resolve_col(col)
        out = out or f"{col}_lag{periods}"
        df = self._sorted()
        df[out] = self._grouped_value(df, col).shift(periods)
        return self._spawn(df, value=out)

    def lead(
        self,
        periods: int = 1,
        col: str | None = None,
        out: str | None = None,
        *,
        acknowledge_lookahead: bool = False,
    ) -> "Frame":
        """Shift a series backward in time. Intentionally gated.

        Refuses unless ``acknowledge_lookahead=True`` is passed, because using a
        future value in a screen is unsafe lookahead.
        """
        if not acknowledge_lookahead:
            raise LookaheadError(
                "lead introduces lookahead; pass acknowledge_lookahead=True to proceed",
                code="lookahead_unacknowledged",
            )
        col = self._resolve_col(col)
        out = out or f"{col}_lead{periods}"
        df = self._sorted()
        df[out] = self._grouped_value(df, col).shift(-periods)
        return self._spawn(df, value=out)

    def diff(self, periods: int = 1, col: str | None = None, out: str | None = None) -> "Frame":
        """Difference a numeric series with a lag, retaining the metric's unit."""
        col = self._resolve_col(col)
        out = out or f"{col}_diff{periods}"
        df = self._sorted()
        df[out] = self._grouped_value(df, col).diff(periods)
        return self._spawn(df, value=out)

    def pct_change(self, periods: int = 1, col: str | None = None, out: str | None = None) -> "Frame":
        """Compute percent change with a lag -- the default relative-change primitive."""
        col = self._resolve_col(col)
        out = out or f"{col}_pct{periods}"
        df = self._sorted()
        df[out] = self._grouped_value(df, col).pct_change(periods, fill_method=None)
        return self._spawn(df, value=out)

    def log_return(self, periods: int = 1, col: str | None = None, out: str | None = None) -> "Frame":
        """Compute natural-log return for positive numeric series.

        Refuses (rather than silently producing NaN/inf) when the series
        contains non-positive values.
        """
        col = self._resolve_col(col)
        if (self._df[col].dropna() <= 0).any():
            raise ConversionError(
                f"log_return requires strictly positive values in {col!r}",
                code="log_return_nonpositive",
            )
        out = out or f"{col}_logret{periods}"
        df = self._sorted()
        g = self._grouped_value(df, col)
        df[out] = np.log(df[col]) - np.log(g.shift(periods))
        return self._spawn(df, value=out)

    def yoy(self, periods: int = 4, col: str | None = None, out: str | None = None) -> "Frame":
        """Year-over-year percent change.

        Packages the common annual comparison into a named operator. Defaults to
        ``periods=4`` for quarterly financial-statement panels.
        """
        col = self._resolve_col(col)
        out = out or f"{col}_yoy"
        return self.pct_change(periods, col=col, out=out)

    def qoq(self, periods: int = 1, col: str | None = None, out: str | None = None) -> "Frame":
        """Quarter-over-quarter percent change -- the quarterly analogue to ``yoy``."""
        col = self._resolve_col(col)
        out = out or f"{col}_qoq"
        return self.pct_change(periods, col=col, out=out)

    def trailing_avg_with_min_coverage(
        self, window: int, min_coverage: int, col: str | None = None, out: str | None = None
    ) -> "Frame":
        """Trailing average emitted only when enough rows are present.

        Coverage-sensitive: rows with fewer than ``min_coverage`` observations in
        the window stay NaN instead of pretending a sparse panel is dense.
        """
        col = self._resolve_col(col)
        out = out or f"{col}_tavg{window}"
        df = self._sorted()
        df[out] = self._grouped_value(df, col).transform(
            lambda s: s.rolling(window, min_periods=min_coverage).mean()
        )
        return self._spawn(df, value=out)

    # ================================================================== #
    # 3.4 Rolling (per entity)
    # ================================================================== #
    def _rolling(self, window: int, agg: str, col: str | None, out: str | None, min_periods: int | None):
        col = self._resolve_col(col)
        out = out or f"{col}_roll_{agg}{window}"
        df = self._sorted()
        mp = window if min_periods is None else min_periods
        df[out] = self._grouped_value(df, col).transform(
            lambda s: getattr(s.rolling(window, min_periods=mp), agg)()
        )
        return self._spawn(df, value=out)

    def rolling_mean(self, window: int, col: str | None = None, out: str | None = None, min_periods: int | None = None) -> "Frame":
        """Rolling arithmetic mean -- the basic smoothing primitive for noisy series."""
        return self._rolling(window, "mean", col, out, min_periods)

    def rolling_sum(self, window: int, col: str | None = None, out: str | None = None, min_periods: int | None = None) -> "Frame":
        """Rolling sum -- supports flow-style metrics accumulated over recent periods."""
        return self._rolling(window, "sum", col, out, min_periods)

    def rolling_std(self, window: int, col: str | None = None, out: str | None = None, min_periods: int | None = None) -> "Frame":
        """Rolling sample standard deviation -- the local volatility / dispersion primitive."""
        return self._rolling(window, "std", col, out, min_periods)

    def rolling_min(self, window: int, col: str | None = None, out: str | None = None, min_periods: int | None = None) -> "Frame":
        """Rolling minimum -- exposes local baseline / floor behavior."""
        return self._rolling(window, "min", col, out, min_periods)

    def rolling_max(self, window: int, col: str | None = None, out: str | None = None, min_periods: int | None = None) -> "Frame":
        """Rolling maximum -- exposes local peak / ceiling behavior."""
        return self._rolling(window, "max", col, out, min_periods)

    def rolling_median(self, window: int, col: str | None = None, out: str | None = None, min_periods: int | None = None) -> "Frame":
        """Rolling median -- the robust alternative to ``rolling_mean``."""
        return self._rolling(window, "median", col, out, min_periods)

    def rolling_percentile(
        self, window: int, q: float, col: str | None = None, out: str | None = None, min_periods: int | None = None
    ) -> "Frame":
        """Rolling percentile -- an explicit quantile surface over the window."""
        col = self._resolve_col(col)
        out = out or f"{col}_roll_p{int(q * 100)}_{window}"
        df = self._sorted()
        mp = window if min_periods is None else min_periods
        df[out] = self._grouped_value(df, col).transform(
            lambda s: s.rolling(window, min_periods=mp).quantile(q)
        )
        return self._spawn(df, value=out)

    def ewma(self, span: int, col: str | None = None, out: str | None = None) -> "Frame":
        """Exponentially weighted moving average with deterministic decay."""
        col = self._resolve_col(col)
        out = out or f"{col}_ewma{span}"
        df = self._sorted()
        df[out] = self._grouped_value(df, col).transform(
            lambda s: s.ewm(span=span, adjust=False).mean()
        )
        return self._spawn(df, value=out)

    # ================================================================== #
    # 3.5 Trend And Regression
    # ================================================================== #
    @staticmethod
    def _ols_slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 2 or np.isnan(y).any():
            return np.nan
        x = np.arange(n, dtype=float)
        x = x - x.mean()
        denom = (x * x).sum()
        if denom == 0:
            return np.nan
        return float((x * (y - y.mean())).sum() / denom)

    @staticmethod
    def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 2:
            return np.nan
        y, x = y[mask], x[mask]
        xc = x - x.mean()
        denom = (xc * xc).sum()
        if denom == 0:
            return np.nan
        return float((xc * (y - y.mean())).sum() / denom)

    def trend_slope(self, window: int, col: str | None = None, out: str | None = None) -> "Frame":
        """OLS slope over a trailing window -- the deterministic trend primitive."""
        col = self._resolve_col(col)
        out = out or f"{col}_slope{window}"
        df = self._sorted()
        df[out] = self._grouped_value(df, col).transform(
            lambda s: s.rolling(window, min_periods=window).apply(self._ols_slope, raw=True)
        )
        return self._spawn(df, value=out)

    def linreg_beta(self, x_col: str, window: int, col: str | None = None, out: str | None = None) -> "Frame":
        """OLS beta of one series on another over a trailing window."""
        col = self._resolve_col(col)
        self._require(x_col)
        out = out or f"{col}_beta_{x_col}{window}"
        df = self._sorted()
        betas = np.full(len(df), np.nan)
        pos = 0
        for _, g in df.groupby(self._entity, sort=False):
            yv, xv = g[col].to_numpy(float), g[x_col].to_numpy(float)
            for i in range(len(g)):
                if i + 1 >= window:
                    betas[pos + i] = self._ols_beta(
                        yv[i + 1 - window : i + 1], xv[i + 1 - window : i + 1]
                    )
            pos += len(g)
        df[out] = betas
        return self._spawn(df, value=out)

    def rolling_beta(self, x_col: str, window: int, col: str | None = None, out: str | None = None) -> "Frame":
        """Rolling beta of a dependent variable -- alias of :meth:`linreg_beta`.

        Separates the part of a series explained by a declared exposure from its
        idiosyncratic movement, computed in a rolling window.
        """
        return self.linreg_beta(x_col, window, col=col, out=out)

    def cagr(self, window: int, periods_per_year: float = 4.0, col: str | None = None, out: str | None = None) -> "Frame":
        """Compound annual growth rate over a trailing window.

        Gives growth recipes a period-normalized rate instead of a raw
        start/end difference.
        """
        col = self._resolve_col(col)
        out = out or f"{col}_cagr{window}"
        df = self._sorted()

        def _cagr(s: np.ndarray) -> float:
            start, end = s[0], s[-1]
            if np.isnan(start) or np.isnan(end) or start <= 0 or end <= 0:
                return np.nan
            years = (len(s) - 1) / periods_per_year
            if years <= 0:
                return np.nan
            return float((end / start) ** (1.0 / years) - 1.0)

        df[out] = self._grouped_value(df, col).transform(
            lambda s: s.rolling(window, min_periods=window).apply(_cagr, raw=True)
        )
        return self._spawn(df, value=out)

    def conditional_variance(
        self, condition: Callable[[pd.DataFrame], pd.Series], col: str | None = None, out: str | None = None
    ) -> "Frame":
        """Variance over a conditional subset, broadcast back per entity.

        Dispersion is measured only inside the declared regime; rows outside the
        regime do not contribute.
        """
        col = self._resolve_col(col)
        out = out or f"{col}_condvar"
        df = self._sorted()
        mask = condition(df).to_numpy(bool)

        def _var(g: pd.DataFrame) -> pd.Series:
            m = mask[g.index.to_numpy()]
            v = g[col][m].var()
            return pd.Series(v, index=g.index)

        df[out] = (
            df.groupby(self._entity, sort=False, group_keys=False)
            .apply(_var, include_groups=False)
            .to_numpy()
        )
        return self._spawn(df, value=out)

    def residualize_against_factors(
        self, factor_cols: Sequence[str], col: str | None = None, out: str | None = None
    ) -> "Frame":
        """Remove factor-column effects from a dependent variable per entity.

        Regresses the value on the declared factor columns within each entity
        and returns the residual -- the factor-aware part that strips out known
        exposures.
        """
        col = self._resolve_col(col)
        self._require(*factor_cols)
        out = out or f"{col}_resid"
        df = self._sorted()

        def _resid(g: pd.DataFrame) -> pd.Series:
            y = g[col].to_numpy(float)
            X = g[list(factor_cols)].to_numpy(float)
            mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            res = np.full(len(g), np.nan)
            if mask.sum() > len(factor_cols):
                Xm = np.column_stack([np.ones(mask.sum()), X[mask]])
                coef, *_ = np.linalg.lstsq(Xm, y[mask], rcond=None)
                res[mask] = y[mask] - Xm @ coef
            return pd.Series(res, index=g.index)

        df[out] = (
            df.groupby(self._entity, sort=False, group_keys=False)
            .apply(_resid, include_groups=False)
            .to_numpy()
        )
        return self._spawn(df, value=out)

    # ================================================================== #
    # 3.6 Cross-Sectional (per date, across entities)
    # ================================================================== #
    def _grouped_date(self, df: pd.DataFrame):
        return df.groupby(self._date, sort=False, group_keys=False)

    def factor_neutral_z_score(
        self, factor_cols: Sequence[str], col: str | None = None, out: str | None = None
    ) -> "Frame":
        """Cross-sectional z-score after neutralizing against factor columns.

        The factor-aware ranking primitive for screens that must not merely
        rediscover known exposures.
        """
        col = self._resolve_col(col)
        self._require(*factor_cols)
        out = out or f"{col}_fnz"
        df = self._df.copy()

        def _fnz(g: pd.DataFrame) -> pd.Series:
            y = g[col].to_numpy(float)
            X = g[list(factor_cols)].to_numpy(float)
            mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            res = np.full(len(g), np.nan)
            if mask.sum() > len(factor_cols):
                Xm = np.column_stack([np.ones(mask.sum()), X[mask]])
                coef, *_ = np.linalg.lstsq(Xm, y[mask], rcond=None)
                r = y[mask] - Xm @ coef
                sd = r.std(ddof=1)
                res[mask] = (r - r.mean()) / sd if sd and not np.isnan(sd) else np.nan
            return pd.Series(res, index=g.index)

        df[out] = self._grouped_date(df).apply(_fnz, include_groups=False).to_numpy()
        return self._spawn(df, value=out)

    def rank(self, col: str | None = None, out: str | None = None, ascending: bool = True, pct: bool = False, group: str | None = None) -> "Frame":
        """Cross-sectional rank within a period (optionally within a group)."""
        col = self._resolve_col(col)
        out = out or f"{col}_rank"
        df = self._df.copy()
        keys = [self._date] + ([group] if group else [])
        if group:
            self._require(group)
        df[out] = df.groupby(keys, sort=False)[col].rank(ascending=ascending, pct=pct)
        return self._spawn(df, value=out)

    def z_score(self, col: str | None = None, out: str | None = None, winsorize_pct: float | None = None) -> "Frame":
        """Cross-sectional z-score with optional winsorization."""
        col = self._resolve_col(col)
        out = out or f"{col}_z"
        base = self.winsorize(winsorize_pct, col=col, out=f"{col}__wz") if winsorize_pct else self
        src = base._df.copy()
        use = f"{col}__wz" if winsorize_pct else col

        def _z(g: pd.DataFrame) -> pd.Series:
            v = g[use]
            sd = v.std(ddof=1)
            return (v - v.mean()) / sd if sd and not np.isnan(sd) else v * np.nan

        src[out] = src.groupby(self._date, sort=False, group_keys=False)[use].transform(
            lambda v: (v - v.mean()) / v.std(ddof=1) if v.std(ddof=1) else v * np.nan
        )
        if winsorize_pct:
            src = src.drop(columns=[use])
        return self._spawn(src, value=out)

    def winsorize(self, pct: float, col: str | None = None, out: str | None = None) -> "Frame":
        """Clip values to low and high percentile bounds per period.

        Bounds outlier influence before ranking, regression, or aggregation.
        """
        col = self._resolve_col(col)
        out = out or f"{col}_wins"
        df = self._df.copy()

        def _w(v: pd.Series) -> pd.Series:
            lo, hi = v.quantile(pct), v.quantile(1 - pct)
            return v.clip(lo, hi)

        df[out] = df.groupby(self._date, sort=False, group_keys=False)[col].transform(_w)
        return self._spawn(df, value=out)

    def quantile(self, q: float, col: str | None = None, out: str | None = None, group: str | None = None) -> "Frame":
        """Emit the value at a percentile per group -- a deterministic cut point."""
        col = self._resolve_col(col)
        out = out or f"{col}_q{int(q * 100)}"
        df = self._df.copy()
        keys = [self._date] + ([group] if group else [])
        if group:
            self._require(group)
        df[out] = df.groupby(keys, sort=False)[col].transform(lambda v: v.quantile(q))
        return self._spawn(df, value=out)

    def demean_by_group(self, group: str, col: str | None = None, out: str | None = None) -> "Frame":
        """Subtract the group mean per period -- group-relative location, scale preserved."""
        col = self._resolve_col(col)
        self._require(group)
        out = out or f"{col}_demean_{group}"
        df = self._df.copy()
        df[out] = df.groupby([self._date, group], sort=False)[col].transform(
            lambda v: v - v.mean()
        )
        return self._spawn(df, value=out)

    def neutralize_by_group(self, group: str, col: str | None = None, out: str | None = None) -> "Frame":
        """Residualize values against group means per period.

        A stronger group-neutralization: emits a z-scored, group-demeaned value.
        """
        col = self._resolve_col(col)
        self._require(group)
        out = out or f"{col}_neutral_{group}"
        df = self._df.copy()

        def _n(v: pd.Series) -> pd.Series:
            r = v - v.mean()
            sd = r.std(ddof=1)
            return r / sd if sd and not np.isnan(sd) else r * np.nan

        df[out] = df.groupby([self._date, group], sort=False, group_keys=False)[col].transform(_n)
        return self._spawn(df, value=out)

    def standardize(self, strategy: str = "mean_std", col: str | None = None, out: str | None = None, group: str | None = None) -> "Frame":
        """Robust standardisation by group.

        Supported strategies: ``"mean_std"``, ``"median_mad"``, ``"min_max"``.
        """
        col = self._resolve_col(col)
        out = out or f"{col}_std"
        df = self._df.copy()
        keys = [self._date] + ([group] if group else [])
        if group:
            self._require(group)

        def _s(v: pd.Series) -> pd.Series:
            if strategy == "mean_std":
                sd = v.std(ddof=1)
                return (v - v.mean()) / sd if sd else v * np.nan
            if strategy == "median_mad":
                med = v.median()
                mad = (v - med).abs().median()
                return (v - med) / (1.4826 * mad) if mad else v * np.nan
            if strategy == "min_max":
                lo, hi = v.min(), v.max()
                return (v - lo) / (hi - lo) if hi > lo else v * np.nan
            raise DslError(f"unknown standardize strategy: {strategy!r}", code="standardize_strategy")

        df[out] = df.groupby(keys, sort=False, group_keys=False)[col].transform(_s)
        return self._spawn(df, value=out)

    # ================================================================== #
    # 3.7 Predicate
    # ================================================================== #
    def regime_condition(self, predicate: Callable[[pd.DataFrame], pd.Series], out: str = "regime") -> "Frame":
        """Evaluate a deterministic regime predicate into a boolean column.

        Lets recipes name a market or data regime without moving that judgment
        into prose.
        """
        df = self._df.copy()
        df[out] = predicate(df).astype(bool)
        return self._spawn(df, value=out)

    def filter(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> "Frame":
        """Filter rows by a predicate expression -- the general row-selection primitive."""
        df = self._df
        return self._spawn(df[predicate(df).to_numpy(bool)])

    def where(self, universe: Iterable[str]) -> "Frame":
        """Restrict rows to a declared universe -- the universe-boundary primitive."""
        self._require(self._entity)
        u = set(universe)
        return self._spawn(self._df[self._df[self._entity].isin(u)])

    def top_n(self, n: int, col: str | None = None, per_date: bool = True) -> "Frame":
        """Keep the top-N rows by a numeric column (per date by default)."""
        col = self._resolve_col(col)
        if per_date:
            df = self._df.sort_values(col, ascending=False).groupby(self._date, sort=False).head(n)
        else:
            df = self._df.nlargest(n, col)
        return self._spawn(df)

    def bottom_n(self, n: int, col: str | None = None, per_date: bool = True) -> "Frame":
        """Keep the bottom-N rows by a numeric column (per date by default)."""
        col = self._resolve_col(col)
        if per_date:
            df = self._df.sort_values(col, ascending=True).groupby(self._date, sort=False).head(n)
        else:
            df = self._df.nsmallest(n, col)
        return self._spawn(df)

    def between(self, low: float, high: float, col: str | None = None, inclusive: str = "both") -> "Frame":
        """Keep rows whose value falls between two bounds, without custom code."""
        col = self._resolve_col(col)
        mask = self._df[col].between(low, high, inclusive=inclusive)
        return self._spawn(self._df[mask])

    def is_null(self, col: str | None = None, out: str | None = None) -> "Frame":
        """Emit a boolean column for null cells -- makes missingness explicit."""
        col = self._resolve_col(col)
        out = out or f"{col}_isnull"
        df = self._df.copy()
        df[out] = df[col].isna()
        return self._spawn(df, value=out)

    def is_not_null(self, col: str | None = None, out: str | None = None) -> "Frame":
        """Emit a boolean column for non-null cells -- the completeness predicate."""
        col = self._resolve_col(col)
        out = out or f"{col}_notnull"
        df = self._df.copy()
        df[out] = df[col].notna()
        return self._spawn(df, value=out)

    def scalar_compare(self, op: str, value: float, col: str | None = None, out: str | None = None) -> "Frame":
        """Emit a boolean column from a scalar comparison (``gt/ge/lt/le/eq/ne``)."""
        col = self._resolve_col(col)
        out = out or f"{col}_{op}"
        ops = {
            "gt": lambda s: s > value, "ge": lambda s: s >= value,
            "lt": lambda s: s < value, "le": lambda s: s <= value,
            "eq": lambda s: s == value, "ne": lambda s: s != value,
        }
        if op not in ops:
            raise DslError(f"unknown comparison op: {op!r}", code="scalar_compare_op")
        df = self._df.copy()
        df[out] = ops[op](df[col])
        return self._spawn(df, value=out)

    # ================================================================== #
    # 3.8 Join
    # ================================================================== #
    def factor_exposure_join(self, factors: "Frame", factor_cols: Sequence[str]) -> "Frame":
        """Join point-in-time factor exposures onto a primary panel.

        The dedicated join for factor-aware screens: attaches the named factor
        columns keyed on (entity, date).
        """
        self._require(self._entity, self._date)
        factors._require(factors._entity, factors._date, *factor_cols)
        right = factors._df[[factors._entity, factors._date, *factor_cols]].rename(
            columns={factors._entity: self._entity, factors._date: self._date}
        )
        merged = self._df.merge(right, on=[self._entity, self._date], how="left")
        return self._spawn(merged)

    def asof_join(
        self, other: "Frame", on_cols: Sequence[str], direction: str = "backward", tolerance: pd.Timedelta | None = None
    ) -> "Frame":
        """As-of join with directional tolerance -- the bitemporal-safe primitive.

        Attaches the most recent (or next, for ``direction="forward"``) valid
        rows from ``other`` within ``tolerance``.
        """
        left = self._df.sort_values(self._date)
        right = other._df.sort_values(other._date)[[other._entity, other._date, *on_cols]].rename(
            columns={other._entity: self._entity, other._date: self._date}
        )
        merged = pd.merge_asof(
            left, right, on=self._date, by=self._entity, direction=direction, tolerance=tolerance
        )
        return self._spawn(merged)

    def _join(self, other: "Frame", on: Sequence[str], how: str) -> "Frame":
        self._require(*on)
        other._require(*on)
        right_cols = [c for c in other._df.columns if c not in on or c in on]
        merged = self._df.merge(other._df[right_cols], on=list(on), how=how, suffixes=("", "_r"))
        return self._spawn(merged)

    def inner_join(self, other: "Frame", on: Sequence[str] = ("entity", "date")) -> "Frame":
        """Inner join on declared keys -- keeps only rows present on both inputs."""
        return self._join(other, on, "inner")

    def left_join(self, other: "Frame", on: Sequence[str] = ("entity", "date")) -> "Frame":
        """Left join on declared keys -- preserves the primary panel."""
        return self._join(other, on, "left")

    def outer_join(self, other: "Frame", on: Sequence[str] = ("entity", "date")) -> "Frame":
        """Outer join on declared keys -- keeps rows from either input."""
        return self._join(other, on, "outer")

    def concat(self, other: "Frame", axis: int = 0) -> "Frame":
        """Concatenate frames vertically (axis=0) or horizontally (axis=1)."""
        merged = pd.concat([self._df, other._df], axis=axis, ignore_index=(axis == 0))
        return self._spawn(merged)

    def union(self, other: "Frame") -> "Frame":
        """Row set-union with deduplication -- merges overlapping candidate sets."""
        merged = pd.concat([self._df, other._df], ignore_index=True).drop_duplicates()
        return self._spawn(merged)

    # ================================================================== #
    # 3.9 Aggregation
    # ================================================================== #
    def group_by(self, keys: Sequence[str], aggs: dict[str, str]) -> "Frame":
        """Grouped aggregation with declared aggregate functions (one row per group)."""
        self._require(*keys)
        out = self._df.groupby(list(keys), sort=False).agg(aggs).reset_index()
        return self._spawn(out)

    def aggregate(self, aggs: dict[str, str]) -> "Frame":
        """Whole-panel aggregation without group keys -- emits a single row."""
        row = {c: self._df[c].agg(fn) for c, fn in aggs.items()}
        return self._spawn(pd.DataFrame([row]))

    def resample_aggregate(self, freq: str, aggs: dict[str, str]) -> "Frame":
        """Fuse resampling and grouped aggregation -- stable period-bucket summaries."""
        self._require(self._entity, self._date)
        out = (
            self._df.set_index(self._date)
            .groupby(self._entity, group_keys=True)
            .resample(freq)
            .agg(aggs)
            .reset_index()
        )
        return self._spawn(out)

    def cross_sectional_aggregate(
        self, aggs: dict[str, str], group: str | None = None, broadcast: bool = False
    ) -> "Frame":
        """Per-period grouped aggregation, with optional broadcast-back.

        With ``broadcast=True`` each row gains its contemporaneous group
        statistic instead of collapsing to one row per group.
        """
        keys = [self._date] + ([group] if group else [])
        if group:
            self._require(group)
        if not broadcast:
            return self._spawn(self._df.groupby(keys, sort=False).agg(aggs).reset_index())
        df = self._df.copy()
        for c, fn in aggs.items():
            df[f"{c}_{fn}"] = df.groupby(keys, sort=False)[c].transform(fn)
        return self._spawn(df)

    # ================================================================== #
    # 3.10 Column
    # ================================================================== #
    def with_column(self, name: str, expr: Callable[[pd.DataFrame], pd.Series]) -> "Frame":
        """Add or replace a column using a typed expression -- general derivation."""
        df = self._df.copy()
        df[name] = expr(df)
        return self._spawn(df, value=name)

    def select(self, cols: Sequence[str]) -> "Frame":
        """Keep only named columns -- narrows wide panels before sinks."""
        self._require(*cols)
        return self._spawn(self._df[list(cols)])

    def drop(self, cols: Sequence[str]) -> "Frame":
        """Remove named columns -- trims unused payload."""
        return self._spawn(self._df.drop(columns=[c for c in cols if c in self._df.columns]))

    def rename(self, mapping: dict[str, str]) -> "Frame":
        """Rename columns -- makes downstream references explicit after joins/pivots."""
        new_value = mapping.get(self._value, self._value)
        return self._spawn(self._df.rename(columns=mapping), value=new_value)

    def cast(self, col: str, dtype: str) -> "Frame":
        """Explicit dtype cast -- refuses unsupported conversions rather than coercing."""
        self._require(col)
        df = self._df.copy()
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError) as exc:
            raise ConversionError(
                f"cannot cast {col!r} to {dtype!r}: {exc}", code="cast_unsupported"
            ) from exc
        return self._spawn(df)

    def coalesce(self, cols: Sequence[str], out: str = "coalesced") -> "Frame":
        """Emit the first non-null value across a list of columns."""
        self._require(*cols)
        df = self._df.copy()
        result = df[cols[0]]
        for c in cols[1:]:
            result = result.combine_first(df[c])
        df[out] = result
        return self._spawn(df, value=out)

    def if_else(
        self,
        cond: Callable[[pd.DataFrame], pd.Series],
        if_true: Callable[[pd.DataFrame], pd.Series],
        if_false: Callable[[pd.DataFrame], pd.Series],
        out: str = "ifelse",
    ) -> "Frame":
        """Typed ternary expression from a boolean condition -- no branch logic in prose."""
        df = self._df.copy()
        df[out] = np.where(cond(df).to_numpy(bool), if_true(df), if_false(df))
        return self._spawn(df, value=out)

    def clamp(self, low: float, high: float, col: str | None = None, out: str | None = None) -> "Frame":
        """Clamp numeric values to inclusive bounds, preserving numeric dtype."""
        col = self._resolve_col(col)
        out = out or f"{col}_clamp"
        df = self._df.copy()
        df[out] = df[col].clip(low, high)
        return self._spawn(df, value=out)

    def sign(self, col: str | None = None, out: str | None = None) -> "Frame":
        """Map numeric values to -1, 0, or 1 -- useful when only direction matters."""
        col = self._resolve_col(col)
        out = out or f"{col}_sign"
        df = self._df.copy()
        df[out] = np.sign(df[col])
        return self._spawn(df, value=out)

    def abs(self, col: str | None = None, out: str | None = None) -> "Frame":
        """Absolute value -- turns signed deviations into magnitude features."""
        col = self._resolve_col(col)
        out = out or f"{col}_abs"
        df = self._df.copy()
        df[out] = df[col].abs()
        return self._spawn(df, value=out)

    # ================================================================== #
    # 3.11 Composite
    # ================================================================== #
    def bool_combine(self, cols: Sequence[str], op: str = "and", out: str = "combined") -> "Frame":
        """Combine boolean intermediates with AND, OR, or NOT."""
        self._require(*cols)
        df = self._df.copy()
        if op == "and":
            res = df[list(cols)].all(axis=1)
        elif op == "or":
            res = df[list(cols)].any(axis=1)
        elif op == "not":
            if len(cols) != 1:
                raise DslError("NOT takes exactly one column", code="bool_combine_arity")
            res = ~df[cols[0]].astype(bool)
        else:
            raise DslError(f"unknown bool op: {op!r}", code="bool_combine_op")
        df[out] = res
        return self._spawn(df, value=out)

    def score_combine(self, weights: dict[str, float], out: str = "score") -> "Frame":
        """Weighted sum of normalized score columns -- deterministic score assembly."""
        self._require(*weights.keys())
        df = self._df.copy()
        total = np.zeros(len(df))
        for c, w in weights.items():
            total = total + df[c].fillna(0).to_numpy() * w
        df[out] = total
        return self._spawn(df, value=out)

    def call_sub_pipeline(self, recipe: Callable[["Frame"], "Frame"]) -> "Frame":
        """Inline a reusable recipe as a sub-pipeline.

        Lets recipes compose vetted sub-DAGs while keeping composition
        deterministic. ``recipe`` is any callable ``Frame -> Frame``.
        """
        result = recipe(self)
        if not isinstance(result, Frame):
            raise DslError("sub-pipeline must return a Frame", code="sub_pipeline_type")
        return result
