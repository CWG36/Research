# Section 3 — Operators

This is the operator reference for the point-in-time, factor-aware
equity-screening pipeline DSL. Each operator is a deterministic node in a screen
DAG. Operators never mutate their input; they return a new frame. Window and
rolling operators run **per entity in date order**; cross-sectional operators
run **per date across entities**. Operators *refuse* (raise a typed error) on
malformed input rather than silently coercing or inventing data.

> Recreated from the specification shown in
> https://x.com/doodlestein/status/2062360758864785740 and implemented in
> [`panel_screen_dsl/frame.py`](../panel_screen_dsl/frame.py).

## 3.1 Source

- **`load`** — reads rows from a declared source alias. It is the zero-input entry point from already-ingested HFDT evidence into the screen DAG, and refuses with source-family codes when rows, entitlement, universe, or coverage conditions are unmet.
- **`materialize`** — marks the upstream frame as a final materialized output. Use it when a spec needs to make an intermediate panel visible to the planner or sink layer without altering row semantics.
- **`cross_section`** — projects an upstream panel to a single cross-sectional date. It is the safe way to convert bitemporal long-form rows into a point-in-time candidate set.
- **`snapshot`** — projects an upstream panel to a date range. It preserves the panel shape while narrowing the period window used by downstream operators.

## 3.2 Reshape

- **`pivot`** — turns long-form rows into wide-form columns. It is used when downstream operators need one metric per column rather than rows keyed by `metric_name`.
- **`unpivot`** — melts wide-form columns back into long form. It is the escape hatch after a column-oriented calculation when the canonical long-form panel contract must be restored.
- **`resample`** — aligns time series to a regular calendar grid. It refuses on unknown calendars and avoids implicit calendar invention.
- **`reindex`** — projects rows onto an explicit calendar without aggregation. Use it when the desired dates are known and missing rows should remain observable.
- **`align_calendar`** — aligns one frame to a reference frame's calendar. It is a two-input operator for joining panels that must share a date skeleton before later arithmetic.

## 3.3 Windowed

- **`lag`** — shifts a series forward in time within each entity. It is the standard way to compare a current value with an older value without lookahead.
- **`lead`** — shifts a series backward in time and is intentionally gated. It refuses unless the unsafe lookahead acknowledgement path is explicit.
- **`diff`** — differences a numeric series with a lag. It is used for absolute changes while retaining the original metric's unit.
- **`pct_change`** — computes percent change of a numeric series with a lag. It is the default relative-change primitive for one-period and multi-period comparisons.
- **`log_return`** — computes natural-log return for positive numeric series. It refuses unsupported or invalid inputs rather than silently taking logs of non-positive values.
- **`yoy`** — computes year-over-year percent change. It packages the common annual comparison into a named operator so recipes do not hand-roll calendar offsets.
- **`qoq`** — computes quarter-over-quarter percent change. It is the quarterly analogue to `yoy` for financial-statement panels.
- **`trailing_avg_with_min_coverage`** — computes a trailing average only when enough rows are present. It emits coverage-sensitive findings instead of pretending sparse panels are dense.

## 3.4 Rolling

- **`rolling_mean`** — computes a rolling arithmetic mean over a window. It is the basic smoothing primitive for noisy time-series metrics.
- **`rolling_sum`** — computes a rolling sum over a window. It supports flow-style metrics where accumulation over recent periods is the meaningful signal.
- **`rolling_std`** — computes rolling sample standard deviation. It is the local volatility or dispersion primitive for numeric panels.
- **`rolling_min`** — computes rolling minimum over a window. It exposes local baseline or floor behavior.
- **`rolling_max`** — computes rolling maximum over a window. It exposes local peak or ceiling behavior.
- **`rolling_median`** — computes rolling median over a window. It is the robust alternative to `rolling_mean` when outliers should not dominate.
- **`rolling_percentile`** — computes rolling percentile over a window. It gives recipes an explicit quantile surface rather than forcing percentile logic into expressions.
- **`ewma`** — computes an exponentially weighted moving average. It smooths recent values with deterministic decay semantics.

## 3.5 Trend And Regression

- **`trend_slope`** — computes an OLS slope over a trailing window. It is the deterministic trend primitive for single-series panels.
- **`linreg_beta`** — computes OLS beta of one series on another over a trailing window. It is the two-input regression primitive for exposure-like relationships.
- **`rolling_beta`** — computes rolling beta of a dependent variable with rolling residualization. It is the factor-aware analogue for separating idiosyncratic movement from declared exposures.
- **`cagr`** — computes compound annual growth rate over a trailing window. It gives growth recipes a period-normalized rate instead of raw start/end differences.
- **`conditional_variance`** — computes variance over a conditional subset. It is used when dispersion should be measured only inside a declared regime.
- **`residualize_against_factors`** — removes factor-column effects from a dependent variable with residualization. It is the factor-aware part that strips known exposures out of a metric before scoring.

## 3.6 Cross-Sectional

- **`factor_neutral_z_score`** — computes a cross-sectional z-score after neutralizing against factor columns. It is the factor-aware ranking primitive for screens that must not merely rediscover known exposures.
- **`rank`** — computes cross-sectional rank within a period or group. It returns ordering information without changing the underlying metric.
- **`z_score`** — computes a cross-sectional z-score with optional winsorization. It standardizes a metric for comparable candidate scoring.
- **`winsorize`** — clips values to low and high percentile bounds per period. It bounds outlier influence before ranking, regression, or aggregation.
- **`quantile`** — emits the value at a percentile per group. It is the deterministic cut-point primitive for threshold tests.
- **`demean_by_group`** — subtracts the group mean per period. It is the group-relative location primitive that preserves residual scale.
- **`neutralize_by_group`** — residualizes values against group means per period. It is a stronger group-neutralization than a plain cross-sectional screen.
- **`standardize`** — performs robust standardisation by group. Supported strategies include mean/std, median/MAD, and min/max scaling.

## 3.7 Predicate

- **`regime_condition`** — evaluates a deterministic regime predicate into a boolean column. It lets recipes name market or data regimes without moving that judgment into prose.
- **`filter`** — filters rows by a predicate expression. It is the general row-selection primitive.
- **`where`** — restricts rows to a declared universe. It is the universe-boundary primitive and should be used instead of ad hoc ticker lists inside operators.
- **`top_n`** — keeps the top-N rows by a numeric column. It is a deterministic candidate-selection primitive for high-score screens.
- **`bottom_n`** — keeps the bottom-N rows by a numeric column. It is the corresponding low-score selection primitive.
- **`between`** — keeps rows whose value falls between two bounds. It replaces custom expression code for range tests.
- **`is_null`** — emits a boolean column for null cells. It makes missingness explicit for later filters or sufficiency diagnostics.
- **`is_not_null`** — emits a boolean column for non-null cells. It is the positive missingness predicate for completeness filters.
- **`scalar_compare`** — emits a boolean column from a scalar comparison. It covers threshold tests such as greater-than, less-than, and equality checks.

## 3.8 Join

- **`factor_exposure_join`** — joins point-in-time factor exposures onto a primary panel. It is the dedicated join for factor-aware screens.
- **`asof_join`** — performs an as-of join with directional tolerance. It is the bitemporal-safe primitive for attaching stale-but-valid rows.
- **`inner_join`** — performs an inner join on declared keys. It keeps only rows present on both inputs.
- **`left_join`** — performs a left join on declared keys. It preserves the primary panel while attaching optional columns from the second input.
- **`outer_join`** — performs an outer join on declared keys. It keeps rows from either input and makes missingness visible.
- **`concat`** — concatenates frames vertically or horizontally. It is used for explicit union-by-position or column appends where schema compatibility is known.
- **`union`** — computes a row set-union with deduplication. It is the deterministic way to merge overlapping candidate sets.

## 3.9 Aggregation

- **`group_by`** — performs grouped aggregation with declared aggregate functions. It preserves group keys and emits one row per group.
- **`aggregate`** — aggregates without group keys and emits one row. It is the whole-panel summary primitive.
- **`resample_aggregate`** — fuses resampling and grouped aggregation. It is the stable path for period-bucket summaries over time.
- **`cross_sectional_aggregate`** — computes per-period grouped aggregation with optional broadcast-back. It lets a recipe compare each row with contemporaneous group statistics.

## 3.10 Column

- **`with_column`** — adds or replaces a column using a typed expression. It is the general column-derivation primitive.
- **`select`** — keeps only named columns. It narrows wide panels before downstream operators or sinks.
- **`drop`** — removes named columns. It trims unused payload while preserving the rest of the frame.
- **`rename`** — renames columns. It makes downstream operator references explicit after joins, pivots, or recipe composition.
- **`cast`** — performs an explicit dtype cast on a column. It refuses unsupported conversions rather than relying on implicit coercion.
- **`coalesce`** — emits the first non-null value across a list of columns. It is the deterministic fallback primitive for alternate data sources.
- **`if_else`** — emits a typed ternary expression from a boolean condition. It avoids embedding branch logic outside the spec.
- **`clamp`** — clamps numeric values to inclusive lower and upper bounds. It bounds values while preserving numeric dtype.
- **`sign`** — maps numeric values to -1, 0, or 1. It is useful when only direction matters.
- **`abs`** — computes absolute value. It turns signed deviations into magnitude features.

## 3.11 Composite

- **`bool_combine`** — combines boolean intermediates with AND, OR, and NOT. It is the predicate-composition primitive for multi-condition screens.
- **`score_combine`** — computes a weighted sum of normalized score columns. It is deterministic score assembly, not recommendation logic.
- **`call_sub_pipeline`** — inlines a reusable recipe as a sub-pipeline. It lets recipes compose vetted sub-DAGs while keeping `recipe_hash` and dependency expansion deterministic.
