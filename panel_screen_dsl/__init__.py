"""panel_screen_dsl -- a point-in-time, factor-aware equity-screening pipeline DSL.

A working recreation of the operator reference shown in
https://x.com/doodlestein/status/2062360758864785740 : a collection of
composable, deterministic operators for building long/short equity screens over
panel data, organized into the sections Source, Reshape, Windowed, Rolling,
Trend & Regression, Cross-Sectional, Predicate, Join, Aggregation, Column, and
Composite.

Example
-------
>>> from panel_screen_dsl import Frame, SourceRegistry
>>> reg = SourceRegistry().declare("fundamentals", df)
>>> screen = (
...     Frame.load(reg, "fundamentals")
...     .pivot()
...     .yoy(col="revenue")
...     .z_score(col="revenue_yoy")
...     .top_n(10, col="revenue_yoy_z")
...     .materialize("long_candidates")
... )
"""

from .errors import (
    ConversionError,
    CoverageError,
    DslError,
    LookaheadError,
    SchemaError,
    SourceError,
)
from .frame import Frame
from .sources import Source, SourceRegistry

__all__ = [
    "Frame",
    "Source",
    "SourceRegistry",
    "DslError",
    "SourceError",
    "SchemaError",
    "LookaheadError",
    "CoverageError",
    "ConversionError",
]

__version__ = "0.1.0"
