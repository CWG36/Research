"""Source registry for the panel-screen DSL.

In the original system, ``load`` reads from already-ingested HFDT (hedge-fund
data tool) evidence. Here the registry is a simple, deterministic mapping from
a *source alias* to a panel ``DataFrame`` plus its declared entitlement and
universe metadata. ``load`` consults the registry and refuses with a
source-family code when preconditions are unmet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .errors import SourceError


@dataclass
class Source:
    """A declared, ingestible source of panel evidence.

    Attributes
    ----------
    alias:
        Stable name referenced by ``load``.
    frame:
        The long-form panel rows for this source.
    entitled:
        Whether the caller is entitled to this source.
    universe:
        The set of entity ids this source is allowed to surface. ``None`` means
        unrestricted.
    """

    alias: str
    frame: pd.DataFrame
    entitled: bool = True
    universe: set[str] | None = None
    coverage_min_rows: int = 1


@dataclass
class SourceRegistry:
    """A deterministic collection of declared sources."""

    _sources: dict[str, Source] = field(default_factory=dict)

    def declare(
        self,
        alias: str,
        frame: pd.DataFrame,
        *,
        entitled: bool = True,
        universe: set[str] | None = None,
        coverage_min_rows: int = 1,
    ) -> "SourceRegistry":
        """Register a source alias. Returns ``self`` for chaining."""
        self._sources[alias] = Source(
            alias=alias,
            frame=frame.copy(),
            entitled=entitled,
            universe=universe,
            coverage_min_rows=coverage_min_rows,
        )
        return self

    def resolve(self, alias: str) -> pd.DataFrame:
        """Resolve a source alias to its rows, refusing on unmet conditions."""
        src = self._sources.get(alias)
        if src is None:
            raise SourceError(f"unknown source alias: {alias!r}", code="source_unknown")
        if not src.entitled:
            raise SourceError(f"not entitled to source: {alias!r}", code="source_entitlement")
        if src.frame.empty:
            raise SourceError(f"source has no rows: {alias!r}", code="source_no_rows")
        if len(src.frame) < src.coverage_min_rows:
            raise SourceError(
                f"source coverage below minimum ({len(src.frame)} < {src.coverage_min_rows}): {alias!r}",
                code="source_coverage",
            )
        return src.frame.copy()
