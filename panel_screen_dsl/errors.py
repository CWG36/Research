"""Error types for the panel-screen DSL.

The reference document emphasizes that operators *refuse* on malformed input
rather than silently coercing or inventing data. These exception types make
that refusal explicit and machine-readable (each carries a stable ``code``).
"""

from __future__ import annotations


class DslError(Exception):
    """Base class for all DSL refusals.

    Every refusal carries a short, stable ``code`` so that a planner or sink
    layer can react programmatically instead of parsing prose.
    """

    code = "dsl_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class SourceError(DslError):
    """Raised by ``load`` when a source alias is unknown or its preconditions
    (rows, entitlement, universe, coverage) are unmet."""

    code = "source_error"


class SchemaError(DslError):
    """Raised when a required column is missing or has the wrong shape."""

    code = "schema_error"


class LookaheadError(DslError):
    """Raised by forward-looking operators (``lead``) when the explicit unsafe
    lookahead acknowledgement is not provided."""

    code = "lookahead_error"


class CoverageError(DslError):
    """Raised when a coverage-sensitive operator does not have enough rows."""

    code = "coverage_error"


class ConversionError(DslError):
    """Raised by ``cast`` / ``log_return`` on unsupported or invalid conversions."""

    code = "conversion_error"
