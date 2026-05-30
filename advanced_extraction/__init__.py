"""Advanced adversarial model extraction utilities."""

from .metrics import fidelity_score, kl_divergence, normalize_probabilities
from .strategies import (
    QueryStrategy,
    get_strategy,
    list_strategies,
    select_queries,
)

__all__ = [
    "ExtractionConfig",
    "ExtractionResult",
    "QueryStrategy",
    "fidelity_score",
    "get_strategy",
    "kl_divergence",
    "list_strategies",
    "normalize_probabilities",
    "run_extraction_simulation",
    "select_queries",
]


def __getattr__(name):
    if name in {"ExtractionConfig", "ExtractionResult", "run_extraction_simulation"}:
        from .pipeline import ExtractionConfig, ExtractionResult, run_extraction_simulation

        values = {
            "ExtractionConfig": ExtractionConfig,
            "ExtractionResult": ExtractionResult,
            "run_extraction_simulation": run_extraction_simulation,
        }
        return values[name]
    raise AttributeError(f"module 'advanced_extraction' has no attribute {name!r}")
