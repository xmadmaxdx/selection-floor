"""Package marker for statistical inference."""
from __future__ import annotations

from src.stats.inference import (
    McNemarResult,
    gap_for_correct_ranking,
    minimum_detectable_effect,
    paired_gap_se,
    paired_test,
    selection_floor_table,
    standard_error,
    validation_gap_se,
)

__all__ = [
    "McNemarResult",
    "gap_for_correct_ranking",
    "minimum_detectable_effect",
    "paired_gap_se",
    "paired_test",
    "selection_floor_table",
    "standard_error",
    "validation_gap_se",
]