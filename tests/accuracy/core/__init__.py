"""
Accuracy testing core infrastructure.

Provides AccuracyEvaluator and related utilities for measuring
NL2API output quality using real LLM calls.
"""

from tests.accuracy.core.config import AccuracyConfig, Tier
from tests.accuracy.core.evaluator import (
    AccuracyEvaluator,
    AccuracyResult,
    AccuracyReport,
)

__all__ = [
    "AccuracyConfig",
    "Tier",
    "AccuracyEvaluator",
    "AccuracyResult",
    "AccuracyReport",
]
