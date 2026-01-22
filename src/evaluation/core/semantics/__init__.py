"""Semantics evaluation module for LLM-as-Judge."""

from src.evaluation.core.semantics.evaluator import (
    ComparisonScores,
    SemanticsEvaluator,
)
from src.evaluation.core.semantics.prompts import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_USER_PROMPT_TEMPLATE,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT_TEMPLATE,
)

__all__ = [
    "ComparisonScores",
    "SemanticsEvaluator",
    "COMPARISON_SYSTEM_PROMPT",
    "COMPARISON_USER_PROMPT_TEMPLATE",
    "GENERATION_SYSTEM_PROMPT",
    "GENERATION_USER_PROMPT_TEMPLATE",
]
