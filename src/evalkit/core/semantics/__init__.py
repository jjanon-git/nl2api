"""Semantics evaluation module for LLM-as-Judge."""

from src.evalkit.core.semantics.evaluator import (
    ComparisonScores,
    SemanticsEvaluator,
)
from src.evalkit.core.semantics.prompts import (
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
