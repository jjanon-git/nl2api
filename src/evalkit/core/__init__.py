"""Core evaluation components."""

from src.evalkit.core.ast_comparator import ASTComparator, ComparisonResult
from src.evalkit.core.evaluator import Evaluator, EvaluatorConfig
from src.evalkit.core.exporters import (
    CSVExporter,
    EvaluationSummary,
    JSONExporter,
    SummaryExporter,
)
from src.evalkit.core.semantics import ComparisonScores, SemanticsEvaluator

__all__ = [
    # Generic evaluator facade
    "Evaluator",
    "EvaluatorConfig",
    # Exporters
    "JSONExporter",
    "CSVExporter",
    "SummaryExporter",
    "EvaluationSummary",
    # Comparators
    "ASTComparator",
    "ComparisonResult",
    # Semantics
    "SemanticsEvaluator",
    "ComparisonScores",
]
