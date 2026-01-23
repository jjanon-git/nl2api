"""Core evaluation components."""

from src.evaluation.core.ast_comparator import ASTComparator, ComparisonResult
from src.evaluation.core.evaluator import Evaluator, EvaluatorConfig
from src.evaluation.core.evaluators import LogicEvaluator, SyntaxEvaluator, WaterfallEvaluator
from src.evaluation.core.exporters import (
    CSVExporter,
    EvaluationSummary,
    JSONExporter,
    SummaryExporter,
)
from src.evaluation.core.semantics import ComparisonScores, SemanticsEvaluator

__all__ = [
    # Generic evaluator facade
    "Evaluator",
    "EvaluatorConfig",
    # Exporters
    "JSONExporter",
    "CSVExporter",
    "SummaryExporter",
    "EvaluationSummary",
    # Legacy evaluators (NL2API-specific)
    "ASTComparator",
    "ComparisonResult",
    "SyntaxEvaluator",
    "LogicEvaluator",
    "WaterfallEvaluator",
    "SemanticsEvaluator",
    "ComparisonScores",
]
