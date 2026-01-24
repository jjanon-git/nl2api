"""
Evaluation Pipeline Implementations

NOTE: This module previously contained SyntaxEvaluator, LogicEvaluator, and
WaterfallEvaluator. These have been superseded by the pack-based evaluation
framework. Use NL2APIPack from src.evaluation.packs instead.

The evaluation stages are now implemented directly in the pack:
- SyntaxStage: Validates JSON structure and schema
- LogicStage: AST-based comparison of tool calls
- ExecutionStage: API execution comparison (deferred)
- SemanticsStage: LLM-as-Judge semantic comparison

See src/evaluation/packs/nl2api.py for the current implementations.
"""

# This module is kept for potential future use but the main evaluators
# have been migrated to the pack-based system.
