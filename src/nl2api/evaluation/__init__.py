"""
NL2API Evaluation

Evaluation pack and adapters for NL2API tool-calling LLM systems.

The NL2APIPack evaluates using a 4-stage waterfall pipeline:
1. Syntax: Validates JSON structure and schema (GATE)
2. Logic: AST-based comparison of tool calls
3. Execution: Compares API execution results (deferred)
4. Semantics: LLM-as-judge NL response comparison

Usage:
    from src.nl2api.evaluation import NL2APIPack
    from src.evalkit.core.evaluator import Evaluator

    pack = NL2APIPack()
    evaluator = Evaluator(pack=pack)
    scorecard = await evaluator.evaluate(test_case, system_output)
"""

from src.nl2api.evaluation.adapter import NL2APITargetAdapter
from src.nl2api.evaluation.pack import NL2APIPack

__all__ = [
    "NL2APIPack",
    "NL2APITargetAdapter",
]
