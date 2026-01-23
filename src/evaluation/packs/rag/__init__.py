"""
RAG Evaluation Pack

Evaluates Retrieval-Augmented Generation systems with:
- RAG Triad: Retrieval, Context Relevance, Faithfulness, Answer Relevance
- Domain Gates: Citations, Source Policy, Policy Compliance, Rejection Calibration

Usage:
    from src.evaluation.packs.rag import RAGPack
    from src.evaluation.core.evaluator import Evaluator

    pack = RAGPack()
    evaluator = Evaluator(pack=pack)
    scorecard = await evaluator.evaluate(test_case, system_output)
"""

from .pack import RAGPack
from .stages import (
    AnswerRelevanceStage,
    CitationStage,
    ContextRelevanceStage,
    FaithfulnessStage,
    PolicyComplianceStage,
    RejectionCalibrationStage,
    RetrievalStage,
    SourcePolicyStage,
)

__all__ = [
    "RAGPack",
    "RetrievalStage",
    "ContextRelevanceStage",
    "FaithfulnessStage",
    "AnswerRelevanceStage",
    "CitationStage",
    "SourcePolicyStage",
    "PolicyComplianceStage",
    "RejectionCalibrationStage",
]
