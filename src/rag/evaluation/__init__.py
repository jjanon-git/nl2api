"""
RAG Evaluation Pack

Evaluates Retrieval-Augmented Generation systems with:
- RAG Triad: Retrieval, Context Relevance, Faithfulness, Answer Relevance
- Domain Gates: Citations, Source Policy, Policy Compliance, Rejection Calibration

Usage:
    from src.rag.evaluation import RAGPack
    from src.evalkit.core.evaluator import Evaluator

    pack = RAGPack()
    evaluator = Evaluator(pack=pack)
    scorecard = await evaluator.evaluate(test_case, system_output)
"""

from src.rag.evaluation.pack import (
    RAGPack,
    RAGPackConfig,
    RAGRetrievalPack,
    RAGRetrievalPackConfig,
)
from src.rag.evaluation.stages import (
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
    "RAGPackConfig",
    "RAGRetrievalPack",
    "RAGRetrievalPackConfig",
    "RetrievalStage",
    "ContextRelevanceStage",
    "FaithfulnessStage",
    "AnswerRelevanceStage",
    "CitationStage",
    "SourcePolicyStage",
    "PolicyComplianceStage",
    "RejectionCalibrationStage",
]
