"""
RAG Evaluation Stages

All stages for RAG evaluation organized by category:

RAG Triad (Core):
- RetrievalStage: IR metrics (recall@k, precision@k, MRR, NDCG)
- ContextRelevanceStage: Is retrieved context relevant to query?
- FaithfulnessStage: Is response grounded in context?
- AnswerRelevanceStage: Does response answer the question?

Domain Gates:
- CitationStage: Citation presence, validity, accuracy, coverage
- SourcePolicyStage: Quote-only vs summarize enforcement (GATE)
- PolicyComplianceStage: Content policy violations (GATE)
- RejectionCalibrationStage: False positive/negative detection
"""

from .answer_relevance import AnswerRelevanceStage
from .citation import CitationStage
from .context_relevance import ContextRelevanceStage
from .faithfulness import FaithfulnessStage
from .policy_compliance import PolicyComplianceStage
from .rejection_calibration import RejectionCalibrationStage
from .retrieval import RetrievalStage
from .source_policy import SourcePolicyStage

__all__ = [
    # RAG Triad
    "RetrievalStage",
    "ContextRelevanceStage",
    "FaithfulnessStage",
    "AnswerRelevanceStage",
    # Domain Gates
    "CitationStage",
    "SourcePolicyStage",
    "PolicyComplianceStage",
    "RejectionCalibrationStage",
]
