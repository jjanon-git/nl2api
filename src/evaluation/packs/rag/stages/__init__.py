"""
Compatibility shim for src.evaluation.packs.rag.stages -> src.rag.evaluation.stages.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export from new location
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
    "RetrievalStage",
    "ContextRelevanceStage",
    "FaithfulnessStage",
    "AnswerRelevanceStage",
    "CitationStage",
    "SourcePolicyStage",
    "PolicyComplianceStage",
    "RejectionCalibrationStage",
]
