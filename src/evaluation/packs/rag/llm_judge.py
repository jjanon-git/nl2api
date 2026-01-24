"""
Compatibility shim for src.evaluation.packs.rag.llm_judge -> src.rag.evaluation.llm_judge.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export from new location
from src.rag.evaluation.llm_judge import *  # noqa: F401, F403
from src.rag.evaluation.llm_judge import JudgeResult, LLMJudge  # noqa: F401
