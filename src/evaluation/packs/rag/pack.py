"""
Compatibility shim for src.evaluation.packs.rag.pack -> src.rag.evaluation.pack.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export from new location
from src.rag.evaluation.pack import *  # noqa: F401, F403
from src.rag.evaluation.pack import RAGPack, RAGPackConfig  # noqa: F401
