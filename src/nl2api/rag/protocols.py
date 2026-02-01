"""
RAG Retriever Protocols - Re-export from canonical location.

The canonical definitions live in src/rag/retriever/protocols.py.
This module re-exports them for backwards compatibility with code
that imports from src.nl2api.rag.protocols.
"""

from src.rag.retriever.protocols import (
    DocumentType,
    Embedder,
    RAGRetriever,
    RetrievalResult,
)

__all__ = ["DocumentType", "Embedder", "RAGRetriever", "RetrievalResult"]
