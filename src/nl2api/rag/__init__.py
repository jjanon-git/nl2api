"""
RAG (Retrieval-Augmented Generation) Layer

Provides hybrid retrieval (vector + keyword) for field codes,
query examples, and economic indicators.
"""

from src.nl2api.rag.protocols import (
    RAGRetriever,
    RetrievalResult,
    DocumentType,
)
from src.nl2api.rag.retriever import HybridRAGRetriever

__all__ = [
    "RAGRetriever",
    "RetrievalResult",
    "DocumentType",
    "HybridRAGRetriever",
]
