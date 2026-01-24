"""
RAG (Retrieval-Augmented Generation) Layer

Provides hybrid retrieval (vector + keyword) for field codes,
query examples, and economic indicators.
"""

from src.rag.retriever.indexer import (
    FieldCodeDocument,
    QueryExampleDocument,
    RAGIndexer,
    parse_estimates_reference,
    parse_query_examples,
)
from src.rag.retriever.protocols import (
    DocumentType,
    RAGRetriever,
    RetrievalResult,
)
from src.rag.retriever.retriever import HybridRAGRetriever, OpenAIEmbedder

__all__ = [
    "RAGRetriever",
    "RetrievalResult",
    "DocumentType",
    "HybridRAGRetriever",
    "OpenAIEmbedder",
    "RAGIndexer",
    "FieldCodeDocument",
    "QueryExampleDocument",
    "parse_estimates_reference",
    "parse_query_examples",
]
