"""
RAG (Retrieval-Augmented Generation) Module

Consolidates all RAG-related code:
- retriever/: Vector search and retrieval
- ingestion/: SEC filing ingestion pipelines
- ui/: Streamlit-based RAG UI
"""

from src.rag.retriever import (
    DocumentType,
    FieldCodeDocument,
    HybridRAGRetriever,
    OpenAIEmbedder,
    QueryExampleDocument,
    RAGIndexer,
    RAGRetriever,
    RetrievalResult,
    parse_estimates_reference,
    parse_query_examples,
)

__all__ = [
    # Protocols
    "RAGRetriever",
    "RetrievalResult",
    "DocumentType",
    # Implementations
    "HybridRAGRetriever",
    "OpenAIEmbedder",
    # Indexer
    "RAGIndexer",
    "FieldCodeDocument",
    "QueryExampleDocument",
    "parse_estimates_reference",
    "parse_query_examples",
]
