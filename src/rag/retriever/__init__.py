"""
RAG (Retrieval-Augmented Generation) Layer

Provides hybrid retrieval (vector + keyword) for field codes,
query examples, and economic indicators.

Supports multiple backends:
- PostgreSQL + pgvector (HybridRAGRetriever)
- Azure AI Search (AzureAISearchRetriever)

Use create_retriever() factory to instantiate the appropriate backend.
"""

from src.rag.retriever.azure_search import AzureAISearchRetriever
from src.rag.retriever.factory import create_retriever
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
    # Protocols and models
    "RAGRetriever",
    "RetrievalResult",
    "DocumentType",
    # Retriever implementations
    "HybridRAGRetriever",
    "AzureAISearchRetriever",
    # Factory
    "create_retriever",
    # Embedder
    "OpenAIEmbedder",
    # Indexer
    "RAGIndexer",
    "FieldCodeDocument",
    "QueryExampleDocument",
    "parse_estimates_reference",
    "parse_query_examples",
]
