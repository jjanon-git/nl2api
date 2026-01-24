"""
RAG Retriever Protocols

Defines the interface for retrieval-augmented generation components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DocumentType(str, Enum):
    """Types of documents in the RAG index."""

    FIELD_CODE = "field_code"  # API field codes (e.g., TR.EPSMean)
    QUERY_EXAMPLE = "query_example"  # Example NL queries with API calls
    ECONOMIC_INDICATOR = "economic_indicator"  # Economic indicator codes
    SEC_FILING = "sec_filing"  # SEC 10-K/10-Q filing chunks


@dataclass(frozen=True)
class RetrievalResult:
    """
    A single retrieval result from the RAG system.

    Contains the retrieved document and relevance metadata.
    """

    id: str
    content: str
    document_type: DocumentType
    score: float  # Relevance score (0.0 to 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Domain-specific fields
    domain: str | None = None  # e.g., "estimates", "fundamentals"
    field_code: str | None = None  # e.g., "TR.EPSMean"
    example_query: str | None = None  # For query examples
    example_api_call: str | None = None  # For query examples


@runtime_checkable
class RAGRetriever(Protocol):
    """
    Protocol for RAG retrieval operations.

    Supports hybrid retrieval (vector similarity + keyword matching).
    """

    async def retrieve(
        self,
        query: str,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Natural language query
            domain: Optional domain filter (e.g., "estimates")
            document_types: Optional filter by document type
            limit: Maximum results to return
            threshold: Minimum relevance score (0.0 to 1.0)

        Returns:
            List of RetrievalResult ordered by relevance descending
        """
        ...

    async def retrieve_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant field codes for a domain query.

        Specialized method for finding API field codes.

        Args:
            query: Natural language query describing the data needed
            domain: API domain (e.g., "estimates", "fundamentals")
            limit: Maximum field codes to return

        Returns:
            List of RetrievalResult for field codes
        """
        ...

    async def retrieve_examples(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve similar query examples with their API calls.

        Used for few-shot prompting.

        Args:
            query: Natural language query
            domain: Optional domain filter
            limit: Maximum examples to return

        Returns:
            List of RetrievalResult for query examples
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for text embedding generation.

    Used by the RAG retriever for vector similarity search.
    """

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...
