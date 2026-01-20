"""
Hybrid RAG Retriever

Implementation using pgvector for vector similarity and PostgreSQL
full-text search for keyword matching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.nl2api.rag.protocols import DocumentType, RetrievalResult

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


class HybridRAGRetriever:
    """
    Hybrid retrieval combining vector similarity and keyword search.

    Uses PostgreSQL with pgvector extension for vector operations
    and built-in full-text search for keyword matching.
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        embedding_dimension: int = 1536,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            pool: asyncpg connection pool
            embedding_dimension: Dimension of embedding vectors
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            keyword_weight: Weight for keyword matching (0.0 to 1.0)
        """
        self._pool = pool
        self._embedding_dimension = embedding_dimension
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._embedder: Any = None

    def set_embedder(self, embedder: Any) -> None:
        """Set the embedder for generating query embeddings."""
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant documents using hybrid search.

        Combines vector similarity and keyword matching with configurable weights.

        Args:
            query: Natural language query
            domain: Optional domain filter
            document_types: Optional document type filter
            limit: Maximum results
            threshold: Minimum relevance score

        Returns:
            List of RetrievalResult ordered by relevance
        """
        if self._embedder is None:
            raise RuntimeError("Embedder not set. Call set_embedder() first.")

        # Generate query embedding
        query_embedding = await self._embedder.embed(query)

        # Build the hybrid search query
        type_filter = ""
        if document_types:
            types = ", ".join(f"'{dt.value}'" for dt in document_types)
            type_filter = f"AND document_type IN ({types})"

        domain_filter = ""
        if domain:
            domain_filter = f"AND domain = '{domain}'"

        sql = f"""
        WITH vector_search AS (
            SELECT
                id,
                content,
                document_type,
                domain,
                field_code,
                example_query,
                example_api_call,
                metadata,
                1 - (embedding <=> $1::vector) as vector_score
            FROM rag_documents
            WHERE embedding IS NOT NULL
            {type_filter}
            {domain_filter}
            ORDER BY embedding <=> $1::vector
            LIMIT $2 * 2
        ),
        keyword_search AS (
            SELECT
                id,
                content,
                document_type,
                domain,
                field_code,
                example_query,
                example_api_call,
                metadata,
                ts_rank(search_vector, plainto_tsquery('english', $3)) as keyword_score
            FROM rag_documents
            WHERE search_vector @@ plainto_tsquery('english', $3)
            {type_filter}
            {domain_filter}
            ORDER BY keyword_score DESC
            LIMIT $2 * 2
        )
        SELECT
            COALESCE(v.id, k.id) as id,
            COALESCE(v.content, k.content) as content,
            COALESCE(v.document_type, k.document_type) as document_type,
            COALESCE(v.domain, k.domain) as domain,
            COALESCE(v.field_code, k.field_code) as field_code,
            COALESCE(v.example_query, k.example_query) as example_query,
            COALESCE(v.example_api_call, k.example_api_call) as example_api_call,
            COALESCE(v.metadata, k.metadata) as metadata,
            (
                COALESCE(v.vector_score, 0) * $4 +
                COALESCE(k.keyword_score, 0) * $5
            ) as combined_score
        FROM vector_search v
        FULL OUTER JOIN keyword_search k ON v.id = k.id
        WHERE (
            COALESCE(v.vector_score, 0) * $4 +
            COALESCE(k.keyword_score, 0) * $5
        ) >= $6
        ORDER BY combined_score DESC
        LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                sql,
                query_embedding,
                limit,
                query,
                self._vector_weight,
                self._keyword_weight,
                threshold,
            )

        results = []
        for row in rows:
            results.append(RetrievalResult(
                id=row["id"],
                content=row["content"],
                document_type=DocumentType(row["document_type"]),
                score=float(row["combined_score"]),
                domain=row["domain"],
                field_code=row["field_code"],
                example_query=row["example_query"],
                example_api_call=row["example_api_call"],
                metadata=row["metadata"] or {},
            ))

        return results

    async def retrieve_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant field codes for a domain.

        Specialized retrieval focused on field codes.
        """
        return await self.retrieve(
            query=query,
            domain=domain,
            document_types=[DocumentType.FIELD_CODE],
            limit=limit,
            threshold=0.3,  # Lower threshold for field codes
        )

    async def retrieve_examples(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve similar query examples.

        Used for few-shot prompting.
        """
        return await self.retrieve(
            query=query,
            domain=domain,
            document_types=[DocumentType.QUERY_EXAMPLE],
            limit=limit,
            threshold=0.4,
        )


class OpenAIEmbedder:
    """
    Embedder using OpenAI's text-embedding-ada-002 model.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key)

        # Dimension varies by model
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimensions.get(self._model, 1536)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]
