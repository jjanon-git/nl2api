"""
Hybrid RAG Retriever

Implementation using pgvector for vector similarity and PostgreSQL
full-text search for keyword matching.

Features:
- Hybrid search (vector + keyword)
- Two-stage retrieval with cross-encoder reranking
- Optional Redis caching for query results
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from src.evalkit.common.telemetry import get_tracer
from src.rag.retriever.protocols import DocumentType, RetrievalResult

if TYPE_CHECKING:
    import asyncpg

    from src.evalkit.common.cache import RedisCache
    from src.rag.retriever.hyde import HyDEExpander
    from src.rag.retriever.reranker import Reranker

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class HybridRAGRetriever:
    """
    Hybrid retrieval combining vector similarity and keyword search.

    Uses PostgreSQL with pgvector extension for vector operations
    and built-in full-text search for keyword matching.

    Features:
    - Hybrid vector + keyword search
    - Two-stage retrieval with optional cross-encoder reranking
    - Optional HyDE (Hypothetical Document Embeddings) query expansion
    - Optional Redis caching for query results
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_dimension: int = 1536,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        redis_cache: RedisCache | None = None,
        cache_ttl_seconds: int = 3600,
        reranker: Reranker | None = None,
        first_stage_limit: int = 50,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            pool: asyncpg connection pool
            embedding_dimension: Dimension of embedding vectors
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            keyword_weight: Weight for keyword matching (0.0 to 1.0)
            redis_cache: Optional Redis cache for query results
            cache_ttl_seconds: TTL for cached query results (default 1 hour)
            reranker: Optional reranker for two-stage retrieval
            first_stage_limit: Number of candidates for first stage (before reranking)
        """
        self._pool = pool
        self._embedding_dimension = embedding_dimension
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._embedder: Any = None
        self._redis_cache = redis_cache
        self._cache_ttl = cache_ttl_seconds
        self._reranker = reranker
        self._first_stage_limit = first_stage_limit
        self._hyde_expander: HyDEExpander | None = None

    def set_embedder(self, embedder: Any) -> None:
        """Set the embedder for generating query embeddings."""
        self._embedder = embedder

    def set_reranker(self, reranker: Reranker) -> None:
        """Set the reranker for two-stage retrieval."""
        self._reranker = reranker

    def set_hyde_expander(self, expander: HyDEExpander) -> None:
        """Set the HyDE expander for query expansion."""
        self._hyde_expander = expander

    def _make_cache_key(
        self,
        query: str,
        domain: str | None,
        document_types: list[DocumentType] | None,
        limit: int,
        ticker: str | None = None,
    ) -> str:
        """Generate cache key for a query."""
        key_parts = [
            hashlib.sha256(query.encode()).hexdigest()[:16],
            domain or "all",
            ",".join(sorted(dt.value for dt in document_types)) if document_types else "all",
            str(limit),
            ticker or "all",
        ]
        return f"rag:{':'.join(key_parts)}"

    async def retrieve(
        self,
        query: str,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
        limit: int = 10,
        threshold: float = 0.5,
        use_cache: bool = True,
        ticker: str | None = None,
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
            use_cache: Whether to use Redis cache (if configured)
            ticker: Optional ticker filter (e.g., "AAPL") for entity-specific retrieval

        Returns:
            List of RetrievalResult ordered by relevance
        """
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.domain", domain or "all")
            span.set_attribute("rag.limit", limit)
            span.set_attribute("rag.threshold", threshold)
            span.set_attribute("rag.use_cache", use_cache)
            span.set_attribute("rag.ticker", ticker or "all")
            if document_types:
                span.set_attribute("rag.document_types", [dt.value for dt in document_types])

            return await self._retrieve_impl(
                query, domain, document_types, limit, threshold, use_cache, span, ticker
            )

    async def _retrieve_impl(
        self,
        query: str,
        domain: str | None,
        document_types: list[DocumentType] | None,
        limit: int,
        threshold: float,
        use_cache: bool,
        span: Any,
        ticker: str | None = None,
    ) -> list[RetrievalResult]:
        """Internal implementation of retrieve with span for tracing."""
        if self._embedder is None:
            raise RuntimeError("Embedder not set. Call set_embedder() first.")

        # Use first_stage_limit if reranker is configured for two-stage retrieval
        first_stage_limit = self._first_stage_limit if self._reranker else limit

        # Check Redis cache first
        cache_key = self._make_cache_key(query, domain, document_types, limit, ticker)
        if use_cache and self._redis_cache:
            cached = await self._redis_cache.get(cache_key)
            if cached:
                logger.debug(f"RAG cache hit for query: {query[:50]}...")
                span.set_attribute("rag.cache_hit", True)
                span.set_attribute("rag.result_count", len(cached))
                return [
                    RetrievalResult(
                        id=r["id"],
                        content=r["content"],
                        document_type=DocumentType(r["document_type"]),
                        score=r["score"],
                        domain=r.get("domain"),
                        field_code=r.get("field_code"),
                        example_query=r.get("example_query"),
                        example_api_call=r.get("example_api_call"),
                        metadata=r.get("metadata", {}),
                    )
                    for r in cached
                ]

        span.set_attribute("rag.cache_hit", False)

        # Optionally expand query using HyDE before embedding
        text_to_embed = query
        if self._hyde_expander:
            hypothetical = await self._hyde_expander.expand(query)
            text_to_embed = hypothetical
            span.set_attribute("rag.hyde_enabled", True)
            span.set_attribute("rag.hyde_length", len(hypothetical))
        else:
            span.set_attribute("rag.hyde_enabled", False)

        # Generate embedding and convert to PostgreSQL vector format
        query_embedding_list = await self._embedder.embed(text_to_embed)
        # Convert to string format for pgvector: "[1.0, 2.0, 3.0]"
        query_embedding = "[" + ",".join(str(x) for x in query_embedding_list) + "]"

        # Build the hybrid search query with parameterized filters
        # Convert document_types to array of strings for SQL parameter
        type_values: list[str] | None = None
        if document_types:
            type_values = [dt.value for dt in document_types]

        sql = """
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
            AND ($7::text[] IS NULL OR document_type = ANY($7))
            AND ($8::text IS NULL OR domain = $8)
            AND ($9::text IS NULL OR metadata->>'ticker' = $9)
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
            AND ($7::text[] IS NULL OR document_type = ANY($7))
            AND ($8::text IS NULL OR domain = $8)
            AND ($9::text IS NULL OR metadata->>'ticker' = $9)
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
            # Use explicit 60s timeout for hybrid search query (default is 60s but
            # under concurrent load the pool acquire can delay, making total time > 60s)
            rows = await conn.fetch(
                sql,
                query_embedding,
                first_stage_limit,  # Use first_stage_limit for two-stage retrieval
                query,
                self._vector_weight,
                self._keyword_weight,
                threshold,
                type_values,  # $7: document type filter (array or NULL)
                domain,  # $8: domain filter (string or NULL)
                ticker,  # $9: ticker/entity filter (string or NULL)
                timeout=60.0,
            )

        results = []
        for row in rows:
            # Parse metadata - can be JSONB (dict), JSON string, or None
            raw_metadata = row["metadata"]
            if raw_metadata is None:
                metadata = {}
            elif isinstance(raw_metadata, str):
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = raw_metadata

            results.append(
                RetrievalResult(
                    id=row["id"],
                    content=row["content"],
                    document_type=DocumentType(row["document_type"]),
                    score=float(row["combined_score"]),
                    domain=row["domain"],
                    field_code=row["field_code"],
                    example_query=row["example_query"],
                    example_api_call=row["example_api_call"],
                    metadata=metadata,
                )
            )

        # Two-stage retrieval: rerank if reranker is configured
        if self._reranker and len(results) > limit:
            first_stage_count = len(results)
            span.set_attribute("rag.reranking_enabled", True)
            span.set_attribute("rag.first_stage_count", first_stage_count)
            results = await self._reranker.rerank(query, results, top_k=limit)
            logger.debug(f"Reranked {first_stage_count} candidates to {len(results)} results")
        else:
            span.set_attribute("rag.reranking_enabled", False)
            # Trim to limit if no reranker
            results = results[:limit]

        # Cache results in Redis (cache final reranked results)
        if use_cache and self._redis_cache and results:
            cache_data = [
                {
                    "id": r.id,
                    "content": r.content,
                    "document_type": r.document_type.value,
                    "score": r.score,
                    "domain": r.domain,
                    "field_code": r.field_code,
                    "example_query": r.example_query,
                    "example_api_call": r.example_api_call,
                    "metadata": r.metadata,
                }
                for r in results
            ]
            await self._redis_cache.set(cache_key, cache_data, ttl=self._cache_ttl)

        span.set_attribute("rag.result_count", len(results))
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
        Falls back to keyword-only search if embedder not available.
        """
        # Use keyword-only search if no embedder configured
        if self._embedder is None:
            return await self.retrieve_by_keyword(
                query=query,
                domain=domain,
                document_types=[DocumentType.FIELD_CODE],
                limit=limit,
            )
        return await self.retrieve(
            query=query,
            domain=domain,
            document_types=[DocumentType.FIELD_CODE],
            limit=limit,
            threshold=0.3,  # Lower threshold for field codes
        )

    async def retrieve_by_keyword(
        self,
        query: str,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents using keyword search only.

        Does not require embedder or embeddings. Useful when:
        - Embedder not configured
        - Documents don't have embeddings yet
        - Pure keyword search is desired

        Args:
            query: Natural language query
            domain: Optional domain filter
            document_types: Optional document type filter
            limit: Maximum results

        Returns:
            List of RetrievalResult ordered by keyword relevance
        """
        with tracer.start_as_current_span("rag.retrieve_by_keyword") as span:
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.domain", domain or "all")
            span.set_attribute("rag.limit", limit)
            if document_types:
                span.set_attribute("rag.document_types", [dt.value for dt in document_types])

            # Convert document_types to array of strings for SQL parameter
            type_values: list[str] | None = None
            if document_types:
                type_values = [dt.value for dt in document_types]

            sql = """
            SELECT
                id,
                content,
                document_type,
                domain,
                field_code,
                example_query,
                example_api_call,
                metadata,
                ts_rank(search_vector, plainto_tsquery('english', $1)) as score
            FROM rag_documents
            WHERE search_vector @@ plainto_tsquery('english', $1)
            AND ($3::text[] IS NULL OR document_type = ANY($3))
            AND ($4 IS NULL OR domain = $4)
            ORDER BY score DESC
            LIMIT $2
            """

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, query, limit, type_values, domain)

            results = []
            for row in rows:
                # Parse metadata - can be JSONB (dict), JSON string, or None
                raw_metadata = row["metadata"]
                if raw_metadata is None:
                    metadata = {}
                elif isinstance(raw_metadata, str):
                    try:
                        metadata = json.loads(raw_metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                else:
                    metadata = raw_metadata

                results.append(
                    RetrievalResult(
                        id=row["id"],
                        content=row["content"],
                        document_type=DocumentType(row["document_type"]),
                        score=float(row["score"]),
                        domain=row["domain"],
                        field_code=row["field_code"],
                        example_query=row["example_query"],
                        example_api_call=row["example_api_call"],
                        metadata=metadata,
                    )
                )

            span.set_attribute("rag.result_count", len(results))
            logger.debug(f"Keyword search found {len(results)} results for: {query[:50]}...")
            return results

    async def retrieve_examples(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve similar query examples.

        Used for few-shot prompting.
        Falls back to keyword-only search if embedder not available.
        """
        # Use keyword-only search if no embedder configured
        if self._embedder is None:
            return await self.retrieve_by_keyword(
                query=query,
                domain=domain,
                document_types=[DocumentType.QUERY_EXAMPLE],
                limit=limit,
            )
        return await self.retrieve(
            query=query,
            domain=domain,
            document_types=[DocumentType.QUERY_EXAMPLE],
            limit=limit,
            threshold=0.4,
        )

    def _parse_metadata(self, raw_metadata: Any) -> dict[str, Any]:
        """Parse metadata from database row (JSONB, JSON string, or None)."""
        if raw_metadata is None:
            return {}
        elif isinstance(raw_metadata, str):
            try:
                return json.loads(raw_metadata)
            except json.JSONDecodeError:
                return {}
        else:
            return raw_metadata

    async def retrieve_with_parents(
        self,
        query: str,
        limit: int = 10,
        child_limit: int = 30,
        threshold: float = 0.0,
        domain: str | None = None,
        use_cache: bool = True,
        ticker: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Small-to-big retrieval: search children, return parents.

        This strategy:
        1. Searches small child chunks (512 chars) for precise matching
        2. Groups children by their parent chunks
        3. Returns parent chunks (4000 chars) ranked by matching children
        4. Provides better context while maintaining precise search

        Args:
            query: Search query
            limit: Maximum parent chunks to return
            child_limit: Number of children to retrieve in first stage
            threshold: Minimum relevance score
            domain: Optional domain filter
            use_cache: Whether to use caching
            ticker: Optional ticker filter (e.g., "AAPL") for entity-specific retrieval

        Returns:
            List of parent RetrievalResult objects with full context
        """
        with tracer.start_as_current_span("rag.retrieve_with_parents") as span:
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.domain", domain or "all")
            span.set_attribute("rag.limit", limit)
            span.set_attribute("rag.child_limit", child_limit)
            span.set_attribute("rag.retrieval_type", "small_to_big")
            span.set_attribute("rag.ticker", ticker or "all")

            if not self._embedder:
                raise RuntimeError("Embedder not set. Call set_embedder() first.")

            # Generate cache key for this specific retrieval type
            cache_key = (
                f"rag_parents:{hashlib.md5(f'{query}{domain}{limit}{ticker}'.encode()).hexdigest()}"
            )

            # Check cache
            if use_cache and self._redis_cache:
                cached = await self._redis_cache.get(cache_key)
                if cached:
                    span.set_attribute("rag.cache_hit", True)
                    return [
                        RetrievalResult(
                            id=r["id"],
                            content=r["content"],
                            document_type=DocumentType(r["document_type"]),
                            score=r["score"],
                            domain=r.get("domain"),
                            field_code=r.get("field_code"),
                            example_query=r.get("example_query"),
                            example_api_call=r.get("example_api_call"),
                            metadata=r.get("metadata", {}),
                        )
                        for r in cached
                    ]

            span.set_attribute("rag.cache_hit", False)

            # Get query embedding
            query_embedding_list = await self._embedder.embed(query)
            query_embedding = "[" + ",".join(str(x) for x in query_embedding_list) + "]"

            async with self._pool.acquire() as conn:
                # Step 1: Search child chunks (chunk_level = 1)
                # Note: chunk_level and parent_id are columns, not metadata fields
                child_sql = f"""
                    SELECT
                        id,
                        parent_id,
                        content,
                        document_type,
                        domain,
                        field_code,
                        example_query,
                        example_api_call,
                        metadata,
                        (
                            {self._vector_weight} * (1 - (embedding <=> $1::vector))
                            + {self._keyword_weight} * COALESCE(
                                ts_rank(search_vector, plainto_tsquery('english', $2)),
                                0
                            )
                        ) as combined_score
                    FROM rag_documents
                    WHERE
                        embedding IS NOT NULL
                        AND chunk_level = 1
                        AND parent_id IS NOT NULL
                        AND ($4 IS NULL OR domain = $4)
                        AND ($5::text IS NULL OR metadata->>'ticker' = $5)
                    ORDER BY combined_score DESC
                    LIMIT $3
                """

                child_rows = await conn.fetch(
                    child_sql,
                    query_embedding,
                    query,
                    child_limit,
                    domain,
                    ticker,
                )

                span.set_attribute("rag.child_count", len(child_rows))

                if not child_rows:
                    span.set_attribute("rag.result_count", 0)
                    return []

                # Step 2: Group children by parent and calculate parent scores
                parent_scores: dict[str, float] = {}
                parent_child_count: dict[str, int] = {}

                for row in child_rows:
                    # parent_id is a column (UUID), not metadata
                    parent_id = str(row["parent_id"]) if row["parent_id"] else None

                    if parent_id and row["combined_score"] >= threshold:
                        child_score = float(row["combined_score"])
                        if parent_id not in parent_scores:
                            parent_scores[parent_id] = 0.0
                            parent_child_count[parent_id] = 0
                        parent_scores[parent_id] += child_score
                        parent_child_count[parent_id] += 1

                if not parent_scores:
                    # Fallback: return children if no parents found
                    logger.debug("No parent chunks found, returning child results")
                    results = [
                        RetrievalResult(
                            id=str(row["id"]),
                            content=row["content"],
                            document_type=DocumentType(row["document_type"]),
                            score=float(row["combined_score"]),
                            domain=row.get("domain"),
                            field_code=row.get("field_code"),
                            example_query=row.get("example_query"),
                            example_api_call=row.get("example_api_call"),
                            metadata=self._parse_metadata(row.get("metadata")),
                        )
                        for row in child_rows[:limit]
                        if float(row["combined_score"]) >= threshold
                    ]
                    span.set_attribute("rag.result_count", len(results))
                    span.set_attribute("rag.fallback_to_children", True)
                    return results

                # Normalize scores by child count
                for pid in parent_scores:
                    parent_scores[pid] = parent_scores[pid] / max(parent_child_count[pid], 1)

                # Sort parents by score and get top N
                sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)
                top_parent_ids = [p[0] for p in sorted_parents[:limit]]

                span.set_attribute("rag.unique_parents", len(parent_scores))

                # Step 3: Fetch parent chunk details by their UUID
                parent_sql = """
                    SELECT
                        id,
                        content,
                        document_type,
                        domain,
                        field_code,
                        example_query,
                        example_api_call,
                        metadata
                    FROM rag_documents
                    WHERE id = ANY($1::uuid[])
                """

                parent_rows = await conn.fetch(parent_sql, top_parent_ids)

                # Build result list maintaining score order
                parent_map = {str(row["id"]): row for row in parent_rows}

                results = []
                for parent_id, score in sorted_parents[:limit]:
                    row = parent_map.get(parent_id)
                    if row:
                        metadata = self._parse_metadata(row.get("metadata"))
                        metadata["matching_children"] = parent_child_count.get(parent_id, 0)
                        metadata["retrieval_type"] = "small_to_big"

                        results.append(
                            RetrievalResult(
                                id=str(row["id"]),
                                content=row["content"],
                                document_type=DocumentType(row["document_type"]),
                                score=score,
                                domain=row.get("domain"),
                                field_code=row.get("field_code"),
                                example_query=row.get("example_query"),
                                example_api_call=row.get("example_api_call"),
                                metadata=metadata,
                            )
                        )

            # Cache results
            if use_cache and self._redis_cache and results:
                cache_data = [
                    {
                        "id": r.id,
                        "content": r.content,
                        "document_type": r.document_type.value,
                        "score": r.score,
                        "domain": r.domain,
                        "field_code": r.field_code,
                        "example_query": r.example_query,
                        "example_api_call": r.example_api_call,
                        "metadata": r.metadata,
                    }
                    for r in results
                ]
                await self._redis_cache.set(cache_key, cache_data, ttl=self._cache_ttl)

            span.set_attribute("rag.result_count", len(results))
            logger.debug(
                f"Small-to-big retrieval found {len(results)} parents from {len(child_rows)} children"
            )
            return results


# Re-export embedders for backwards compatibility
# Prefer using src.rag.retriever.embedders directly
from src.rag.retriever.embedders import (
    Embedder,
    LocalEmbedder,
    OpenAIEmbedder,
    create_embedder,
)

# Re-export reranker for convenience
from src.rag.retriever.reranker import (
    CrossEncoderReranker,
    Reranker,
    create_reranker,
)

__all__ = [
    "HybridRAGRetriever",
    "Embedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "create_embedder",
    "Reranker",
    "CrossEncoderReranker",
    "create_reranker",
]
