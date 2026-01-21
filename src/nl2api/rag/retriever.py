"""
Hybrid RAG Retriever

Implementation using pgvector for vector similarity and PostgreSQL
full-text search for keyword matching.

Features:
- Hybrid search (vector + keyword)
- Optional Redis caching for query results
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from src.common.telemetry import get_tracer
from src.nl2api.rag.protocols import DocumentType, RetrievalResult

if TYPE_CHECKING:
    import asyncpg
    from src.common.cache import RedisCache

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class HybridRAGRetriever:
    """
    Hybrid retrieval combining vector similarity and keyword search.

    Uses PostgreSQL with pgvector extension for vector operations
    and built-in full-text search for keyword matching.

    Features:
    - Hybrid vector + keyword search
    - Optional Redis caching for query results
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        embedding_dimension: int = 1536,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        redis_cache: "RedisCache | None" = None,
        cache_ttl_seconds: int = 3600,
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
        """
        self._pool = pool
        self._embedding_dimension = embedding_dimension
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._embedder: Any = None
        self._redis_cache = redis_cache
        self._cache_ttl = cache_ttl_seconds

    def set_embedder(self, embedder: Any) -> None:
        """Set the embedder for generating query embeddings."""
        self._embedder = embedder

    def _make_cache_key(
        self,
        query: str,
        domain: str | None,
        document_types: list[DocumentType] | None,
        limit: int,
    ) -> str:
        """Generate cache key for a query."""
        key_parts = [
            hashlib.sha256(query.encode()).hexdigest()[:16],
            domain or "all",
            ",".join(sorted(dt.value for dt in document_types)) if document_types else "all",
            str(limit),
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

        Returns:
            List of RetrievalResult ordered by relevance
        """
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.domain", domain or "all")
            span.set_attribute("rag.limit", limit)
            span.set_attribute("rag.threshold", threshold)
            span.set_attribute("rag.use_cache", use_cache)
            if document_types:
                span.set_attribute("rag.document_types", [dt.value for dt in document_types])

            return await self._retrieve_impl(query, domain, document_types, limit, threshold, use_cache, span)

    async def _retrieve_impl(
        self,
        query: str,
        domain: str | None,
        document_types: list[DocumentType] | None,
        limit: int,
        threshold: float,
        use_cache: bool,
        span: Any,
    ) -> list[RetrievalResult]:
        """Internal implementation of retrieve with span for tracing."""
        if self._embedder is None:
            raise RuntimeError("Embedder not set. Call set_embedder() first.")

        # Check Redis cache first
        cache_key = self._make_cache_key(query, domain, document_types, limit)
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

        # Cache results in Redis
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

            type_filter = ""
            if document_types:
                types = ", ".join(f"'{dt.value}'" for dt in document_types)
                type_filter = f"AND document_type IN ({types})"

            domain_filter = ""
            if domain:
                domain_filter = f"AND domain = '{domain}'"

            sql = f"""
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
            {type_filter}
            {domain_filter}
            ORDER BY score DESC
            LIMIT $2
            """

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, query, limit)

            results = []
            for row in rows:
                results.append(RetrievalResult(
                    id=row["id"],
                    content=row["content"],
                    document_type=DocumentType(row["document_type"]),
                    score=float(row["score"]),
                    domain=row["domain"],
                    field_code=row["field_code"],
                    example_query=row["example_query"],
                    example_api_call=row["example_api_call"],
                    metadata=row["metadata"] or {},
                ))

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


class OpenAIEmbedder:
    """
    Embedder using OpenAI's text-embedding models.

    Features:
    - Concurrency control via semaphore
    - Automatic retry with exponential backoff for rate limits
    - Token tracking for rate limit awareness
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        max_concurrent: int = 5,
        requests_per_minute: int = 3000,
    ):
        """
        Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            max_concurrent: Maximum concurrent requests
            requests_per_minute: Rate limit (for logging/monitoring)
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._max_concurrent = max_concurrent
        self._requests_per_minute = requests_per_minute

        # Concurrency control
        import asyncio
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Dimension varies by model
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._rate_limit_hits = 0

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimensions.get(self._model, 1536)

    @property
    def stats(self) -> dict[str, int]:
        """Get embedder statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "rate_limit_hits": self._rate_limit_hits,
        }

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        async with self._semaphore:
            return await self._embed_with_retry(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Uses semaphore to limit concurrent requests and handles rate limits.
        """
        async with self._semaphore:
            return await self._embed_batch_with_retry(texts)

    async def _embed_with_retry(
        self,
        text: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> list[float]:
        """Embed with retry logic for rate limits."""
        import asyncio
        import random

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=text,
                )
                self._total_requests += 1
                self._total_tokens += response.usage.total_tokens
                return response.data[0].embedding
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    self._rate_limit_hits += 1
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"OpenAI rate limit hit, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_error or RuntimeError("Embed failed after retries")

    async def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> list[list[float]]:
        """Embed batch with retry logic for rate limits."""
        import asyncio
        import random

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                self._total_requests += 1
                self._total_tokens += response.usage.total_tokens
                return [item.embedding for item in response.data]
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    self._rate_limit_hits += 1
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"OpenAI rate limit hit, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_error or RuntimeError("Embed batch failed after retries")
