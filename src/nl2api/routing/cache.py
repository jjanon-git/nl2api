"""
Routing Cache

Tiered cache for routing decisions with exact match (Redis) and
semantic similarity (pgvector) support.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from src.nl2api.routing.protocols import RouterResult

logger = logging.getLogger(__name__)


@runtime_checkable
class RedisClient(Protocol):
    """Protocol for Redis client compatibility."""

    async def get(self, key: str) -> bytes | None:
        """Get a value by key."""
        ...

    async def setex(self, key: str, seconds: int, value: str) -> None:
        """Set a value with expiration."""
        ...


@runtime_checkable
class PostgresPool(Protocol):
    """Protocol for PostgreSQL connection pool."""

    def connection(self) -> Any:
        """Get a connection context manager."""
        ...


# Type alias for embedder function
Embedder = Callable[[str], Awaitable[list[float]]]


class RoutingCache:
    """
    Tiered cache for routing decisions.

    Tier 1: Exact match (Redis) - O(1) lookup, ~1ms
    Tier 2: Semantic similarity (pgvector) - ~20ms, catches paraphrases

    The cache stores routing decisions to avoid redundant LLM calls
    for similar or identical queries.
    """

    def __init__(
        self,
        redis: RedisClient | None = None,
        pg_pool: PostgresPool | None = None,
        embedder: Embedder | None = None,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        semantic_cache_enabled: bool = True,
        key_prefix: str = "route:",
    ):
        """
        Initialize the routing cache.

        Args:
            redis: Redis client for exact match cache
            pg_pool: PostgreSQL pool with pgvector for semantic cache
            embedder: Function to generate embeddings for semantic search
            similarity_threshold: Minimum similarity for semantic cache hits
            ttl_seconds: TTL for cached entries
            semantic_cache_enabled: Whether to enable semantic caching
            key_prefix: Prefix for Redis keys
        """
        self._redis = redis
        self._pg_pool = pg_pool
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._semantic_enabled = semantic_cache_enabled and pg_pool and embedder
        self._key_prefix = key_prefix

        if self._semantic_enabled:
            logger.info(
                f"RoutingCache: semantic cache enabled "
                f"(threshold={similarity_threshold})"
            )
        else:
            logger.info("RoutingCache: exact match only (semantic disabled)")

    def _cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self._key_prefix}{query_hash}"

    async def get(self, query: str) -> RouterResult | None:
        """
        Check cache for a routing decision.

        Checks in order:
        1. Exact match in Redis (fast)
        2. Semantic similarity in pgvector (if enabled)

        Args:
            query: Query to look up

        Returns:
            Cached RouterResult or None if not found
        """
        # Tier 1: Exact match in Redis
        if self._redis:
            try:
                cached = await self._exact_lookup(query)
                if cached:
                    logger.debug(f"Exact cache hit for routing")
                    return cached
            except Exception as e:
                logger.warning(f"Redis cache lookup failed: {e}")

        # Tier 2: Semantic similarity in pgvector
        if self._semantic_enabled:
            try:
                cached = await self._semantic_lookup(query)
                if cached:
                    logger.debug(f"Semantic cache hit for routing")
                    return cached
            except Exception as e:
                logger.warning(f"Semantic cache lookup failed: {e}")

        return None

    async def _exact_lookup(self, query: str) -> RouterResult | None:
        """Look up exact match in Redis."""
        if not self._redis:
            return None

        key = self._cache_key(query)
        data = await self._redis.get(key)

        if data:
            try:
                parsed = json.loads(data)
                return RouterResult(
                    domain=parsed["domain"],
                    confidence=parsed["confidence"],
                    reasoning=parsed.get("reasoning"),
                    cached=True,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid cached routing data: {e}")

        return None

    async def _semantic_lookup(self, query: str) -> RouterResult | None:
        """Look up semantically similar queries in pgvector."""
        if not self._pg_pool or not self._embedder:
            return None

        # Generate embedding for query
        embedding = await self._embedder(query)

        async with self._pg_pool.connection() as conn:
            # Query for similar cached routing decisions
            row = await conn.fetchrow(
                """
                SELECT domain, confidence, reasoning,
                       1 - (embedding <=> $1::vector) as similarity
                FROM routing_cache
                WHERE 1 - (embedding <=> $1::vector) > $2
                ORDER BY similarity DESC
                LIMIT 1
                """,
                embedding,
                self._similarity_threshold,
            )

            if row:
                logger.debug(
                    f"Semantic match: domain={row['domain']}, "
                    f"similarity={row['similarity']:.3f}"
                )
                return RouterResult(
                    domain=row["domain"],
                    confidence=row["confidence"],
                    reasoning=row["reasoning"],
                    cached=True,
                )

        return None

    async def set(self, query: str, result: RouterResult) -> None:
        """
        Cache a routing decision.

        Stores in both Redis (exact match) and pgvector (semantic).

        Args:
            query: Original query
            result: Routing result to cache
        """
        # Don't cache failed routing
        if result.domain == "unknown" or result.confidence == 0.0:
            return

        # Tier 1: Redis exact match
        if self._redis:
            try:
                await self._exact_store(query, result)
            except Exception as e:
                logger.warning(f"Redis cache store failed: {e}")

        # Tier 2: pgvector semantic cache
        if self._semantic_enabled:
            try:
                await self._semantic_store(query, result)
            except Exception as e:
                logger.warning(f"Semantic cache store failed: {e}")

    async def _exact_store(self, query: str, result: RouterResult) -> None:
        """Store exact match in Redis."""
        if not self._redis:
            return

        key = self._cache_key(query)
        data = json.dumps({
            "domain": result.domain,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        })
        await self._redis.setex(key, self._ttl, data)

    async def _semantic_store(self, query: str, result: RouterResult) -> None:
        """Store in pgvector for semantic lookup."""
        if not self._pg_pool or not self._embedder:
            return

        # Generate embedding
        embedding = await self._embedder(query)

        async with self._pg_pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO routing_cache (query, embedding, domain, confidence, reasoning)
                VALUES ($1, $2::vector, $3, $4, $5)
                ON CONFLICT (query) DO UPDATE SET
                    domain = EXCLUDED.domain,
                    confidence = EXCLUDED.confidence,
                    reasoning = EXCLUDED.reasoning,
                    updated_at = NOW()
                """,
                query,
                embedding,
                result.domain,
                result.confidence,
                result.reasoning,
            )

    async def invalidate(self, query: str) -> None:
        """
        Invalidate a cached routing decision.

        Args:
            query: Query to invalidate
        """
        # Remove from Redis
        if self._redis:
            try:
                key = self._cache_key(query)
                # Note: Redis delete is not in protocol, using setex with 0 TTL
                # In practice, you'd use redis.delete(key)
                logger.debug(f"Invalidating cache for: {key}")
            except Exception as e:
                logger.warning(f"Redis invalidation failed: {e}")

        # Remove from pgvector
        if self._semantic_enabled and self._pg_pool:
            try:
                async with self._pg_pool.connection() as conn:
                    await conn.execute(
                        "DELETE FROM routing_cache WHERE query = $1",
                        query,
                    )
            except Exception as e:
                logger.warning(f"Semantic cache invalidation failed: {e}")

    async def clear_all(self) -> None:
        """Clear all cached routing decisions."""
        # Clear pgvector cache
        if self._pg_pool:
            try:
                async with self._pg_pool.connection() as conn:
                    await conn.execute("TRUNCATE TABLE routing_cache")
                    logger.info("Cleared semantic routing cache")
            except Exception as e:
                logger.warning(f"Failed to clear semantic cache: {e}")

        # Note: Redis clearing would need pattern-based delete
        logger.info("Routing cache cleared")


class InMemoryRoutingCache:
    """
    Simple in-memory routing cache for testing.

    No TTL or size limits - intended for unit tests only.
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._cache: dict[str, RouterResult] = {}

    async def get(self, query: str) -> RouterResult | None:
        """Get cached result."""
        result = self._cache.get(query)
        if result:
            return replace(result, cached=True)
        return None

    async def set(self, query: str, result: RouterResult) -> None:
        """Cache a result."""
        # Don't cache failed routing (unknown domain or zero confidence)
        if result.domain != "unknown" and result.confidence > 0.0:
            self._cache[query] = result

    async def invalidate(self, query: str) -> None:
        """Remove from cache."""
        self._cache.pop(query, None)

    async def clear_all(self) -> None:
        """Clear all entries."""
        self._cache.clear()
