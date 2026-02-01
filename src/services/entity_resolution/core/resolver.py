"""
Entity Resolver Implementation

Resolves entity names to financial identifiers using:
- Database lookups (entity_aliases table)
- OpenFIGI API fallback
- Multi-level caching (L1 in-memory, L2 Redis)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from src.evalkit.common.entity_resolution import (
    ResolvedEntity,
    normalize_entity,
    resolve_via_database,
    resolve_via_openfigi,
)
from src.evalkit.common.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryConfig,
    retry_with_backoff,
)

from .extractor import EntityExtractor

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Entity resolver using database and external APIs.

    Extracts company mentions from queries and resolves them to RICs
    using database lookups and external APIs (OpenFIGI).

    Features:
    - Circuit breaker: Fails fast when external service is unhealthy
    - Retry with backoff: Handles transient failures gracefully
    - Timeout: Prevents hanging on slow responses
    - Caching: In-memory and optional Redis caching
    - Database lookup: Entity aliases from GLEIF/SEC EDGAR
    - OpenFIGI fallback: External API for unknown entities
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool | None = None,
        redis_client: Any | None = None,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        use_cache: bool = True,
        timeout_seconds: float = 5.0,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
        retry_max_attempts: int = 3,
        redis_cache_ttl_seconds: int = 86400,
    ):
        self._db_pool = db_pool
        self._redis_client = redis_client
        self._api_endpoint = api_endpoint
        self._api_key = api_key
        self._use_cache = use_cache
        self._timeout_seconds = timeout_seconds
        self._redis_ttl = redis_cache_ttl_seconds

        # In-memory L1 cache
        self._cache: dict[str, ResolvedEntity] = {}

        # Entity extractor
        self._extractor = EntityExtractor()

        # Circuit breaker for external API
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_seconds,
            name="entity-resolution",
        )

        # Retry config
        self._retry_config = RetryConfig(
            max_attempts=retry_max_attempts,
            base_delay=0.5,
            max_delay=5.0,
            retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        )

    async def resolve(self, query: str) -> dict[str, str]:
        """
        Extract and resolve entities from a query.

        Args:
            query: Natural language query

        Returns:
            Dictionary mapping entity names to RICs
        """
        entities = self._extractor.extract(query)
        resolved: dict[str, str] = {}

        for entity in entities:
            result = await self.resolve_single(entity)
            if result:
                resolved[result.original] = result.identifier

        return resolved

    async def resolve_single(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> ResolvedEntity | None:
        """
        Resolve a single entity to its identifier.

        Uses multi-level caching:
        1. L1: In-memory cache
        2. L2: Redis cache (if configured)
        3. Database lookup
        4. External API (OpenFIGI)

        Args:
            entity: Entity name (e.g., "Apple Inc.")
            entity_type: Optional type hint

        Returns:
            ResolvedEntity if found
        """
        # Normalize entity name
        normalized = normalize_entity(entity)

        # Skip common words
        if normalized in self._extractor._ignore_words:
            return None

        # L1: Check in-memory cache
        if self._use_cache and normalized in self._cache:
            return self._cache[normalized]

        # L2: Check Redis cache
        if self._redis_client and self._use_cache:
            cached = await self._check_redis_cache(normalized)
            if cached:
                self._cache[normalized] = cached
                return cached

        # Database lookup (using shared module)
        if self._db_pool:
            db_result = await resolve_via_database(self._db_pool, entity, normalized)
            if db_result:
                await self._cache_result(normalized, db_result)
                return db_result

        # External API (OpenFIGI via shared module)
        api_result = await self._resolve_via_api(entity)
        if api_result:
            await self._cache_result(normalized, api_result)
            return api_result

        logger.debug(f"Could not resolve entity: {entity}")
        return None

    async def resolve_batch(self, entities: list[str]) -> list[ResolvedEntity]:
        """Resolve multiple entities in batch."""
        results = []
        for entity in entities:
            result = await self.resolve_single(entity)
            if result:
                results.append(result)
        return results

    async def _check_redis_cache(self, normalized: str) -> ResolvedEntity | None:
        """Check Redis L2 cache."""
        try:
            cache_key = f"entity:{normalized}"
            cached = await self._redis_client.get(cache_key)
            if cached:
                return ResolvedEntity.from_dict(cached)
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")
        return None

    async def _cache_result(self, normalized: str, result: ResolvedEntity) -> None:
        """Cache a resolved entity in L1 and L2 caches."""
        if not self._use_cache:
            return

        # L1: In-memory cache
        self._cache[normalized] = result

        # L2: Redis cache
        if self._redis_client:
            try:
                cache_key = f"entity:{normalized}"
                await self._redis_client.set(cache_key, result.to_dict(), ex=self._redis_ttl)
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")

    async def _resolve_via_api(self, entity: str) -> ResolvedEntity | None:
        """Resolve entity using OpenFIGI API with resilience patterns."""
        # Try OpenFIGI first (via shared module)
        try:
            result = await resolve_via_openfigi(
                query=entity,
                api_key=self._api_key,
                timeout=self._timeout_seconds,
            )
            if result:
                return result
        except Exception as e:
            logger.debug(f"OpenFIGI resolution failed: {e}")

        # Fall back to configured external API
        if not self._api_endpoint:
            return None

        async def _make_request() -> ResolvedEntity | None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"

                async with session.get(
                    f"{self._api_endpoint}/resolve",
                    params={"entity": entity},
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("found"):
                            return ResolvedEntity(
                                original=entity,
                                identifier=data["identifier"],
                                entity_type=data.get("type", "company"),
                                confidence=data.get("confidence", 0.8),
                                alternatives=tuple(data.get("alternatives", [])),
                                metadata=data.get("metadata", {}),
                            )
                    elif response.status >= 500:
                        raise ConnectionError(f"Server error: {response.status}")
            return None

        try:
            return await self._circuit_breaker.call(
                retry_with_backoff,
                _make_request,
                config=self._retry_config,
            )
        except CircuitOpenError:
            logger.warning(f"Circuit open, skipping API for: {entity}")
            return None
        except Exception as e:
            logger.warning(f"API resolution error: {e}")
            return None

    @property
    def circuit_breaker_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        stats = self._circuit_breaker.stats
        return {
            "state": stats.state.value,
            "failure_count": stats.failure_count,
            "total_calls": stats.total_calls,
            "total_failures": stats.total_failures,
            "total_successes": stats.total_successes,
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        self._circuit_breaker.reset()
