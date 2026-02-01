"""
Entity Resolver Implementation

Implements entity resolution using external API with resilience patterns:
- Circuit breaker for failing fast when service is down
- Retry with exponential backoff for transient failures
- Configurable timeouts
- Optional Redis caching for resolved entities
"""

from __future__ import annotations

import logging
import re
import warnings
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
from src.evalkit.common.telemetry import get_tracer

if TYPE_CHECKING:
    import asyncpg

    from src.evalkit.common.cache import RedisCache

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ExternalEntityResolver:
    """
    Entity resolver using database and external APIs for company/RIC resolution.

    .. deprecated::
        Direct instantiation is deprecated. Use ``create_entity_resolver(config)``
        from ``src.nl2api.resolution`` instead, which returns the appropriate
        resolver based on configuration (HTTP service or local).

    Extracts company mentions from queries and resolves them to RICs
    using database lookups and external APIs (OpenFIGI).

    Features:
    - Circuit breaker: Fails fast when external service is unhealthy
    - Retry with backoff: Handles transient failures gracefully
    - Timeout: Prevents hanging on slow responses
    - Caching: In-memory and optional Redis caching for resolved entities
    - Database lookup: 2.9M entities from GLEIF/SEC EDGAR
    - OpenFIGI fallback: External API for entities not in database
    """

    def __init__(
        self,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        use_cache: bool = True,
        timeout_seconds: float = 5.0,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
        retry_max_attempts: int = 3,
        redis_cache: RedisCache | None = None,
        redis_cache_ttl_seconds: int = 86400,
        db_pool: asyncpg.Pool | None = None,
        *,
        _internal: bool = False,  # Set by factory to suppress warning
    ):
        """
        Initialize the entity resolver.

        .. deprecated::
            Use ``create_entity_resolver(config)`` instead of direct instantiation.

        Args:
            api_endpoint: External API endpoint for resolution
            api_key: API key for authentication
            use_cache: Whether to cache resolved entities
            timeout_seconds: Timeout for API calls
            circuit_failure_threshold: Failures before opening circuit
            circuit_recovery_seconds: Seconds before trying to recover
            retry_max_attempts: Maximum retry attempts for transient failures
            redis_cache: Optional Redis cache for distributed caching
            redis_cache_ttl_seconds: TTL for Redis cache entries (default 24 hours)
            db_pool: Optional asyncpg connection pool for database lookups
        """
        if not _internal:
            warnings.warn(
                "Direct instantiation of ExternalEntityResolver is deprecated. "
                "Use create_entity_resolver(config) from src.nl2api.resolution instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._api_endpoint = api_endpoint
        self._api_key = api_key
        self._use_cache = use_cache
        self._timeout_seconds = timeout_seconds
        self._cache: dict[str, ResolvedEntity] = {}  # In-memory L1 cache
        self._redis_cache = redis_cache  # Optional L2 distributed cache
        self._redis_ttl = redis_cache_ttl_seconds
        self._db_pool = db_pool  # Optional database pool for entity lookups

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_seconds,
            name="entity-resolution",
        )

        # Retry config for transient failures
        self._retry_config = RetryConfig(
            max_attempts=retry_max_attempts,
            base_delay=0.5,
            max_delay=5.0,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
            ),
        )

        # Common words that should not be treated as companies
        self._ignore_words = {
            "what",
            "how",
            "show",
            "get",
            "find",
            "list",
            "who",
            "when",
            "where",
            "why",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "with",
            "forecast",
            "estimate",
            "eps",
            "revenue",
            "price",
            "target",
            "rating",
            "of",
            "in",
            "to",
            "at",
            "by",
            "from",
            "on",
            "about",
            "above",
            "below",
            "best",
            "stock",
            "stocks",
            "market",
            "data",
            "show",
            "me",
            "tell",
            "need",
            "corp",
            "inc",
            "ltd",
            "limited",
            "company",
            "corporation",
        }

    async def resolve(
        self,
        query: str,
    ) -> dict[str, str]:
        """
        Extract and resolve entities from a query.

        Uses regex patterns to identify potential company names,
        then resolves them to RICs.

        Args:
            query: Natural language query

        Returns:
            Dictionary mapping entity names to RICs
        """
        with tracer.start_as_current_span("entity.resolve") as span:
            span.set_attribute("entity.query_length", len(query))

            entities = self._extract_entities(query)
            span.set_attribute("entity.extracted_count", len(entities))

            resolved: dict[str, str] = {}

            for entity in entities:
                result = await self.resolve_single(entity)
                if result:
                    resolved[result.original] = result.identifier

            span.set_attribute("entity.resolved_count", len(resolved))
            return resolved

    async def resolve_single(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> ResolvedEntity | None:
        """
        Resolve a single entity to its identifier.

        Uses multi-level caching:
        1. L1: In-memory cache (fastest)
        2. L2: Redis cache (distributed, if configured)
        3. Database lookup
        4. External API (OpenFIGI)

        Args:
            entity: Entity name (e.g., "Apple Inc.")
            entity_type: Optional type hint

        Returns:
            ResolvedEntity if found
        """
        with tracer.start_as_current_span("entity.resolve_single") as span:
            span.set_attribute("entity.name", entity[:50])  # Truncate for span
            return await self._resolve_single_impl(entity, entity_type, span)

    async def _resolve_single_impl(
        self,
        entity: str,
        entity_type: str | None,
        span: Any,
    ) -> ResolvedEntity | None:
        """Internal implementation of resolve_single."""
        # Normalize entity name using shared function
        normalized = normalize_entity(entity)

        # Skip common words
        if normalized in self._ignore_words:
            span.set_attribute("entity.source", "ignored")
            return None

        # L1: Check in-memory cache
        if self._use_cache and normalized in self._cache:
            span.set_attribute("entity.source", "l1_cache")
            return self._cache[normalized]

        # L2: Check Redis cache
        if self._redis_cache and self._use_cache:
            cache_key = f"entity:{normalized}"
            cached = await self._redis_cache.get(cache_key)
            if cached:
                result = ResolvedEntity(
                    original=cached["original"],
                    identifier=cached["identifier"],
                    entity_type=cached.get("entity_type", "company"),
                    confidence=cached.get("confidence", 0.8),
                    alternatives=tuple(cached.get("alternatives", [])),
                    metadata=cached.get("metadata", {}),
                )
                # Populate L1 cache
                self._cache[normalized] = result
                span.set_attribute("entity.source", "l2_cache")
                return result

        # Try database lookup (using shared module)
        if self._db_pool:
            db_result = await resolve_via_database(self._db_pool, entity, normalized)
            if db_result:
                await self._cache_result(normalized, db_result)
                span.set_attribute("entity.source", "database")
                return db_result

        # Try external API (OpenFIGI via shared module, then configured API)
        result = await self._resolve_via_api(entity)
        if result:
            await self._cache_result(normalized, result)
            span.set_attribute("entity.source", "external_api")
            return result

        # No resolution found
        span.set_attribute("entity.source", "not_found")
        logger.debug(f"Could not resolve entity: {entity}")
        return None

    async def _cache_result(self, normalized: str, result: ResolvedEntity) -> None:
        """Cache a resolved entity in L1 and L2 caches."""
        if not self._use_cache:
            return

        # L1: In-memory cache
        self._cache[normalized] = result

        # L2: Redis cache
        if self._redis_cache:
            cache_key = f"entity:{normalized}"
            cache_data = {
                "original": result.original,
                "identifier": result.identifier,
                "entity_type": result.entity_type,
                "confidence": result.confidence,
                "alternatives": list(result.alternatives) if result.alternatives else [],
                "metadata": dict(result.metadata) if result.metadata else {},
            }
            await self._redis_cache.set(cache_key, cache_data, ttl=self._redis_ttl)

    async def resolve_batch(
        self,
        entities: list[str],
    ) -> list[ResolvedEntity]:
        """
        Resolve multiple entities in batch.

        Args:
            entities: List of entity names

        Returns:
            List of resolved entities (may be shorter than input)
        """
        results = []
        for entity in entities:
            result = await self.resolve_single(entity)
            if result:
                results.append(result)
        return results

    def _extract_entities(self, query: str) -> list[str]:
        """
        Extract potential company/entity names from a query.

        Uses patterns to identify company names.

        Args:
            query: Natural language query

        Returns:
            List of potential entity names
        """
        entities = []

        # Pattern 1: Capitalized words that might be company names
        cap_pattern = r"\b([A-Z][a-z]+(?:\s+(?:&\s+)?[A-Z][a-z]+)*(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|PLC))?)\b"
        matches = re.findall(cap_pattern, query)
        entities.extend(matches)

        # Pattern 2: Ticker-like patterns (all caps 1-5 letters)
        ticker_pattern = r"\b([A-Z]{1,5})\b"
        ticker_matches = re.findall(ticker_pattern, query)
        common_words = {"I", "A", "THE", "FOR", "AND", "OR", "EPS", "PE", "ROE", "ROA"}
        for ticker in ticker_matches:
            if ticker not in common_words:
                entities.append(ticker)

        # Pattern 3: Possessive forms - strong signal of entity
        possessive_pattern = r"\b([a-zA-Z][a-zA-Z]+)(?:'s|'s)\b"
        possessive_matches = re.findall(possessive_pattern, query, re.IGNORECASE)
        for match in possessive_matches:
            entities.append(match.title())

        # Pattern 4: Words before financial terms
        financial_context_pattern = r"\b([a-zA-Z][a-zA-Z]+)\s+(?:revenue|earnings|income|profit|10-[kq]|filing|stock|shares|price)\b"
        context_matches = re.findall(financial_context_pattern, query, re.IGNORECASE)
        for match in context_matches:
            if match.lower() not in self._ignore_words:
                entities.append(match.title())

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            normalized = entity.lower().strip()

            if len(normalized) < 2 or normalized in self._ignore_words:
                continue

            check_name = re.sub(
                r"\s+(inc\.?|corp\.?|ltd\.?|llc|plc)$", "", normalized, flags=re.I
            ).strip()

            if (
                check_name not in seen
                and check_name not in self._ignore_words
                and len(check_name) >= 2
            ):
                seen.add(check_name)
                unique_entities.append(entity)

        return unique_entities

    async def _resolve_via_api(self, entity: str) -> ResolvedEntity | None:
        """
        Resolve entity using external API with resilience patterns.

        Uses OpenFIGI as a primary source (via shared module),
        then falls back to configured API.

        Args:
            entity: Entity name to resolve

        Returns:
            ResolvedEntity if found
        """
        # 1. Try OpenFIGI first (via shared module)
        try:
            result = await resolve_via_openfigi(
                query=entity,
                api_key=self._api_key if not self._api_endpoint else None,
                timeout=self._timeout_seconds,
            )
            if result:
                return result
        except Exception as e:
            logger.debug(f"OpenFIGI resolution failed: {e}")

        # 2. Fall back to configured external API
        if not self._api_endpoint:
            return None

        async def _make_request() -> ResolvedEntity | None:
            """Inner function for circuit breaker and retry."""
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
            logger.warning(f"Entity resolution circuit open, using fallback for: {entity}")
            return None
        except Exception as e:
            logger.warning(f"Error resolving entity via API: {e}")
            return None

    @property
    def circuit_breaker_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics for monitoring."""
        stats = self._circuit_breaker.stats
        return {
            "state": stats.state.value,
            "failure_count": stats.failure_count,
            "total_calls": stats.total_calls,
            "total_failures": stats.total_failures,
            "total_successes": stats.total_successes,
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self._circuit_breaker.reset()
