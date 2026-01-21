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
from typing import TYPE_CHECKING, Any

from src.common.resilience import CircuitBreaker, CircuitOpenError, RetryConfig, retry_with_backoff
from src.nl2api.resolution.protocols import ResolvedEntity

if TYPE_CHECKING:
    from src.common.cache import RedisCache

logger = logging.getLogger(__name__)


class ExternalEntityResolver:
    """
    Entity resolver using an external API for company/RIC resolution.

    Extracts company mentions from queries and resolves them to RICs
    using a configurable external service.

    Features:
    - Circuit breaker: Fails fast when external service is unhealthy
    - Retry with backoff: Handles transient failures gracefully
    - Timeout: Prevents hanging on slow responses
    - Caching: In-memory and optional Redis caching for resolved entities
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
        redis_cache: "RedisCache | None" = None,
        redis_cache_ttl_seconds: int = 86400,
    ):
        """
        Initialize the entity resolver.

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
        """
        self._api_endpoint = api_endpoint
        self._api_key = api_key
        self._use_cache = use_cache
        self._timeout_seconds = timeout_seconds
        self._cache: dict[str, ResolvedEntity] = {}  # In-memory L1 cache
        self._redis_cache = redis_cache  # Optional L2 distributed cache
        self._redis_ttl = redis_cache_ttl_seconds

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

        # Common company name to RIC mappings (fallback)
        self._common_mappings: dict[str, str] = {
            "apple": "AAPL.O",
            "microsoft": "MSFT.O",
            "google": "GOOGL.O",
            "alphabet": "GOOGL.O",
            "amazon": "AMZN.O",
            "meta": "META.O",
            "facebook": "META.O",
            "tesla": "TSLA.O",
            "nvidia": "NVDA.O",
            "jpmorgan": "JPM.N",
            "jp morgan": "JPM.N",
            "goldman sachs": "GS.N",
            "bank of america": "BAC.N",
            "wells fargo": "WFC.N",
            "exxon": "XOM.N",
            "chevron": "CVX.N",
            "walmart": "WMT.N",
            "johnson & johnson": "JNJ.N",
            "j&j": "JNJ.N",
            "procter & gamble": "PG.N",
            "p&g": "PG.N",
            "coca-cola": "KO.N",
            "pepsi": "PEP.O",
            "pepsico": "PEP.O",
            "disney": "DIS.N",
            "netflix": "NFLX.O",
            "adobe": "ADBE.O",
            "salesforce": "CRM.N",
            "oracle": "ORCL.N",
            "intel": "INTC.O",
            "amd": "AMD.O",
            "cisco": "CSCO.O",
            "ibm": "IBM.N",
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
        entities = self._extract_entities(query)
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
        1. L1: In-memory cache (fastest)
        2. L2: Redis cache (distributed, if configured)
        3. Static mappings
        4. External API

        Args:
            entity: Entity name (e.g., "Apple Inc.")
            entity_type: Optional type hint

        Returns:
            ResolvedEntity if found
        """
        # Normalize entity name
        normalized = entity.lower().strip()
        normalized = re.sub(r'\s+(inc\.?|corp\.?|ltd\.?|llc|plc)$', '', normalized, flags=re.I)
        normalized = normalized.strip()

        # L1: Check in-memory cache
        if self._use_cache and normalized in self._cache:
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
                )
                # Populate L1 cache
                self._cache[normalized] = result
                return result

        # Check common mappings
        if normalized in self._common_mappings:
            result = ResolvedEntity(
                original=entity,
                identifier=self._common_mappings[normalized],
                entity_type="company",
                confidence=0.95,
            )
            await self._cache_result(normalized, result)
            return result

        # Try external API if configured
        if self._api_endpoint:
            result = await self._resolve_via_api(entity)
            if result:
                await self._cache_result(normalized, result)
                return result

        # No resolution found
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
        # Matches: "Apple", "Microsoft Corporation", "JP Morgan"
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+(?:&\s+)?[A-Z][a-z]+)*(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|PLC))?)\b'
        matches = re.findall(cap_pattern, query)
        entities.extend(matches)

        # Pattern 2: Known company name patterns
        for company in self._common_mappings.keys():
            if company.lower() in query.lower():
                # Find the actual casing in the query
                pattern = re.compile(re.escape(company), re.IGNORECASE)
                match = pattern.search(query)
                if match:
                    entities.append(match.group())

        # Pattern 3: Ticker-like patterns (all caps 1-5 letters)
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        ticker_matches = re.findall(ticker_pattern, query)
        # Only add if they look like real tickers (not common words)
        common_words = {"I", "A", "THE", "FOR", "AND", "OR", "EPS", "PE", "ROE", "ROA"}
        for ticker in ticker_matches:
            if ticker not in common_words:
                entities.append(ticker)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            normalized = entity.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_entities.append(entity)

        return unique_entities

    async def _resolve_via_api(self, entity: str) -> ResolvedEntity | None:
        """
        Resolve entity using external API with resilience patterns.

        Uses circuit breaker to fail fast when service is unhealthy,
        and retry with backoff for transient failures.

        Args:
            entity: Entity name to resolve

        Returns:
            ResolvedEntity if found, None if not found or on error

        Note:
            Returns None (graceful degradation) on circuit open or
            after retry exhaustion, allowing fallback to static mappings.
        """
        if not self._api_endpoint:
            return None

        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed, cannot use external API")
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
                            )
                    elif response.status >= 500:
                        # Server error - should trigger retry/circuit
                        raise ConnectionError(f"Server error: {response.status}")
            return None

        try:
            # Apply circuit breaker and retry
            return await self._circuit_breaker.call(
                retry_with_backoff,
                _make_request,
                config=self._retry_config,
            )
        except CircuitOpenError:
            logger.warning(
                f"Entity resolution circuit open, using fallback for: {entity}"
            )
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
