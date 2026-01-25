"""
HTTP Client for Entity Resolution Service

Implements the EntityResolver protocol by calling the standalone
entity resolution HTTP service.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.evalkit.common.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryConfig,
    retry_with_backoff,
)
from src.evalkit.common.telemetry import get_tracer
from src.nl2api.resolution.protocols import ResolvedEntity

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class HttpEntityResolver:
    """
    Entity resolver that calls the standalone HTTP entity resolution service.

    Implements the EntityResolver protocol using HTTP calls to the service
    deployed at the configured endpoint.

    Features:
    - Circuit breaker: Fails fast when service is unhealthy
    - Retry with backoff: Handles transient failures gracefully
    - Timeout: Prevents hanging on slow responses
    - Connection pooling: Reuses HTTP connections for efficiency
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
        retry_max_attempts: int = 3,
    ):
        """
        Initialize the HTTP entity resolver client.

        Args:
            base_url: Base URL of the entity resolution service
                      (e.g., "http://localhost:8085")
            api_key: Optional API key for authentication
            timeout_seconds: Timeout for API calls
            circuit_failure_threshold: Failures before opening circuit
            circuit_recovery_seconds: Seconds before trying to recover
            retry_max_attempts: Maximum retry attempts for transient failures
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_seconds,
            name="entity-resolution-http",
        )

        # Retry config for transient failures
        self._retry_config = RetryConfig(
            max_attempts=retry_max_attempts,
            base_delay=0.5,
            max_delay=5.0,
            retryable_exceptions=(
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ),
        )

        # HTTP client (created lazily)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout_seconds),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def resolve(self, query: str) -> dict[str, str]:
        """
        Extract and resolve entities from a query.

        Calls the /api/extract endpoint to extract and resolve entities.

        Args:
            query: Natural language query

        Returns:
            Dictionary mapping entity names to RICs
        """
        with tracer.start_as_current_span("entity.http.resolve") as span:
            span.set_attribute("entity.query_length", len(query))

            async def _make_request() -> dict[str, str]:
                client = await self._get_client()
                response = await client.post(
                    "/api/extract",
                    json={"query": query},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("entities", {})

            try:
                result = await self._circuit_breaker.call(
                    retry_with_backoff,
                    _make_request,
                    config=self._retry_config,
                )
                span.set_attribute("entity.resolved_count", len(result))
                return result
            except CircuitOpenError:
                logger.warning("Entity resolution circuit open for query")
                span.set_attribute("entity.circuit_open", True)
                return {}
            except Exception as e:
                logger.warning(f"Error resolving entities via HTTP: {e}")
                span.set_attribute("entity.error", str(e))
                return {}

    async def resolve_single(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> ResolvedEntity | None:
        """
        Resolve a single entity to its identifier.

        Calls the /api/resolve endpoint.

        Args:
            entity: Entity name (e.g., "Apple Inc.")
            entity_type: Optional type hint

        Returns:
            ResolvedEntity if found
        """
        with tracer.start_as_current_span("entity.http.resolve_single") as span:
            span.set_attribute("entity.name", entity[:50])

            async def _make_request() -> ResolvedEntity | None:
                client = await self._get_client()
                payload: dict[str, Any] = {"entity": entity}
                if entity_type:
                    payload["entity_type"] = entity_type

                response = await client.post("/api/resolve", json=payload)
                response.raise_for_status()
                data = response.json()

                if data.get("found"):
                    return ResolvedEntity(
                        original=data["original"],
                        identifier=data["identifier"],
                        entity_type=data.get("entity_type", "company"),
                        confidence=data.get("confidence", 0.8),
                        alternatives=tuple(data.get("alternatives", [])),
                        metadata=data.get("metadata", {}),
                    )
                return None

            try:
                result = await self._circuit_breaker.call(
                    retry_with_backoff,
                    _make_request,
                    config=self._retry_config,
                )
                if result:
                    span.set_attribute("entity.found", True)
                    span.set_attribute("entity.identifier", result.identifier)
                else:
                    span.set_attribute("entity.found", False)
                return result
            except CircuitOpenError:
                logger.warning(f"Entity resolution circuit open for: {entity}")
                span.set_attribute("entity.circuit_open", True)
                return None
            except Exception as e:
                logger.warning(f"Error resolving entity via HTTP: {e}")
                span.set_attribute("entity.error", str(e))
                return None

    async def resolve_batch(self, entities: list[str]) -> list[ResolvedEntity]:
        """
        Resolve multiple entities in batch.

        Calls the /api/resolve/batch endpoint.

        Args:
            entities: List of entity names

        Returns:
            List of resolved entities (may be shorter than input)
        """
        with tracer.start_as_current_span("entity.http.resolve_batch") as span:
            span.set_attribute("entity.batch_size", len(entities))

            async def _make_request() -> list[ResolvedEntity]:
                client = await self._get_client()
                response = await client.post(
                    "/api/resolve/batch",
                    json={"entities": entities},
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("resolved", []):
                    results.append(
                        ResolvedEntity(
                            original=item["original"],
                            identifier=item["identifier"],
                            entity_type=item.get("entity_type", "company"),
                            confidence=item.get("confidence", 0.8),
                            alternatives=tuple(item.get("alternatives", [])),
                            metadata=item.get("metadata", {}),
                        )
                    )
                return results

            try:
                results = await self._circuit_breaker.call(
                    retry_with_backoff,
                    _make_request,
                    config=self._retry_config,
                )
                span.set_attribute("entity.resolved_count", len(results))
                return results
            except CircuitOpenError:
                logger.warning("Entity resolution circuit open for batch")
                span.set_attribute("entity.circuit_open", True)
                return []
            except Exception as e:
                logger.warning(f"Error resolving entities via HTTP: {e}")
                span.set_attribute("entity.error", str(e))
                return []

    async def health_check(self) -> bool:
        """
        Check if the entity resolution service is healthy.

        Returns:
            True if service is healthy
        """
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def ready_check(self) -> dict[str, Any]:
        """
        Check if the entity resolution service is ready.

        Returns:
            Readiness status with dependency checks
        """
        try:
            client = await self._get_client()
            response = await client.get("/ready")
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
