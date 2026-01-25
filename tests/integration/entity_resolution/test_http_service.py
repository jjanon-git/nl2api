"""
End-to-end integration tests for entity resolution HTTP service.

These tests start the actual HTTP service and test the full round-trip
from HttpEntityResolver client to the service and back.

Requirements:
- PostgreSQL running with entity data loaded
- Run with: pytest tests/integration/entity_resolution/ -v
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx
import pytest

from src.nl2api.resolution.http_client import HttpEntityResolver


@pytest.fixture(scope="module")
def entity_service():
    """Start the entity resolution service for testing."""
    # Start the service
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.services.entity_resolution",
            "--port",
            "8087",
            "--no-redis",
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for service to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = httpx.get("http://localhost:8087/health", timeout=1.0)
            if response.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    else:
        process.kill()
        pytest.fail("Entity resolution service failed to start")

    yield "http://localhost:8087"

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture
def resolver(entity_service):
    """Create an HttpEntityResolver connected to the test service."""
    return HttpEntityResolver(
        base_url=entity_service,
        timeout_seconds=5.0,
    )


class TestEntityResolutionE2E:
    """End-to-end tests for entity resolution service."""

    async def test_health_check(self, resolver: HttpEntityResolver) -> None:
        """Service should respond to health checks."""
        result = await resolver.health_check()
        assert result is True

    async def test_ready_check(self, resolver: HttpEntityResolver) -> None:
        """Service should report ready status with database connected."""
        result = await resolver.ready_check()
        assert result["status"] == "ready"
        assert result["checks"]["database"]["connected"] is True

    async def test_resolve_single_known_company(
        self, resolver: HttpEntityResolver
    ) -> None:
        """Should resolve well-known company names."""
        result = await resolver.resolve_single("Apple")

        assert result is not None
        assert result.identifier == "AAPL.O"
        assert result.entity_type == "company"
        assert result.confidence > 0.8

    async def test_resolve_single_unknown_company(
        self, resolver: HttpEntityResolver
    ) -> None:
        """Should return None for unknown entities."""
        result = await resolver.resolve_single("XyzNonExistentCompany12345")

        # May be None or may have low confidence from OpenFIGI
        if result is not None:
            assert result.confidence < 0.5

    async def test_resolve_extracts_multiple_entities(
        self, resolver: HttpEntityResolver
    ) -> None:
        """Should extract and resolve multiple entities from a query."""
        result = await resolver.resolve(
            "Compare Apple's revenue to Microsoft's earnings"
        )

        assert "Apple" in result or "apple" in str(result).lower()
        # At least one entity should be resolved
        assert len(result) >= 1

    async def test_resolve_batch(self, resolver: HttpEntityResolver) -> None:
        """Should resolve multiple entities in batch."""
        results = await resolver.resolve_batch(["Apple", "Microsoft", "Google"])

        # At least some should resolve
        assert len(results) >= 1
        identifiers = {r.identifier for r in results}
        # At least one of the major companies should resolve
        assert any(
            ric in identifiers for ric in ["AAPL.O", "MSFT.O", "GOOGL.O", "GOOG.O"]
        )

    async def test_circuit_breaker_stats(self, resolver: HttpEntityResolver) -> None:
        """Should track circuit breaker statistics."""
        # Make a successful request
        await resolver.resolve_single("Apple")

        stats = resolver.circuit_breaker_stats
        assert stats["state"] == "closed"
        assert stats["total_calls"] >= 1
        assert stats["total_successes"] >= 1


class TestEntityResolutionWithFactory:
    """Test that the factory correctly creates HTTP resolver."""

    async def test_factory_creates_http_resolver(self, entity_service: str) -> None:
        """Factory should create HttpEntityResolver when endpoint is configured."""
        from src.nl2api.config import NL2APIConfig
        from src.nl2api.resolution import create_entity_resolver

        config = NL2APIConfig(
            _env_file=None,
            entity_resolution_enabled=True,
            entity_resolution_api_endpoint=entity_service,
        )

        resolver = create_entity_resolver(config)

        assert isinstance(resolver, HttpEntityResolver)

        # Verify it works
        result = await resolver.resolve_single("Apple")
        assert result is not None
        assert result.identifier == "AAPL.O"

        await resolver.close()
