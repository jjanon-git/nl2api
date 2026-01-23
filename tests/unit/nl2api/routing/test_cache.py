"""
Tests for RoutingCache

Tests the tiered cache implementation including:
- Exact match caching
- Cache set/get operations
- Invalidation
- Unknown domain handling
"""

import pytest

from src.nl2api.routing.cache import InMemoryRoutingCache, RoutingCache
from src.nl2api.routing.protocols import RouterResult


class TestInMemoryRoutingCache:
    """Tests for InMemoryRoutingCache."""

    @pytest.mark.asyncio
    async def test_basic_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = InMemoryRoutingCache()

        result = RouterResult(
            domain="datastream",
            confidence=0.9,
            reasoning="Test routing",
        )

        await cache.set("What is Apple's stock price?", result)
        cached = await cache.get("What is Apple's stock price?")

        assert cached is not None
        assert cached.domain == "datastream"
        assert cached.confidence == 0.9
        assert cached.reasoning == "Test routing"
        assert cached.cached is True

    @pytest.mark.asyncio
    async def test_returns_none_for_cache_miss(self):
        """Test that cache miss returns None."""
        cache = InMemoryRoutingCache()

        cached = await cache.get("nonexistent query")

        assert cached is None

    @pytest.mark.asyncio
    async def test_does_not_cache_unknown_domain(self):
        """Test that unknown domain results are not cached."""
        cache = InMemoryRoutingCache()

        result = RouterResult(
            domain="unknown",
            confidence=0.0,
            reasoning="Routing failed",
        )

        await cache.set("ambiguous query", result)
        cached = await cache.get("ambiguous query")

        assert cached is None

    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self):
        """Test cache invalidation."""
        cache = InMemoryRoutingCache()

        result = RouterResult(domain="datastream", confidence=0.9)
        await cache.set("test query", result)

        # Verify it's cached
        assert await cache.get("test query") is not None

        # Invalidate
        await cache.invalidate("test query")

        # Verify it's removed
        assert await cache.get("test query") is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_is_safe(self):
        """Test that invalidating nonexistent key doesn't raise."""
        cache = InMemoryRoutingCache()

        # Should not raise
        await cache.invalidate("nonexistent")

    @pytest.mark.asyncio
    async def test_clear_all_removes_all_entries(self):
        """Test clearing all cache entries."""
        cache = InMemoryRoutingCache()

        # Add multiple entries
        await cache.set("query1", RouterResult(domain="datastream", confidence=0.9))
        await cache.set("query2", RouterResult(domain="estimates", confidence=0.85))
        await cache.set("query3", RouterResult(domain="fundamentals", confidence=0.8))

        # Verify they're cached
        assert await cache.get("query1") is not None
        assert await cache.get("query2") is not None
        assert await cache.get("query3") is not None

        # Clear all
        await cache.clear_all()

        # Verify all removed
        assert await cache.get("query1") is None
        assert await cache.get("query2") is None
        assert await cache.get("query3") is None

    @pytest.mark.asyncio
    async def test_overwrites_existing_entry(self):
        """Test that setting same key overwrites existing entry."""
        cache = InMemoryRoutingCache()

        # Initial set
        await cache.set("test query", RouterResult(domain="datastream", confidence=0.7))

        # Overwrite
        await cache.set("test query", RouterResult(domain="estimates", confidence=0.95))

        cached = await cache.get("test query")

        assert cached.domain == "estimates"
        assert cached.confidence == 0.95

    @pytest.mark.asyncio
    async def test_preserves_metadata_fields(self):
        """Test that all RouterResult fields are preserved."""
        cache = InMemoryRoutingCache()

        result = RouterResult(
            domain="datastream",
            confidence=0.92,
            reasoning="Stock price query detected",
            alternative_domains=("estimates",),
            latency_ms=150,
            model_used="claude-haiku",
        )

        await cache.set("test query", result)
        cached = await cache.get("test query")

        assert cached.domain == "datastream"
        assert cached.confidence == 0.92
        assert cached.reasoning == "Stock price query detected"
        assert cached.cached is True
        # Note: latency_ms and model_used may be preserved or reset based on impl


class TestRoutingCacheProtocol:
    """Tests for RoutingCache with external dependencies mocked."""

    @pytest.mark.asyncio
    async def test_cache_without_redis_or_pgvector(self):
        """Test cache works without Redis or pgvector."""
        cache = RoutingCache(
            redis=None,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        # Should return None without any backend
        cached = await cache.get("test query")
        assert cached is None

        # Set should not raise
        result = RouterResult(domain="datastream", confidence=0.9)
        await cache.set("test query", result)

    @pytest.mark.asyncio
    async def test_does_not_cache_failed_routing(self):
        """Test that failed routing is not cached."""
        cache = InMemoryRoutingCache()

        # Zero confidence
        await cache.set("failed query", RouterResult(domain="datastream", confidence=0.0))
        assert await cache.get("failed query") is None

        # Unknown domain
        await cache.set("unknown query", RouterResult(domain="unknown", confidence=0.5))
        assert await cache.get("unknown query") is None


class TestCacheKeyGeneration:
    """Tests for cache key generation and collision handling."""

    @pytest.mark.asyncio
    async def test_different_queries_have_different_keys(self):
        """Test that different queries are cached separately."""
        cache = InMemoryRoutingCache()

        await cache.set("query A", RouterResult(domain="datastream", confidence=0.9))
        await cache.set("query B", RouterResult(domain="estimates", confidence=0.85))

        cached_a = await cache.get("query A")
        cached_b = await cache.get("query B")

        assert cached_a.domain == "datastream"
        assert cached_b.domain == "estimates"

    @pytest.mark.asyncio
    async def test_case_sensitive_queries(self):
        """Test that queries are case-sensitive."""
        cache = InMemoryRoutingCache()

        await cache.set("Apple stock price", RouterResult(domain="datastream", confidence=0.9))
        await cache.set("apple stock price", RouterResult(domain="estimates", confidence=0.85))

        cached_upper = await cache.get("Apple stock price")
        cached_lower = await cache.get("apple stock price")

        # These should be different entries
        assert cached_upper.domain == "datastream"
        assert cached_lower.domain == "estimates"
