"""
Tests for RoutingCache with Redis and pgvector

Tests the tiered cache implementation with mocked external dependencies.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from src.nl2api.routing.cache import RoutingCache
from src.nl2api.routing.protocols import RouterResult


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self.get = AsyncMock(side_effect=self._get)
        self.setex = AsyncMock(side_effect=self._setex)

    async def _get(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def _setex(self, key: str, seconds: int, value: str) -> None:
        self._data[key] = value.encode() if isinstance(value, str) else value


class MockPostgresPool:
    """Mock PostgreSQL pool for testing."""

    def __init__(self):
        self._rows: list[dict] = []
        self._executed: list[tuple] = []

    def connection(self):
        return MockConnection(self)


class MockConnection:
    """Mock PostgreSQL connection."""

    def __init__(self, pool: MockPostgresPool):
        self._pool = pool

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def fetchrow(self, query: str, *args):
        self._pool._executed.append((query, args))
        return self._pool._rows[0] if self._pool._rows else None

    async def execute(self, query: str, *args):
        self._pool._executed.append((query, args))


async def mock_embedder(text: str) -> list[float]:
    """Mock embedder that returns consistent embeddings."""
    # Return a simple hash-based embedding for testing
    hash_val = hash(text)
    return [float(hash_val % 1000) / 1000.0] * 1536


# =============================================================================
# RoutingCache Tests
# =============================================================================


class TestRoutingCacheInit:
    """Tests for RoutingCache initialization."""

    def test_init_without_dependencies(self):
        """Test cache initializes without Redis or pgvector."""
        cache = RoutingCache(
            redis=None,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        # Should not raise
        assert cache._redis is None
        assert cache._pg_pool is None

    def test_init_with_redis_only(self):
        """Test cache with only Redis."""
        redis = MockRedisClient()
        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        assert cache._redis is redis
        assert cache._semantic_enabled is False

    def test_init_with_semantic_enabled(self):
        """Test cache with semantic search enabled."""
        redis = MockRedisClient()
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=redis,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        # _semantic_enabled is truthy when all dependencies are present
        assert cache._semantic_enabled

    def test_semantic_disabled_without_pg_pool(self):
        """Test semantic is disabled if pg_pool missing."""
        cache = RoutingCache(
            redis=MockRedisClient(),
            pg_pool=None,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        # _semantic_enabled is falsy when pg_pool is missing
        assert not cache._semantic_enabled

    def test_semantic_disabled_without_embedder(self):
        """Test semantic is disabled if embedder missing."""
        cache = RoutingCache(
            redis=MockRedisClient(),
            pg_pool=MockPostgresPool(),
            embedder=None,
            semantic_cache_enabled=True,
        )

        # _semantic_enabled is falsy when embedder is missing
        assert not cache._semantic_enabled


class TestRoutingCacheGet:
    """Tests for RoutingCache.get()."""

    @pytest.mark.asyncio
    async def test_get_returns_none_without_backends(self):
        """Test get returns None without any backend."""
        cache = RoutingCache(
            redis=None,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        result = await cache.get("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_exact_match_from_redis(self):
        """Test exact match lookup from Redis."""
        redis = MockRedisClient()
        # Pre-populate Redis with cached data
        redis._data["route:098f6bcd4621d373cade4e832627b4f6"] = json.dumps({
            "domain": "datastream",
            "confidence": 0.9,
            "reasoning": "Test",
        }).encode()

        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
            key_prefix="route:",
        )

        result = await cache.get("test")

        assert result is not None
        assert result.domain == "datastream"
        assert result.confidence == 0.9
        assert result.cached is True

    @pytest.mark.asyncio
    async def test_get_handles_redis_error(self):
        """Test get handles Redis errors gracefully."""
        redis = MockRedisClient()
        redis.get = AsyncMock(side_effect=Exception("Redis connection error"))

        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        # Should not raise, return None
        result = await cache.get("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_handles_invalid_json(self):
        """Test get handles invalid JSON in Redis."""
        redis = MockRedisClient()
        redis._data["route:098f6bcd4621d373cade4e832627b4f6"] = b"not valid json"

        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
            key_prefix="route:",
        )

        result = await cache.get("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_semantic_lookup_when_exact_misses(self):
        """Test semantic lookup when exact match misses."""
        redis = MockRedisClient()  # Empty cache
        pg_pool = MockPostgresPool()
        pg_pool._rows = [{
            "domain": "estimates",
            "confidence": 0.85,
            "reasoning": "Semantic match",
            "similarity": 0.95,
        }]

        cache = RoutingCache(
            redis=redis,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
            similarity_threshold=0.9,
        )

        result = await cache.get("test query")

        assert result is not None
        assert result.domain == "estimates"
        assert result.cached is True

    @pytest.mark.asyncio
    async def test_get_semantic_returns_none_when_no_match(self):
        """Test semantic lookup returns None when no similar query."""
        redis = MockRedisClient()
        pg_pool = MockPostgresPool()
        pg_pool._rows = []  # No matches

        cache = RoutingCache(
            redis=redis,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        result = await cache.get("unique query")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_handles_semantic_error(self):
        """Test get handles semantic lookup errors gracefully."""
        redis = MockRedisClient()
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=redis,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        # Make fetchrow raise an exception
        original_connection = pg_pool.connection

        def broken_connection():
            conn = original_connection()
            conn.fetchrow = AsyncMock(side_effect=Exception("DB error"))
            return conn

        pg_pool.connection = broken_connection

        result = await cache.get("test query")

        assert result is None


class TestRoutingCacheSet:
    """Tests for RoutingCache.set()."""

    @pytest.mark.asyncio
    async def test_set_does_not_cache_unknown_domain(self):
        """Test that unknown domains are not cached."""
        redis = MockRedisClient()

        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        result = RouterResult(domain="unknown", confidence=0.0)
        await cache.set("test query", result)

        redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_does_not_cache_zero_confidence(self):
        """Test that zero confidence results are not cached."""
        redis = MockRedisClient()

        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        result = RouterResult(domain="datastream", confidence=0.0)
        await cache.set("test query", result)

        redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_stores_in_redis(self):
        """Test that results are stored in Redis."""
        redis = MockRedisClient()

        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
            ttl_seconds=3600,
        )

        result = RouterResult(
            domain="datastream",
            confidence=0.9,
            reasoning="Test routing",
        )
        await cache.set("test query", result)

        redis.setex.assert_called_once()
        call_args = redis.setex.call_args
        assert call_args[0][1] == 3600  # TTL

    @pytest.mark.asyncio
    async def test_set_handles_redis_error(self):
        """Test set handles Redis errors gracefully."""
        redis = MockRedisClient()
        redis.setex = AsyncMock(side_effect=Exception("Redis write error"))

        cache = RoutingCache(
            redis=redis,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
        )

        result = RouterResult(domain="datastream", confidence=0.9)

        # Should not raise
        await cache.set("test query", result)

    @pytest.mark.asyncio
    async def test_set_stores_in_pgvector(self):
        """Test that results are stored in pgvector."""
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=None,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        result = RouterResult(
            domain="estimates",
            confidence=0.85,
            reasoning="Semantic store test",
        )
        await cache.set("test query", result)

        # Verify execute was called
        assert len(pg_pool._executed) > 0
        query, args = pg_pool._executed[0]
        assert "INSERT INTO routing_cache" in query
        assert args[2] == "estimates"  # domain

    @pytest.mark.asyncio
    async def test_set_handles_pgvector_error(self):
        """Test set handles pgvector errors gracefully."""
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=None,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        # Make execute raise an exception
        original_connection = pg_pool.connection

        def broken_connection():
            conn = original_connection()
            conn.execute = AsyncMock(side_effect=Exception("DB write error"))
            return conn

        pg_pool.connection = broken_connection

        result = RouterResult(domain="datastream", confidence=0.9)

        # Should not raise
        await cache.set("test query", result)


class TestRoutingCacheInvalidate:
    """Tests for RoutingCache.invalidate()."""

    @pytest.mark.asyncio
    async def test_invalidate_removes_from_pgvector(self):
        """Test invalidate removes from pgvector."""
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=None,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        await cache.invalidate("test query")

        assert len(pg_pool._executed) > 0
        query, args = pg_pool._executed[0]
        assert "DELETE FROM routing_cache" in query

    @pytest.mark.asyncio
    async def test_invalidate_handles_error(self):
        """Test invalidate handles errors gracefully."""
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=None,
            pg_pool=pg_pool,
            embedder=mock_embedder,
            semantic_cache_enabled=True,
        )

        original_connection = pg_pool.connection

        def broken_connection():
            conn = original_connection()
            conn.execute = AsyncMock(side_effect=Exception("DB error"))
            return conn

        pg_pool.connection = broken_connection

        # Should not raise
        await cache.invalidate("test query")


class TestRoutingCacheClearAll:
    """Tests for RoutingCache.clear_all()."""

    @pytest.mark.asyncio
    async def test_clear_all_truncates_pgvector(self):
        """Test clear_all truncates pgvector table."""
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=None,
            pg_pool=pg_pool,
            embedder=None,
            semantic_cache_enabled=False,
        )

        await cache.clear_all()

        assert len(pg_pool._executed) > 0
        query, _ = pg_pool._executed[0]
        assert "TRUNCATE TABLE routing_cache" in query

    @pytest.mark.asyncio
    async def test_clear_all_handles_error(self):
        """Test clear_all handles errors gracefully."""
        pg_pool = MockPostgresPool()

        cache = RoutingCache(
            redis=None,
            pg_pool=pg_pool,
            embedder=None,
            semantic_cache_enabled=False,
        )

        original_connection = pg_pool.connection

        def broken_connection():
            conn = original_connection()
            conn.execute = AsyncMock(side_effect=Exception("DB error"))
            return conn

        pg_pool.connection = broken_connection

        # Should not raise
        await cache.clear_all()


class TestRoutingCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_cache_key_is_consistent(self):
        """Test that same query generates same key."""
        cache = RoutingCache(
            redis=None,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
            key_prefix="test:",
        )

        key1 = cache._cache_key("What is Apple's price?")
        key2 = cache._cache_key("What is Apple's price?")

        assert key1 == key2

    def test_cache_key_differs_for_different_queries(self):
        """Test that different queries generate different keys."""
        cache = RoutingCache(
            redis=None,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
            key_prefix="test:",
        )

        key1 = cache._cache_key("query A")
        key2 = cache._cache_key("query B")

        assert key1 != key2

    def test_cache_key_uses_prefix(self):
        """Test that cache key uses configured prefix."""
        cache = RoutingCache(
            redis=None,
            pg_pool=None,
            embedder=None,
            semantic_cache_enabled=False,
            key_prefix="custom:",
        )

        key = cache._cache_key("test")

        assert key.startswith("custom:")
