"""
Tests for Redis cache (using in-memory fallback).
"""

import pytest

from src.common.cache.redis_cache import (
    RedisCache,
    CacheConfig,
    MemoryCache,
    cache_key_for_entity,
    cache_key_for_query,
)


class TestMemoryCache:
    """Tests for MemoryCache (used as Redis fallback)."""

    @pytest.fixture
    def cache(self):
        """Create memory cache with small limits for testing."""
        return MemoryCache(max_size=3, default_ttl=60)

    @pytest.mark.asyncio
    async def test_get_set(self, cache):
        """Should get and set values."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache):
        """Should return None for missing key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        """Should evict oldest items when at capacity."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        # Cache is now full

        # Adding new key should evict oldest (key1)
        await cache.set("key4", "value4")

        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_access_updates_lru(self, cache):
        """Accessing a key should move it to end of LRU."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1, moving it to end
        await cache.get("key1")

        # key2 should now be evicted when adding new key
        await cache.set("key4", "value4")

        assert await cache.get("key1") == "value1"  # Still exists
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Should delete keys."""
        await cache.set("key1", "value1")
        result = await cache.delete("key1")
        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache):
        """Should return False for deleting nonexistent key."""
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Should clear all entries."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestRedisCache:
    """Tests for RedisCache (using in-memory fallback without Redis)."""

    @pytest.fixture
    def cache(self):
        """Create Redis cache that will fall back to memory."""
        config = CacheConfig(
            memory_cache_max_size=100,
            memory_cache_ttl_seconds=60,
        )
        return RedisCache(config)

    @pytest.mark.asyncio
    async def test_get_set_memory_fallback(self, cache):
        """Should work with memory fallback when Redis unavailable."""
        # Don't connect to Redis - should use memory cache
        await cache.set("key1", {"value": "test"})
        result = await cache.get("key1")
        assert result == {"value": "test"}

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache):
        """Should return None for missing key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_key_prefix(self, cache):
        """Should apply key prefix."""
        key = cache._make_key("test")
        assert key.startswith("nl2api:")
        assert "test" in key

    @pytest.mark.asyncio
    async def test_get_or_set(self, cache):
        """Should compute and cache value on miss."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return {"computed": True}

        # First call - computes value
        result1 = await cache.get_or_set("key", factory)
        assert result1 == {"computed": True}
        assert call_count == 1

        # Second call - returns cached value
        result2 = await cache.get_or_set("key", factory)
        assert result2 == {"computed": True}
        assert call_count == 1  # Factory not called again

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Should delete keys."""
        await cache.set("key1", "value1")
        result = await cache.delete("key1")
        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_stats_tracking(self, cache):
        """Should track cache statistics."""
        # Miss
        await cache.get("nonexistent")
        assert cache.stats.misses == 1

        # Set
        await cache.set("key1", "value1")
        assert cache.stats.sets == 1

        # Hit
        await cache.get("key1")
        assert cache.stats.hits == 1

        # Check hit rate
        assert cache.stats.hit_rate == 0.5  # 1 hit, 1 miss


class TestCacheKeyFunctions:
    """Tests for cache key generation functions."""

    def test_entity_cache_key(self):
        """Should generate consistent entity keys."""
        key1 = cache_key_for_entity("Apple")
        key2 = cache_key_for_entity("apple")  # Case insensitive
        assert key1 == key2
        assert "entity" in key1

    def test_entity_cache_key_with_type(self):
        """Should include entity type in key."""
        key = cache_key_for_entity("Apple", entity_type="company")
        assert "company" in key

    def test_query_cache_key(self):
        """Should generate query cache keys with hash."""
        key = cache_key_for_query("What is the EPS for Apple?")
        assert "rag" in key
        # Key should be deterministic
        key2 = cache_key_for_query("What is the EPS for Apple?")
        assert key == key2

    def test_query_cache_key_with_domain(self):
        """Should include domain in key."""
        key = cache_key_for_query("query", domain="estimates")
        assert "estimates" in key

    def test_query_cache_key_with_limit(self):
        """Should include limit in key."""
        key1 = cache_key_for_query("query", limit=5)
        key2 = cache_key_for_query("query", limit=10)
        assert key1 != key2
