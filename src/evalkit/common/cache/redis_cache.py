"""
Redis Cache Implementation

Provides distributed caching with Redis, with fallback to in-memory cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from src.evalkit.common.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for Redis cache."""

    redis_url: str = "redis://localhost:6379/0"
    default_ttl_seconds: int = 3600  # 1 hour
    key_prefix: str = "nl2api:"
    max_connections: int = 10
    socket_timeout: float = 5.0
    # Fallback in-memory cache settings
    memory_cache_max_size: int = 1000
    memory_cache_ttl_seconds: int = 300  # 5 minutes


@dataclass
class CacheStats:
    """Statistics for cache operations."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


class MemoryCache:
    """
    Simple in-memory LRU cache with TTL support.

    Used as fallback when Redis is unavailable.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        async with self._lock:
            ttl = ttl or self._default_ttl
            expiry = time.time() + ttl

            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, expiry)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()


class RedisCache:
    """
    Redis-based cache with automatic fallback to in-memory cache.

    Features:
    - Async Redis operations
    - Automatic JSON serialization
    - Configurable TTL
    - In-memory fallback when Redis unavailable
    - Stats tracking
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize Redis cache.

        Args:
            config: Cache configuration
        """
        self._config = config or CacheConfig()
        self._redis: Any = None  # redis.asyncio.Redis
        self._connected = False
        self._memory_cache = MemoryCache(
            max_size=self._config.memory_cache_max_size,
            default_ttl=self._config.memory_cache_ttl_seconds,
        )
        self._stats = CacheStats()
        self._use_redis = True

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to Redis server.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            logger.warning(
                "redis package not installed, using memory cache only. "
                "Install with: pip install redis"
            )
            self._use_redis = False
            return False

        try:
            self._redis = redis.from_url(
                self._config.redis_url,
                max_connections=self._config.max_connections,
                socket_timeout=self._config.socket_timeout,
                decode_responses=True,
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self._config.redis_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using memory cache.")
            self._connected = False
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self._config.key_prefix}{key}"

    def _hash_key(self, data: str) -> str:
        """Create hash key from data string."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with tracer.start_as_current_span("cache.get") as span:
            full_key = self._make_key(key)
            span.set_attribute("cache.key", key[:50])  # Truncate for span

            # Try Redis first
            if self._use_redis and self._connected:
                try:
                    value = await self._redis.get(full_key)
                    if value is not None:
                        self._stats.hits += 1
                        span.set_attribute("cache.hit", True)
                        span.set_attribute("cache.source", "redis")
                        return json.loads(value)
                    self._stats.misses += 1
                    span.set_attribute("cache.hit", False)
                    span.set_attribute("cache.source", "redis")
                    return None
                except Exception as e:
                    self._stats.errors += 1
                    span.set_attribute("cache.error", str(e))
                    logger.warning(f"Redis get error: {e}")
                    # Fall through to memory cache

            # Fallback to memory cache
            value = await self._memory_cache.get(full_key)
            if value is not None:
                self._stats.hits += 1
                span.set_attribute("cache.hit", True)
            else:
                self._stats.misses += 1
                span.set_attribute("cache.hit", False)
            span.set_attribute("cache.source", "memory")
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (default from config)

        Returns:
            True if set successfully
        """
        with tracer.start_as_current_span("cache.set") as span:
            full_key = self._make_key(key)
            ttl = ttl or self._config.default_ttl_seconds
            span.set_attribute("cache.key", key[:50])  # Truncate for span
            span.set_attribute("cache.ttl", ttl)

            # Try Redis first
            if self._use_redis and self._connected:
                try:
                    serialized = json.dumps(value)
                    await self._redis.setex(full_key, ttl, serialized)
                    self._stats.sets += 1
                    span.set_attribute("cache.source", "redis")
                    return True
                except Exception as e:
                    self._stats.errors += 1
                    span.set_attribute("cache.error", str(e))
                    logger.warning(f"Redis set error: {e}")
                    # Fall through to memory cache

            # Fallback to memory cache
            await self._memory_cache.set(full_key, value, ttl)
            self._stats.sets += 1
            span.set_attribute("cache.source", "memory")
            return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        full_key = self._make_key(key)

        deleted = False
        if self._use_redis and self._connected:
            try:
                deleted = await self._redis.delete(full_key) > 0
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")

        # Also delete from memory cache
        mem_deleted = await self._memory_cache.delete(full_key)
        return deleted or mem_deleted

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        ttl: int | None = None,
    ) -> T:
        """
        Get value from cache or compute and cache it.

        Args:
            key: Cache key
            factory: Async function to compute value if not cached
            ttl: Time-to-live in seconds

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        value = await factory()

        # Cache it
        await self.set(key, value, ttl)

        return value

    async def clear_prefix(self, prefix: str) -> int:
        """
        Clear all keys with given prefix.

        Args:
            prefix: Key prefix to clear

        Returns:
            Number of keys deleted
        """
        full_prefix = self._make_key(prefix)
        deleted = 0

        if self._use_redis and self._connected:
            try:
                cursor = "0"
                while cursor != 0:
                    cursor, keys = await self._redis.scan(
                        cursor=cursor,
                        match=f"{full_prefix}*",
                        count=100,
                    )
                    if keys:
                        deleted += await self._redis.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear_prefix error: {e}")

        return deleted


def cache_key_for_entity(entity: str, entity_type: str | None = None) -> str:
    """
    Generate cache key for entity resolution.

    Args:
        entity: Entity name
        entity_type: Optional entity type

    Returns:
        Cache key string
    """
    key_parts = ["entity", entity.lower().strip()]
    if entity_type:
        key_parts.append(entity_type)
    return ":".join(key_parts)


def cache_key_for_query(
    query: str,
    domain: str | None = None,
    limit: int = 10,
) -> str:
    """
    Generate cache key for RAG query.

    Args:
        query: Query string
        domain: Optional domain filter
        limit: Result limit

    Returns:
        Cache key string
    """
    # Hash the query to keep key short
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    key_parts = ["rag", query_hash]
    if domain:
        key_parts.append(domain)
    key_parts.append(str(limit))
    return ":".join(key_parts)
