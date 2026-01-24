"""
Caching Module

Provides Redis-based caching with fallback to in-memory cache.
"""

from src.evalkit.common.cache.redis_cache import CacheConfig, RedisCache

__all__ = ["RedisCache", "CacheConfig"]
