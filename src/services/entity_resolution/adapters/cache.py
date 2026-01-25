"""
Redis cache adapter for entity resolution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Simple Redis cache wrapper.

    Provides get/set operations with JSON serialization.
    """

    def __init__(self, client: Any):
        """
        Initialize with a redis client.

        Args:
            client: Redis client (from redis-py)
        """
        self._client = client

    async def get(self, key: str) -> dict | None:
        """Get value from cache."""
        try:
            data = await self._client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
        return None

    async def set(self, key: str, value: dict, ex: int | None = None) -> bool:
        """Set value in cache with optional expiry."""
        try:
            data = json.dumps(value)
            await self._client.set(key, data, ex=ex)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False


async def create_redis_client(redis_url: str) -> Any:
    """
    Create a Redis client.

    Args:
        redis_url: Redis connection URL

    Returns:
        Redis client
    """
    import redis.asyncio as redis

    logger.info("Connecting to Redis")
    client = redis.from_url(redis_url, decode_responses=True)
    # Test connection
    await client.ping()
    logger.info("Redis connected")
    return client


async def check_redis_health(client: Any) -> dict:
    """
    Check Redis health.

    Returns:
        Health status dict
    """
    try:
        await client.ping()
        return {"connected": True}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"connected": False, "error": str(e)}
