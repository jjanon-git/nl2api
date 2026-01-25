"""
PostgreSQL Connection Pool Management

Provides async connection pooling using asyncpg with production-ready
configuration options.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import asyncpg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.evalkit.common.storage.config import StorageConfig

# Module-level pool for connection reuse
_pool: asyncpg.Pool | None = None


async def create_pool(
    postgres_url: str,
    min_size: int = 2,
    max_size: int = 10,
    command_timeout: int = 60,
    statement_cache_size: int = 1024,
    max_queries: int = 50000,
    max_inactive_connection_lifetime: float = 300.0,
) -> asyncpg.Pool:
    """
    Create a connection pool with production settings.

    Args:
        postgres_url: PostgreSQL connection URL
        min_size: Minimum pool connections
        max_size: Maximum pool connections
        command_timeout: Timeout for commands in seconds
        statement_cache_size: Size of prepared statement cache
        max_queries: Maximum queries per connection before recycling
        max_inactive_connection_lifetime: Seconds before idle connection closes

    Returns:
        asyncpg connection pool
    """
    global _pool
    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        postgres_url,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        statement_cache_size=statement_cache_size,
        max_queries=max_queries if max_queries > 0 else None,
        max_inactive_connection_lifetime=max_inactive_connection_lifetime,
    )
    return _pool


async def get_pool() -> asyncpg.Pool:
    """Get the existing pool or raise if not initialized."""
    if _pool is None:
        raise RuntimeError("Connection pool not initialized. Call create_pool() first.")
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_connection():
    """
    Context manager for getting a connection from the pool.

    Usage:
        async with get_connection() as conn:
            result = await conn.fetch("SELECT * FROM test_cases")
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def init_from_config(config: StorageConfig) -> asyncpg.Pool:
    """
    Initialize pool from StorageConfig.

    Args:
        config: Storage configuration

    Returns:
        asyncpg connection pool
    """
    return await create_pool(
        config.postgres_url,
        min_size=config.postgres_pool_min,
        max_size=config.postgres_pool_max,
        command_timeout=config.postgres_pool_command_timeout,
        statement_cache_size=config.postgres_pool_statement_cache_size,
        max_queries=config.postgres_pool_max_queries,
        max_inactive_connection_lifetime=config.postgres_pool_max_inactive_lifetime,
    )


async def check_pool_health() -> dict[str, Any]:
    """
    Check connection pool health status.

    Returns:
        Dictionary with pool health metrics:
        - healthy: Boolean indicating overall health
        - size: Current pool size
        - free: Number of idle connections
        - used: Number of active connections
        - min_size: Configured minimum size
        - max_size: Configured maximum size
        - utilization: Percentage of pool in use

    Raises:
        RuntimeError: If pool is not initialized
    """
    pool = await get_pool()

    size = pool.get_size()
    free = pool.get_idle_size()
    used = size - free
    min_size = pool.get_min_size()
    max_size = pool.get_max_size()
    utilization = (used / max_size * 100) if max_size > 0 else 0.0

    # Test a simple query to verify connectivity
    healthy = True
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        healthy = False

    return {
        "healthy": healthy,
        "size": size,
        "free": free,
        "used": used,
        "min_size": min_size,
        "max_size": max_size,
        "utilization": round(utilization, 1),
    }
