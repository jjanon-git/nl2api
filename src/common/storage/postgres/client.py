"""
PostgreSQL Connection Pool Management

Provides async connection pooling using asyncpg.
"""

from __future__ import annotations

import asyncpg
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.config import StorageConfig

# Module-level pool for connection reuse
_pool: asyncpg.Pool | None = None


async def create_pool(
    postgres_url: str,
    min_size: int = 2,
    max_size: int = 10,
) -> asyncpg.Pool:
    """
    Create a connection pool.

    Args:
        postgres_url: PostgreSQL connection URL
        min_size: Minimum pool connections
        max_size: Maximum pool connections

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
        command_timeout=60,
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


async def init_from_config(config: "StorageConfig") -> asyncpg.Pool:
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
    )
