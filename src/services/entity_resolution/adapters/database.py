"""
Database adapter for entity resolution.

Handles PostgreSQL connection pool management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


async def create_db_pool(
    postgres_url: str,
    min_size: int = 2,
    max_size: int = 10,
) -> asyncpg.Pool:
    """
    Create a PostgreSQL connection pool.

    Args:
        postgres_url: Connection URL
        min_size: Minimum connections
        max_size: Maximum connections

    Returns:
        asyncpg connection pool
    """
    import asyncpg

    logger.info(f"Creating database pool (min={min_size}, max={max_size})")
    pool = await asyncpg.create_pool(
        postgres_url,
        min_size=min_size,
        max_size=max_size,
    )
    logger.info("Database pool created")
    return pool


async def check_db_health(pool: asyncpg.Pool) -> dict:
    """
    Check database health.

    Returns:
        Health status dict
    """
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"connected": True, "pool_size": pool.get_size()}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"connected": False, "error": str(e)}
