"""
Shared fixtures for integration tests.

Integration tests run against real dependencies (PostgreSQL, Redis).
Requires: docker compose up -d
"""

import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def db_pool():
    """
    Create a database connection pool for integration tests.

    Requires PostgreSQL to be running via docker compose.
    """
    import asyncpg

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api"
    )

    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
    yield pool
    await pool.close()


@pytest_asyncio.fixture(loop_scope="session")
async def db_connection(db_pool) -> AsyncGenerator:
    """Get a database connection from the pool."""
    async with db_pool.acquire() as conn:
        # Start a transaction that will be rolled back
        transaction = conn.transaction()
        await transaction.start()
        yield conn
        await transaction.rollback()
