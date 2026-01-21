"""
Shared fixtures for integration tests.

Integration tests run against real dependencies (PostgreSQL, Redis).
Requires: docker compose up -d
"""

import pytest
import asyncio
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_pool():
    """
    Create a database connection pool for integration tests.

    Requires PostgreSQL to be running via docker compose.
    """
    import asyncpg
    import os

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/nl2api_test"
    )

    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
    yield pool
    await pool.close()


@pytest.fixture
async def db_connection(db_pool) -> AsyncGenerator:
    """Get a database connection from the pool."""
    async with db_pool.acquire() as conn:
        # Start a transaction that will be rolled back
        transaction = conn.transaction()
        await transaction.start()
        yield conn
        await transaction.rollback()
