"""
PostgreSQL Storage Backend

Local development implementation using PostgreSQL + pgvector.
Provides the same interface as the Azure backend for seamless switching.
"""

from src.storage.postgres.client import create_pool, close_pool
from src.storage.postgres.test_case_repo import PostgresTestCaseRepository
from src.storage.postgres.scorecard_repo import PostgresScorecardRepository

__all__ = [
    "create_pool",
    "close_pool",
    "PostgresTestCaseRepository",
    "PostgresScorecardRepository",
]
