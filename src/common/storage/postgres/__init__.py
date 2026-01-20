"""
PostgreSQL Storage Backend

Local development implementation using PostgreSQL + pgvector.
Provides the same interface as the Azure backend for seamless switching.
"""

from src.common.storage.postgres.batch_repo import PostgresBatchJobRepository
from src.common.storage.postgres.client import close_pool, create_pool
from src.common.storage.postgres.scorecard_repo import PostgresScorecardRepository
from src.common.storage.postgres.test_case_repo import PostgresTestCaseRepository

__all__ = [
    "create_pool",
    "close_pool",
    "PostgresTestCaseRepository",
    "PostgresScorecardRepository",
    "PostgresBatchJobRepository",
]
