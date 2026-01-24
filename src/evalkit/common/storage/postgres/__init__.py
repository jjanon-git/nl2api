"""
PostgreSQL Storage Backend

Local development implementation using PostgreSQL + pgvector.
Provides the same interface as the Azure backend for seamless switching.
"""

from src.evalkit.common.storage.postgres.batch_repo import PostgresBatchJobRepository
from src.evalkit.common.storage.postgres.client import close_pool, create_pool
from src.evalkit.common.storage.postgres.entity_repo import (
    Entity,
    EntityAlias,
    EntityMatch,
    EntityStats,
    PostgresEntityRepository,
)
from src.evalkit.common.storage.postgres.scorecard_repo import PostgresScorecardRepository
from src.evalkit.common.storage.postgres.test_case_repo import PostgresTestCaseRepository

__all__ = [
    # Connection management
    "create_pool",
    "close_pool",
    # Repositories
    "PostgresTestCaseRepository",
    "PostgresScorecardRepository",
    "PostgresBatchJobRepository",
    "PostgresEntityRepository",
    # Entity data models
    "Entity",
    "EntityAlias",
    "EntityMatch",
    "EntityStats",
]
