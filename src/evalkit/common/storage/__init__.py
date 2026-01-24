"""
Storage Abstraction Layer

Provides pluggable backends for test case and scorecard persistence.
Supports local PostgreSQL development and future Azure services migration.
"""

from src.evalkit.common.storage.config import StorageConfig
from src.evalkit.common.storage.factory import (
    close_repositories,
    create_repositories,
    get_repositories,
    repository_context,
)
from src.evalkit.common.storage.protocols import (
    BatchJobRepository,
    ScorecardRepository,
    TestCaseRepository,
)

__all__ = [
    "StorageConfig",
    "TestCaseRepository",
    "ScorecardRepository",
    "BatchJobRepository",
    "create_repositories",
    "get_repositories",
    "close_repositories",
    "repository_context",
]
