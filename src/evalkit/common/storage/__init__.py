"""
Storage Abstraction Layer

Provides pluggable backends for test case and scorecard persistence.
Supports local PostgreSQL development and future Azure services migration.

Recommended usage with RepositoryProvider (dependency injection):

    from src.evalkit.common.storage import RepositoryProvider, StorageConfig

    async with RepositoryProvider(StorageConfig(backend="memory")) as provider:
        test_case = await provider.test_cases.get("test-id")
        await provider.scorecards.create(scorecard)

Legacy usage with module-level functions (backward compatible):

    from src.evalkit.common.storage import create_repositories, close_repositories

    repos = await create_repositories()
    test_case_repo, scorecard_repo, batch_repo = repos
    # ... use repos ...
    await close_repositories()
"""

from src.evalkit.common.storage.config import StorageConfig
from src.evalkit.common.storage.factory import (
    RepositoryProvider,
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
    # Config
    "StorageConfig",
    # Protocols
    "TestCaseRepository",
    "ScorecardRepository",
    "BatchJobRepository",
    # Provider (recommended)
    "RepositoryProvider",
    # Legacy factory functions (backward compatible)
    "create_repositories",
    "get_repositories",
    "close_repositories",
    "repository_context",
]
