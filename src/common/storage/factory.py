"""
Repository Factory

Factory functions to create repository instances based on configuration.
Supports PostgreSQL (local), Azure (production), and Memory (tests) backends.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from src.common.storage.config import StorageConfig
from src.common.storage.protocols import BatchJobRepository, ScorecardRepository, TestCaseRepository

if TYPE_CHECKING:
    import asyncpg


# Module-level state for singleton pattern
_test_case_repo: TestCaseRepository | None = None
_scorecard_repo: ScorecardRepository | None = None
_batch_repo: BatchJobRepository | None = None
_pool: asyncpg.Pool | None = None


async def create_repositories(
    config: StorageConfig | None = None,
) -> tuple[TestCaseRepository, ScorecardRepository, BatchJobRepository]:
    """
    Factory function to create repository instances based on config.

    Args:
        config: Storage configuration. If None, reads from environment.

    Returns:
        Tuple of (TestCaseRepository, ScorecardRepository, BatchJobRepository)

    Raises:
        NotImplementedError: If Azure backend is requested (not yet implemented)
        ValueError: If unknown backend is specified
    """
    global _test_case_repo, _scorecard_repo, _batch_repo, _pool

    config = config or StorageConfig()

    if config.backend == "postgres":
        from src.common.storage.postgres import (
            PostgresBatchJobRepository,
            PostgresScorecardRepository,
            PostgresTestCaseRepository,
            create_pool,
        )

        _pool = await create_pool(
            config.postgres_url,
            min_size=config.postgres_pool_min,
            max_size=config.postgres_pool_max,
        )
        _test_case_repo = PostgresTestCaseRepository(_pool)
        _scorecard_repo = PostgresScorecardRepository(_pool)
        _batch_repo = PostgresBatchJobRepository(_pool)

    elif config.backend == "memory":
        from src.common.storage.memory import (
            InMemoryBatchJobRepository,
            InMemoryScorecardRepository,
            InMemoryTestCaseRepository,
        )

        _test_case_repo = InMemoryTestCaseRepository()
        _scorecard_repo = InMemoryScorecardRepository()
        _batch_repo = InMemoryBatchJobRepository()

    elif config.backend == "azure":
        raise NotImplementedError(
            "Azure backend not yet implemented. "
            "Use 'postgres' for local development or 'memory' for tests."
        )

    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")

    return _test_case_repo, _scorecard_repo, _batch_repo


async def close_repositories() -> None:
    """Close repository connections and cleanup resources."""
    global _test_case_repo, _scorecard_repo, _batch_repo, _pool

    if _pool is not None:
        from src.common.storage.postgres import close_pool

        await close_pool()
        _pool = None

    _test_case_repo = None
    _scorecard_repo = None
    _batch_repo = None


def get_repositories() -> tuple[TestCaseRepository, ScorecardRepository, BatchJobRepository]:
    """
    Get existing repository instances.

    Raises:
        RuntimeError: If repositories haven't been created yet.
    """
    if _test_case_repo is None or _scorecard_repo is None or _batch_repo is None:
        raise RuntimeError("Repositories not initialized. Call create_repositories() first.")
    return _test_case_repo, _scorecard_repo, _batch_repo


@asynccontextmanager
async def repository_context(config: StorageConfig | None = None):
    """
    Context manager for repository lifecycle management.

    Usage:
        async with repository_context() as (test_case_repo, scorecard_repo):
            test_case = await test_case_repo.get("test-id")
            ...

    Args:
        config: Storage configuration. If None, reads from environment.

    Yields:
        Tuple of (TestCaseRepository, ScorecardRepository)
    """
    repos = await create_repositories(config)
    try:
        yield repos
    finally:
        await close_repositories()
