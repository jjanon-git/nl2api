"""
Repository Factory

Factory functions and RepositoryProvider class for creating repository instances.
Supports PostgreSQL (local), Azure (production), and Memory (tests) backends.

The preferred pattern is to use RepositoryProvider with dependency injection:

    async with RepositoryProvider(config) as provider:
        test_case = await provider.test_cases.get("test-id")
        await provider.scorecards.create(scorecard)

For backward compatibility, the module-level factory functions are still available:

    repos = await create_repositories(config)
    test_case_repo, scorecard_repo, batch_repo = repos
    # ... use repos ...
    await close_repositories()
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from src.evalkit.common.storage.config import StorageConfig
from src.evalkit.common.storage.protocols import (
    BatchJobRepository,
    ScorecardRepository,
    TestCaseRepository,
)
from src.evalkit.exceptions import StorageConnectionError

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


class RepositoryProvider:
    """
    Repository provider with proper dependency injection and lifecycle management.

    This class replaces the global singleton pattern with a proper instance-based
    approach that supports:
    - Multiple concurrent providers (for multi-tenancy or testing)
    - Clean lifecycle management via async context manager
    - Explicit dependency injection

    Usage:
        # As context manager (recommended)
        async with RepositoryProvider(config) as provider:
            test_case = await provider.test_cases.get("test-id")
            await provider.scorecards.create(scorecard)

        # Manual lifecycle management
        provider = RepositoryProvider(config)
        await provider.initialize()
        try:
            test_case = await provider.test_cases.get("test-id")
        finally:
            await provider.close()

    Attributes:
        test_cases: Repository for test case operations
        scorecards: Repository for scorecard operations
        batch_jobs: Repository for batch job operations
    """

    def __init__(self, config: StorageConfig | None = None) -> None:
        """
        Initialize the repository provider.

        Args:
            config: Storage configuration. If None, reads from environment.
        """
        self._config = config or StorageConfig()
        self._pool: asyncpg.Pool | None = None
        self._test_case_repo: TestCaseRepository | None = None
        self._scorecard_repo: ScorecardRepository | None = None
        self._batch_repo: BatchJobRepository | None = None
        self._initialized = False

    @property
    def test_cases(self) -> TestCaseRepository:
        """Get the test case repository."""
        if self._test_case_repo is None:
            raise RuntimeError(
                "Provider not initialized. Call initialize() or use as context manager."
            )
        return self._test_case_repo

    @property
    def scorecards(self) -> ScorecardRepository:
        """Get the scorecard repository."""
        if self._scorecard_repo is None:
            raise RuntimeError(
                "Provider not initialized. Call initialize() or use as context manager."
            )
        return self._scorecard_repo

    @property
    def batch_jobs(self) -> BatchJobRepository:
        """Get the batch job repository."""
        if self._batch_repo is None:
            raise RuntimeError(
                "Provider not initialized. Call initialize() or use as context manager."
            )
        return self._batch_repo

    @property
    def is_initialized(self) -> bool:
        """Check if the provider has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """
        Initialize repository connections.

        Creates the database pool (if applicable) and instantiates repositories.

        Raises:
            NotImplementedError: If Azure backend is requested (not yet implemented)
            ValueError: If unknown backend is specified
        """
        if self._initialized:
            logger.warning("Provider already initialized, skipping re-initialization")
            return

        if self._config.backend == "postgres":
            from src.evalkit.common.storage.postgres import (
                PostgresBatchJobRepository,
                PostgresScorecardRepository,
                PostgresTestCaseRepository,
                create_pool,
            )

            try:
                self._pool = await create_pool(
                    self._config.postgres_url,
                    min_size=self._config.postgres_pool_min,
                    max_size=self._config.postgres_pool_max,
                )

                # Validate connection by running a simple query
                async with self._pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                logger.debug("PostgreSQL connection validated successfully")

            except Exception as e:
                # Clean up pool if validation fails
                if self._pool is not None:
                    await self._pool.close()
                    self._pool = None
                raise StorageConnectionError(
                    "postgres", f"Failed to connect to {self._config.postgres_url}: {e}"
                ) from e

            self._test_case_repo = PostgresTestCaseRepository(self._pool)
            self._scorecard_repo = PostgresScorecardRepository(self._pool)
            self._batch_repo = PostgresBatchJobRepository(self._pool)

        elif self._config.backend == "memory":
            from src.evalkit.common.storage.memory import (
                InMemoryBatchJobRepository,
                InMemoryScorecardRepository,
                InMemoryTestCaseRepository,
            )

            self._test_case_repo = InMemoryTestCaseRepository()
            self._scorecard_repo = InMemoryScorecardRepository()
            self._batch_repo = InMemoryBatchJobRepository()

        elif self._config.backend == "azure":
            raise NotImplementedError(
                "Azure backend not yet implemented. "
                "Use 'postgres' for local development or 'memory' for tests."
            )

        else:
            raise ValueError(f"Unknown storage backend: {self._config.backend}")

        self._initialized = True
        logger.debug(f"RepositoryProvider initialized with backend: {self._config.backend}")

    async def close(self) -> None:
        """
        Close repository connections and cleanup resources.

        Safe to call multiple times - subsequent calls are no-ops.
        """
        if not self._initialized:
            return

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

            # Also reset the module-level pool singleton in client.py
            # This ensures subsequent create_pool() calls create a fresh pool
            if self._config.backend == "postgres":
                from src.evalkit.common.storage.postgres import close_pool
                await close_pool()

        self._test_case_repo = None
        self._scorecard_repo = None
        self._batch_repo = None
        self._initialized = False
        logger.debug("RepositoryProvider closed")

    def get_repositories(
        self,
    ) -> tuple[TestCaseRepository, ScorecardRepository, BatchJobRepository]:
        """
        Get all repositories as a tuple.

        Returns:
            Tuple of (TestCaseRepository, ScorecardRepository, BatchJobRepository)

        Raises:
            RuntimeError: If provider hasn't been initialized yet.
        """
        return self.test_cases, self.scorecards, self.batch_jobs

    async def __aenter__(self) -> RepositoryProvider:
        """Enter async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()


# =============================================================================
# Module-level state for backward compatibility
# =============================================================================

# Global provider instance for backward compatibility with legacy code
_global_provider: RepositoryProvider | None = None


async def create_repositories(
    config: StorageConfig | None = None,
) -> tuple[TestCaseRepository, ScorecardRepository, BatchJobRepository]:
    """
    Factory function to create repository instances based on config.

    Note: This function uses a global singleton for backward compatibility.
    For new code, prefer using RepositoryProvider directly:

        async with RepositoryProvider(config) as provider:
            # use provider.test_cases, provider.scorecards, etc.

    Args:
        config: Storage configuration. If None, reads from environment.

    Returns:
        Tuple of (TestCaseRepository, ScorecardRepository, BatchJobRepository)

    Raises:
        NotImplementedError: If Azure backend is requested (not yet implemented)
        ValueError: If unknown backend is specified
    """
    global _global_provider

    if _global_provider is not None and _global_provider.is_initialized:
        logger.warning(
            "Repositories already initialized. Call close_repositories() first to reinitialize."
        )
        return _global_provider.get_repositories()

    _global_provider = RepositoryProvider(config)
    await _global_provider.initialize()
    return _global_provider.get_repositories()


async def close_repositories() -> None:
    """
    Close repository connections and cleanup resources.

    Note: This function manages the global singleton for backward compatibility.
    For new code, prefer using RepositoryProvider as a context manager.
    """
    global _global_provider

    if _global_provider is not None:
        await _global_provider.close()
        _global_provider = None


def get_repositories() -> tuple[TestCaseRepository, ScorecardRepository, BatchJobRepository]:
    """
    Get existing repository instances.

    Note: This function uses a global singleton for backward compatibility.
    For new code, prefer using RepositoryProvider directly.

    Raises:
        RuntimeError: If repositories haven't been created yet.
    """
    if _global_provider is None or not _global_provider.is_initialized:
        raise RuntimeError("Repositories not initialized. Call create_repositories() first.")
    return _global_provider.get_repositories()


@asynccontextmanager
async def repository_context(config: StorageConfig | None = None):
    """
    Context manager for repository lifecycle management.

    Note: This uses its own RepositoryProvider instance, not the global one.
    This is the preferred pattern for isolated operations.

    Usage:
        async with repository_context() as (test_case_repo, scorecard_repo, batch_repo):
            test_case = await test_case_repo.get("test-id")
            ...

    Args:
        config: Storage configuration. If None, reads from environment.

    Yields:
        Tuple of (TestCaseRepository, ScorecardRepository, BatchJobRepository)
    """
    async with RepositoryProvider(config) as provider:
        yield provider.get_repositories()
