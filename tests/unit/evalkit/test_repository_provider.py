"""
Unit tests for RepositoryProvider class.

Tests the dependency injection pattern and lifecycle management.
"""

from __future__ import annotations

import pytest

from src.evalkit.common.storage import (
    RepositoryProvider,
    StorageConfig,
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


class TestRepositoryProviderLifecycle:
    """Tests for RepositoryProvider initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_provider_as_context_manager(self):
        """Test using RepositoryProvider as async context manager."""
        config = StorageConfig(backend="memory")

        async with RepositoryProvider(config) as provider:
            assert provider.is_initialized
            assert provider.test_cases is not None
            assert provider.scorecards is not None
            assert provider.batch_jobs is not None

        # After exiting context, provider should be closed
        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_provider_manual_lifecycle(self):
        """Test manual initialize/close lifecycle."""
        config = StorageConfig(backend="memory")
        provider = RepositoryProvider(config)

        assert not provider.is_initialized

        await provider.initialize()
        assert provider.is_initialized

        await provider.close()
        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_provider_close_is_idempotent(self):
        """Test that close() can be called multiple times safely."""
        config = StorageConfig(backend="memory")
        provider = RepositoryProvider(config)

        await provider.initialize()
        await provider.close()
        await provider.close()  # Should not raise
        await provider.close()  # Should not raise

        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_provider_reinitialization_logs_warning(self):
        """Test that reinitializing logs a warning but doesn't fail."""
        config = StorageConfig(backend="memory")
        provider = RepositoryProvider(config)

        await provider.initialize()
        # This should log a warning but not fail
        await provider.initialize()

        assert provider.is_initialized
        await provider.close()

    @pytest.mark.asyncio
    async def test_provider_uninitialized_access_raises(self):
        """Test that accessing repos before initialization raises RuntimeError."""
        config = StorageConfig(backend="memory")
        provider = RepositoryProvider(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = provider.test_cases

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = provider.scorecards

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = provider.batch_jobs

        with pytest.raises(RuntimeError, match="not initialized"):
            provider.get_repositories()


class TestRepositoryProviderBackends:
    """Tests for different storage backends."""

    @pytest.mark.asyncio
    async def test_memory_backend(self):
        """Test memory backend initialization."""
        config = StorageConfig(backend="memory")

        async with RepositoryProvider(config) as provider:
            assert isinstance(provider.test_cases, TestCaseRepository)
            assert isinstance(provider.scorecards, ScorecardRepository)
            assert isinstance(provider.batch_jobs, BatchJobRepository)

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self):
        """Test that unknown backend is rejected by config validation."""
        import pydantic

        # StorageConfig validates backend at creation time
        with pytest.raises(pydantic.ValidationError, match="Input should be"):
            StorageConfig(backend="unknown")

    @pytest.mark.asyncio
    async def test_azure_backend_not_implemented(self):
        """Test that azure backend raises NotImplementedError."""
        config = StorageConfig(backend="azure")
        provider = RepositoryProvider(config)

        with pytest.raises(NotImplementedError, match="Azure backend not yet implemented"):
            await provider.initialize()


class TestRepositoryProviderConcurrency:
    """Tests for concurrent provider usage."""

    @pytest.mark.asyncio
    async def test_multiple_providers_independent(self):
        """Test that multiple providers are independent."""
        config = StorageConfig(backend="memory")

        async with RepositoryProvider(config) as provider1:
            async with RepositoryProvider(config) as provider2:
                # Both should be initialized independently
                assert provider1.is_initialized
                assert provider2.is_initialized

                # They should have different repository instances
                assert provider1.test_cases is not provider2.test_cases
                assert provider1.scorecards is not provider2.scorecards
                assert provider1.batch_jobs is not provider2.batch_jobs

            # Provider2 closed, but provider1 still active
            assert provider1.is_initialized
            assert not provider2.is_initialized

    @pytest.mark.asyncio
    async def test_get_repositories_returns_tuple(self):
        """Test get_repositories returns correct tuple."""
        config = StorageConfig(backend="memory")

        async with RepositoryProvider(config) as provider:
            test_case_repo, scorecard_repo, batch_repo = provider.get_repositories()

            assert test_case_repo is provider.test_cases
            assert scorecard_repo is provider.scorecards
            assert batch_repo is provider.batch_jobs


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy factory functions."""

    @pytest.mark.asyncio
    async def test_create_and_close_repositories(self):
        """Test legacy create_repositories and close_repositories."""
        config = StorageConfig(backend="memory")

        repos = await create_repositories(config)
        test_case_repo, scorecard_repo, batch_repo = repos

        assert test_case_repo is not None
        assert scorecard_repo is not None
        assert batch_repo is not None

        await close_repositories()

    @pytest.mark.asyncio
    async def test_get_repositories_after_create(self):
        """Test get_repositories returns same instances."""
        config = StorageConfig(backend="memory")

        await create_repositories(config)
        repos = get_repositories()

        assert len(repos) == 3
        assert all(r is not None for r in repos)

        await close_repositories()

    @pytest.mark.asyncio
    async def test_get_repositories_before_create_raises(self):
        """Test get_repositories before create raises RuntimeError."""
        # Make sure no global state from previous tests
        await close_repositories()

        with pytest.raises(RuntimeError, match="not initialized"):
            get_repositories()

    @pytest.mark.asyncio
    async def test_repository_context_isolates_state(self):
        """Test repository_context uses isolated provider."""
        config = StorageConfig(backend="memory")

        # Create global repositories
        await create_repositories(config)
        global_repos = get_repositories()

        # Use context - should create new isolated provider
        async with repository_context(config) as context_repos:
            # Context repos should be different instances
            assert context_repos[0] is not global_repos[0]
            assert context_repos[1] is not global_repos[1]
            assert context_repos[2] is not global_repos[2]

        # Global repos should still work
        assert get_repositories() == global_repos

        await close_repositories()

    @pytest.mark.asyncio
    async def test_create_repositories_twice_returns_same(self):
        """Test that create_repositories twice returns same repos (with warning)."""
        config = StorageConfig(backend="memory")

        repos1 = await create_repositories(config)
        repos2 = await create_repositories(config)  # Should log warning

        # Should return same instances
        assert repos1 == repos2

        await close_repositories()

    @pytest.mark.asyncio
    async def test_close_repositories_is_idempotent(self):
        """Test that close_repositories can be called multiple times."""
        config = StorageConfig(backend="memory")

        await create_repositories(config)
        await close_repositories()
        await close_repositories()  # Should not raise
        await close_repositories()  # Should not raise


class TestRepositoryProviderOperations:
    """Tests for basic repository operations through provider."""

    @pytest.mark.asyncio
    async def test_test_case_operations(self):
        """Test basic test case operations through provider."""
        from src.evalkit.contracts.core import TestCase

        config = StorageConfig(backend="memory")

        async with RepositoryProvider(config) as provider:
            # Create a test case
            test_case = TestCase(
                id="test-001",
                input={"query": "What is Apple's stock price?"},
                expected={"tool": "datastream"},
                tags=("test",),
            )

            # Use save() - the protocol method for storing test cases
            await provider.test_cases.save(test_case)

            # Retrieve it
            retrieved = await provider.test_cases.get("test-001")
            assert retrieved is not None
            assert retrieved.id == "test-001"

    @pytest.mark.asyncio
    async def test_batch_job_operations(self):
        """Test basic batch job operations through provider."""
        from datetime import UTC, datetime

        from src.evalkit.contracts.core import TaskStatus
        from src.evalkit.contracts.worker import BatchJob

        config = StorageConfig(backend="memory")

        async with RepositoryProvider(config) as provider:
            # Create a batch job
            batch_job = BatchJob(
                total_tests=100,
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.now(UTC),
            )

            await provider.batch_jobs.create(batch_job)

            # Retrieve it
            retrieved = await provider.batch_jobs.get(batch_job.batch_id)
            assert retrieved is not None
            assert retrieved.total_tests == 100
