"""
Integration tests for distributed batch evaluation.

Tests the full distributed batch workflow:
- Enqueuing tasks to Redis
- Spawning worker subprocesses
- Processing tasks in parallel
- Saving scorecards to database

Requires: docker compose up -d (PostgreSQL + Redis)
"""

import uuid

import pytest
import pytest_asyncio

from src.evalkit.common.storage.postgres import (
    PostgresBatchJobRepository,
    PostgresScorecardRepository,
    PostgresTestCaseRepository,
)
from src.evalkit.contracts.core import TaskStatus, TestCase, TestCaseMetadata, ToolCall
from src.evalkit.contracts.worker import BatchJob
from src.evalkit.distributed.config import (
    CoordinatorConfig,
    QueueBackend,
    QueueConfig,
)
from src.evalkit.distributed.coordinator import BatchCoordinator
from src.evalkit.distributed.manager import LocalWorkerManager
from src.evalkit.distributed.queue import create_queue

# Skip all tests if dependencies not available
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture(loop_scope="session")
async def test_case_repo(db_pool):
    """Create a test case repository."""
    return PostgresTestCaseRepository(db_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def scorecard_repo(db_pool):
    """Create a scorecard repository."""
    return PostgresScorecardRepository(db_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def batch_repo(db_pool):
    """Create a batch job repository."""
    return PostgresBatchJobRepository(db_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def redis_queue():
    """Create a Redis queue for testing.

    Uses default prefixes so workers can find tasks.
    Workers use hardcoded defaults in __main__.py.
    """
    config = QueueConfig(
        backend=QueueBackend.REDIS,
        redis_url="redis://localhost:6379",
        # Use default prefixes to match what workers expect
        # stream_prefix defaults to "eval:tasks"
        # dlq_prefix defaults to "eval:dlq"
    )
    try:
        queue = await create_queue(config)
        yield queue
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
    finally:
        if "queue" in locals():
            await queue.close()


def create_test_case(index: int, batch_prefix: str) -> TestCase:
    """Create a test case for testing."""
    return TestCase(
        id=str(uuid.uuid4()),  # Must be valid UUID
        nl_query=f"Test query {index}",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["TEST.O"], "fields": ["TR.Revenue"]},
            ),
        ),
        expected_nl_response=f"Test response {index}",
        tags=("test", "distributed"),
        category="test",
        subcategory="distributed",
        metadata=TestCaseMetadata(
            source="integration_test",
            api_version="1.0",
            complexity_level=1,
        ),
    )


@pytest.mark.asyncio(loop_scope="session")
class TestDistributedBatchWorkflow:
    """End-to-end tests for distributed batch evaluation."""

    async def test_distributed_batch_simulated_mode(
        self,
        db_pool,
        test_case_repo,
        scorecard_repo,
        batch_repo,
        redis_queue,
    ):
        """
        Test full distributed batch workflow with simulated responses.

        This test:
        1. Creates test cases in the database
        2. Creates a batch job
        3. Uses BatchCoordinator to enqueue tasks
        4. Starts workers via LocalWorkerManager
        5. Waits for completion
        6. Verifies scorecards were saved
        """
        batch_prefix = f"dist-test-{uuid.uuid4().hex[:8]}"
        num_test_cases = 3

        # Create test cases
        test_cases = [create_test_case(i, batch_prefix) for i in range(num_test_cases)]
        for tc in test_cases:
            await test_case_repo.save(tc)

        # Create batch job
        batch_job = BatchJob(
            total_tests=num_test_cases,
            status=TaskStatus.IN_PROGRESS,
            tags=("test", "distributed"),
        )
        await batch_repo.create(batch_job)
        batch_id = batch_job.batch_id

        # Create coordinator
        coordinator_config = CoordinatorConfig(
            progress_poll_interval_seconds=0.5,
            batch_timeout_seconds=60,
        )
        coordinator = BatchCoordinator(
            queue=redis_queue,
            batch_repo=batch_repo,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            config=coordinator_config,
        )

        manager = None
        try:
            # Enqueue tasks
            enqueued = await coordinator.start_batch(test_cases, batch_id, eval_mode="simulated")
            assert enqueued == num_test_cases

            # Start workers
            manager = LocalWorkerManager(
                worker_count=2,
                redis_url="redis://localhost:6379",
                eval_mode="simulated",
                max_retries=3,
                verbose=False,
            )
            manager.start(batch_id)

            # Wait for completion
            result = await coordinator.wait_for_completion(
                batch_id=batch_id,
                total_tasks=num_test_cases,
            )

            # Verify results
            assert result.completed is True
            assert result.total == num_test_cases
            assert result.passed == num_test_cases  # Simulated mode = 100% pass
            assert result.failed == 0
            assert result.in_dlq == 0

            # Verify scorecards in database
            summary = await scorecard_repo.get_batch_summary(batch_id)
            assert summary["passed"] == num_test_cases
            assert summary["failed"] == 0

            # Clean up queue resources
            await coordinator.cleanup(batch_id)

        finally:
            # Stop workers
            if manager:
                manager.stop(timeout=10)

            # Clean up test data
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM scorecards WHERE batch_id = $1", batch_id)
                await conn.execute("DELETE FROM batch_jobs WHERE id = $1", batch_id)
                for tc in test_cases:
                    await conn.execute("DELETE FROM test_cases WHERE id = $1", tc.id)

    async def test_coordinator_progress_tracking(
        self,
        db_pool,
        test_case_repo,
        scorecard_repo,
        batch_repo,
        redis_queue,
    ):
        """Test that BatchCoordinator correctly tracks progress."""
        batch_prefix = f"prog-test-{uuid.uuid4().hex[:8]}"
        num_test_cases = 5

        # Create test cases
        test_cases = [create_test_case(i, batch_prefix) for i in range(num_test_cases)]
        for tc in test_cases:
            await test_case_repo.save(tc)

        # Create batch job
        batch_job = BatchJob(
            total_tests=num_test_cases,
            status=TaskStatus.IN_PROGRESS,
            tags=("test",),
        )
        await batch_repo.create(batch_job)
        batch_id = batch_job.batch_id

        coordinator = BatchCoordinator(
            queue=redis_queue,
            batch_repo=batch_repo,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
        )

        manager = None
        progress_updates = []

        try:
            await coordinator.start_batch(test_cases, batch_id, eval_mode="simulated")

            manager = LocalWorkerManager(
                worker_count=2,
                redis_url="redis://localhost:6379",
                eval_mode="simulated",
            )
            manager.start(batch_id)

            def on_progress(completed: int, total: int):
                progress_updates.append((completed, total))

            result = await coordinator.wait_for_completion(
                batch_id=batch_id,
                total_tasks=num_test_cases,
                on_progress=on_progress,
            )

            # Verify progress was tracked
            assert len(progress_updates) > 0
            # Final progress should be all complete
            assert progress_updates[-1] == (num_test_cases, num_test_cases)
            assert result.completed is True

            await coordinator.cleanup(batch_id)

        finally:
            if manager:
                manager.stop(timeout=10)

            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM scorecards WHERE batch_id = $1", batch_id)
                await conn.execute("DELETE FROM batch_jobs WHERE id = $1", batch_id)
                for tc in test_cases:
                    await conn.execute("DELETE FROM test_cases WHERE id = $1", tc.id)

    async def test_manager_spawns_correct_workers(self):
        """Test that LocalWorkerManager spawns the right number of workers."""
        batch_id = f"manager-test-{uuid.uuid4().hex[:8]}"

        manager = LocalWorkerManager(
            worker_count=3,
            redis_url="redis://localhost:6379",
            eval_mode="simulated",
        )

        try:
            manager.start(batch_id)

            # Verify workers are running
            assert manager.get_running_count() == 3
            assert len(manager.get_worker_pids()) == 3
            assert manager.is_healthy() is True

        finally:
            manager.stop(timeout=10)

            # Verify workers stopped
            assert manager.get_running_count() == 0


@pytest.mark.asyncio(loop_scope="session")
class TestDistributedBatchEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_empty_batch(
        self,
        scorecard_repo,
        batch_repo,
        test_case_repo,
        redis_queue,
    ):
        """Test handling of empty batch."""
        coordinator = BatchCoordinator(
            queue=redis_queue,
            batch_repo=batch_repo,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
        )

        batch_id = f"empty-{uuid.uuid4().hex[:8]}"
        enqueued = await coordinator.start_batch([], batch_id)

        assert enqueued == 0

    async def test_coordinator_timeout(
        self,
        db_pool,
        test_case_repo,
        scorecard_repo,
        batch_repo,
        redis_queue,
    ):
        """Test that coordinator times out if workers don't finish."""
        batch_prefix = f"timeout-{uuid.uuid4().hex[:8]}"

        # Create a single test case
        tc = create_test_case(0, batch_prefix)
        await test_case_repo.save(tc)

        batch_job = BatchJob(
            total_tests=1,
            status=TaskStatus.IN_PROGRESS,
        )
        await batch_repo.create(batch_job)
        batch_id = batch_job.batch_id

        coordinator_config = CoordinatorConfig(
            progress_poll_interval_seconds=0.1,
            batch_timeout_seconds=1,  # Very short timeout
        )
        coordinator = BatchCoordinator(
            queue=redis_queue,
            batch_repo=batch_repo,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            config=coordinator_config,
        )

        try:
            await coordinator.start_batch([tc], batch_id)

            # Don't start workers - should timeout
            result = await coordinator.wait_for_completion(
                batch_id=batch_id,
                total_tasks=1,
            )

            # Should have timed out without completion
            assert result.completed is False
            assert result.total == 1
            assert result.passed == 0

            await coordinator.cleanup(batch_id)

        finally:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM scorecards WHERE batch_id = $1", batch_id)
                await conn.execute("DELETE FROM batch_jobs WHERE id = $1", batch_id)
                await conn.execute("DELETE FROM test_cases WHERE id = $1", tc.id)
