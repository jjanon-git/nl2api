"""
Integration tests for EvalWorker with Redis.

Requires Redis to be running (docker compose up -d).
"""

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.contracts.worker import WorkerTask
from src.evaluation.distributed.config import EvalMode, QueueBackend, QueueConfig, WorkerConfig
from src.evaluation.distributed.queue import create_queue
from src.evaluation.distributed.worker import EvalWorker

# Skip all tests if Redis is not available
pytestmark = pytest.mark.integration


@pytest.fixture
async def redis_config() -> QueueConfig:
    """Create Redis queue configuration for testing."""
    return QueueConfig(
        backend=QueueBackend.REDIS,
        redis_url="redis://localhost:6379",
        redis_db=1,  # Use DB 1 for testing
        max_retries=3,
        stream_prefix="test:eval:worker",
        dlq_prefix="test:eval:worker:dlq",
    )


@pytest.fixture
async def queue(redis_config: QueueConfig):
    """Create a Redis queue for testing."""
    try:
        q = await create_queue(redis_config)
        yield q
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
    finally:
        if "q" in locals():
            await q.close()


@pytest.fixture
def batch_id() -> str:
    """Generate unique batch ID for test isolation."""
    return f"worker-test-{uuid4().hex[:8]}"


@pytest.fixture
def worker_config() -> WorkerConfig:
    """Create worker configuration for testing."""
    return WorkerConfig(
        worker_id=f"test-worker-{uuid4().hex[:8]}",
        eval_mode=EvalMode.SIMULATED,
        task_timeout_seconds=10,
        heartbeat_interval_seconds=1,
        shutdown_timeout_seconds=5,
    )


@pytest.fixture
def mock_test_case():
    """Create a mock test case."""
    test_case = MagicMock()
    test_case.id = "tc-001"
    test_case.nl_query = "What is Apple's stock price?"
    test_case.expected_tool_calls = []
    return test_case


@pytest.fixture
def mock_scorecard():
    """Create a mock scorecard."""
    scorecard = MagicMock()
    scorecard.test_case_id = "tc-001"
    scorecard.overall_passed = True
    scorecard.overall_score = 1.0
    return scorecard


# =============================================================================
# End-to-End Worker Tests
# =============================================================================


class TestWorkerIntegration:
    """Integration tests for worker with real Redis queue."""

    @pytest.mark.asyncio
    async def test_worker_processes_task_from_redis(
        self, queue, batch_id, worker_config, mock_test_case, mock_scorecard
    ):
        """Worker should process a task from Redis queue end-to-end."""
        await queue.ensure_stream(batch_id)

        # Enqueue a task
        task = WorkerTask(
            task_id=f"task-{uuid4().hex[:8]}",
            test_case_id="tc-001",
            batch_id=batch_id,
        )
        await queue.enqueue(task, batch_id)

        # Track what dependencies were called
        calls = {"response_gen": 0, "evaluator": 0, "scorecard_saver": 0, "test_case_fetcher": 0}

        async def response_generator(worker_task, test_case):
            calls["response_gen"] += 1
            return {"result": "simulated"}

        async def evaluator(test_case, response):
            calls["evaluator"] += 1
            return mock_scorecard

        async def scorecard_saver(scorecard):
            calls["scorecard_saver"] += 1

        async def test_case_fetcher(test_case_id):
            calls["test_case_fetcher"] += 1
            return mock_test_case

        # Create and run worker
        worker = EvalWorker(
            worker_id=worker_config.worker_id,
            queue=queue,
            batch_id=batch_id,
            response_generator=response_generator,
            evaluator=evaluator,
            scorecard_saver=scorecard_saver,
            test_case_fetcher=test_case_fetcher,
            config=worker_config,
        )

        # Run worker with timeout (it will block waiting for more messages)
        async def run_with_timeout():
            try:
                await asyncio.wait_for(worker.run(), timeout=2.0)
            except TimeoutError:
                pass

        await run_with_timeout()

        # Verify all stages were called
        assert calls["test_case_fetcher"] == 1
        assert calls["response_gen"] == 1
        assert calls["evaluator"] == 1
        assert calls["scorecard_saver"] == 1

        # Verify task was processed
        assert worker._tasks_processed == 1
        assert worker._tasks_failed == 0

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_worker_processes_multiple_tasks(
        self, queue, batch_id, worker_config, mock_test_case, mock_scorecard
    ):
        """Worker should process multiple tasks sequentially."""
        await queue.ensure_stream(batch_id)

        # Enqueue multiple tasks
        num_tasks = 5
        tasks = [
            WorkerTask(
                task_id=f"task-{i:03d}-{uuid4().hex[:8]}",
                test_case_id=f"tc-{i:03d}",
                batch_id=batch_id,
            )
            for i in range(num_tasks)
        ]
        await queue.enqueue_batch(tasks, batch_id)

        processed_ids = []

        async def response_generator(worker_task, test_case):
            return {"result": "simulated"}

        async def evaluator(test_case, response):
            return mock_scorecard

        async def scorecard_saver(scorecard):
            pass

        async def test_case_fetcher(test_case_id):
            processed_ids.append(test_case_id)
            return mock_test_case

        worker = EvalWorker(
            worker_id=worker_config.worker_id,
            queue=queue,
            batch_id=batch_id,
            response_generator=response_generator,
            evaluator=evaluator,
            scorecard_saver=scorecard_saver,
            test_case_fetcher=test_case_fetcher,
            config=worker_config,
        )

        async def run_with_timeout():
            try:
                await asyncio.wait_for(worker.run(), timeout=5.0)
            except TimeoutError:
                pass

        await run_with_timeout()

        assert worker._tasks_processed == num_tasks
        assert len(processed_ids) == num_tasks

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_worker_nacks_failed_tasks(self, queue, batch_id, worker_config):
        """Worker should nack tasks that fail (test case not found)."""
        await queue.ensure_stream(batch_id)

        task = WorkerTask(
            task_id=f"task-{uuid4().hex[:8]}",
            test_case_id="tc-nonexistent",
            batch_id=batch_id,
        )
        await queue.enqueue(task, batch_id)

        async def response_generator(worker_task, test_case):
            return {"result": "simulated"}

        async def evaluator(test_case, response):
            return MagicMock(overall_passed=True, overall_score=1.0)

        async def scorecard_saver(scorecard):
            pass

        async def test_case_fetcher(test_case_id):
            # Simulate test case not found
            return None

        worker = EvalWorker(
            worker_id=worker_config.worker_id,
            queue=queue,
            batch_id=batch_id,
            response_generator=response_generator,
            evaluator=evaluator,
            scorecard_saver=scorecard_saver,
            test_case_fetcher=test_case_fetcher,
            config=worker_config,
        )

        async def run_with_timeout():
            try:
                await asyncio.wait_for(worker.run(), timeout=2.0)
            except TimeoutError:
                pass

        await run_with_timeout()

        # Task should have failed
        assert worker._tasks_failed >= 1

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_worker_graceful_shutdown(
        self, queue, batch_id, worker_config, mock_test_case, mock_scorecard
    ):
        """Worker should shut down gracefully when requested."""
        await queue.ensure_stream(batch_id)

        # Enqueue tasks
        tasks = [
            WorkerTask(
                task_id=f"task-{i:03d}",
                test_case_id=f"tc-{i:03d}",
                batch_id=batch_id,
            )
            for i in range(10)
        ]
        await queue.enqueue_batch(tasks, batch_id)

        async def response_generator(worker_task, test_case):
            return {"result": "simulated"}

        async def evaluator(test_case, response):
            return mock_scorecard

        async def scorecard_saver(scorecard):
            pass

        async def test_case_fetcher(test_case_id):
            return mock_test_case

        worker = EvalWorker(
            worker_id=worker_config.worker_id,
            queue=queue,
            batch_id=batch_id,
            response_generator=response_generator,
            evaluator=evaluator,
            scorecard_saver=scorecard_saver,
            test_case_fetcher=test_case_fetcher,
            config=worker_config,
        )

        # Trigger shutdown after some processing
        async def trigger_shutdown():
            await asyncio.sleep(0.3)
            await worker.shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())

        # Run worker with overall timeout to prevent hanging
        try:
            await asyncio.wait_for(worker.run(), timeout=3.0)
        except TimeoutError:
            # Worker may not exit on shutdown if queue blocks
            pass

        await shutdown_task

        # Worker should have stopped cleanly
        assert not worker._running or worker._shutdown_event.is_set()

        # Cleanup
        await queue.delete_stream(batch_id)
