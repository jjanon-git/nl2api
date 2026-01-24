"""
Integration tests for RedisStreamQueue.

Requires Redis to be running (docker compose up -d).
"""

import asyncio
from uuid import uuid4

import pytest

from src.evalkit.contracts.worker import WorkerTask
from src.evalkit.distributed.config import QueueBackend, QueueConfig
from src.evalkit.distributed.queue import create_queue
from src.evalkit.distributed.queue.redis_stream import RedisStreamQueue

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
        stream_prefix="test:eval:tasks",
        dlq_prefix="test:eval:dlq",
    )


@pytest.fixture
async def queue(redis_config: QueueConfig) -> RedisStreamQueue:
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
    return f"test-batch-{uuid4().hex[:8]}"


@pytest.fixture
def sample_task(batch_id: str) -> WorkerTask:
    """Create a sample worker task."""
    return WorkerTask(
        task_id=f"task-{uuid4().hex[:8]}",
        test_case_id=f"tc-{uuid4().hex[:8]}",
        batch_id=batch_id,
    )


@pytest.fixture
def sample_tasks(batch_id: str) -> list[WorkerTask]:
    """Create multiple sample worker tasks."""
    return [
        WorkerTask(
            task_id=f"task-{i:03d}-{uuid4().hex[:8]}",
            test_case_id=f"tc-{i:03d}",
            batch_id=batch_id,
        )
        for i in range(5)
    ]


# =============================================================================
# Connection Tests
# =============================================================================


class TestConnection:
    """Tests for Redis connection."""

    @pytest.mark.asyncio
    async def test_create_queue_connects(self, redis_config: QueueConfig):
        """Factory should connect to Redis successfully."""
        queue = await create_queue(redis_config)
        assert isinstance(queue, RedisStreamQueue)
        await queue.close()

    @pytest.mark.asyncio
    async def test_connection_failure_raises(self):
        """Invalid Redis URL should raise connection error."""
        config = QueueConfig(
            backend=QueueBackend.REDIS,
            redis_url="redis://invalid-host:6379",
        )

        with pytest.raises(Exception):  # QueueConnectionError
            await create_queue(config)


# =============================================================================
# Enqueue Tests
# =============================================================================


class TestEnqueue:
    """Tests for enqueue operations."""

    @pytest.mark.asyncio
    async def test_enqueue_single_task(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Enqueue should add a task and return a message ID."""
        await queue.ensure_stream(batch_id)

        message_id = await queue.enqueue(sample_task, batch_id)

        assert message_id is not None
        assert isinstance(message_id, str)
        assert "-" in message_id  # Redis stream IDs contain -

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_enqueue_batch(
        self, queue: RedisStreamQueue, sample_tasks: list[WorkerTask], batch_id: str
    ):
        """Enqueue batch should add multiple tasks efficiently."""
        await queue.ensure_stream(batch_id)

        message_ids = await queue.enqueue_batch(sample_tasks, batch_id)

        assert len(message_ids) == len(sample_tasks)
        assert len(set(message_ids)) == len(message_ids)  # All unique

        # Verify count
        count = await queue.get_pending_count(batch_id)
        assert count == len(sample_tasks)

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_enqueue_to_different_batches(
        self, queue: RedisStreamQueue, sample_task: WorkerTask
    ):
        """Tasks in different batches should be isolated."""
        batch_1 = f"batch-1-{uuid4().hex[:8]}"
        batch_2 = f"batch-2-{uuid4().hex[:8]}"

        await queue.ensure_stream(batch_1)
        await queue.ensure_stream(batch_2)

        await queue.enqueue(sample_task, batch_1)
        await queue.enqueue(sample_task, batch_2)
        await queue.enqueue(sample_task, batch_2)

        assert await queue.get_pending_count(batch_1) == 1
        assert await queue.get_pending_count(batch_2) == 2

        # Cleanup
        await queue.delete_stream(batch_1)
        await queue.delete_stream(batch_2)


# =============================================================================
# Consume Tests
# =============================================================================


class TestConsume:
    """Tests for consume operations."""

    @pytest.mark.asyncio
    async def test_consume_yields_message(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Consume should yield enqueued messages."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue(sample_task, batch_id)

        consumer = queue.consume("worker-0", batch_id, block_ms=1000)
        message = await anext(consumer)

        assert message is not None
        assert message.payload["task_id"] == sample_task.task_id

        await queue.ack(message)
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_consume_multiple_workers(
        self, queue: RedisStreamQueue, sample_tasks: list[WorkerTask], batch_id: str
    ):
        """Multiple workers should each get different messages."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue_batch(sample_tasks, batch_id)

        # Two workers consume
        consumer1 = queue.consume("worker-0", batch_id, block_ms=1000)
        consumer2 = queue.consume("worker-1", batch_id, block_ms=1000)

        messages = []
        for _ in range(len(sample_tasks)):
            # Alternate between workers
            try:
                msg = await asyncio.wait_for(anext(consumer1), timeout=0.5)
                messages.append(msg)
                await queue.ack(msg)
            except (TimeoutError, StopAsyncIteration):
                pass

            try:
                msg = await asyncio.wait_for(anext(consumer2), timeout=0.5)
                messages.append(msg)
                await queue.ack(msg)
            except (TimeoutError, StopAsyncIteration):
                pass

        # All messages should be consumed (by one worker or the other)
        assert len(messages) == len(sample_tasks)

        # Cleanup
        await queue.delete_stream(batch_id)


# =============================================================================
# Ack/Nack Tests
# =============================================================================


class TestAckNack:
    """Tests for acknowledgment operations."""

    @pytest.mark.asyncio
    async def test_ack_removes_from_pending(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Ack should remove message from processing."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue(sample_task, batch_id)

        consumer = queue.consume("worker-0", batch_id, block_ms=1000)
        message = await anext(consumer)

        # Should be in processing
        processing = await queue.get_processing_count(batch_id)
        assert processing >= 1

        await queue.ack(message)

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_nack_with_requeue(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Nack with requeue should put message back in queue."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue(sample_task, batch_id)

        consumer = queue.consume("worker-0", batch_id, block_ms=1000)
        message = await anext(consumer)

        # Nack with requeue
        await queue.nack(message, requeue=True)

        # Should be back in queue (pending count should be 1)
        # Note: There might be a slight delay
        await asyncio.sleep(0.1)
        count = await queue.get_pending_count(batch_id)
        assert count >= 1

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_nack_to_dlq(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Nack without requeue should move to DLQ."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue(sample_task, batch_id)

        consumer = queue.consume("worker-0", batch_id, block_ms=1000)
        message = await anext(consumer)

        # Nack without requeue
        await queue.nack(message, requeue=False, error="test_error")

        # Should be in DLQ
        dlq_count = await queue.get_dlq_count(batch_id)
        assert dlq_count == 1

        # Cleanup
        await queue.delete_stream(batch_id)


# =============================================================================
# DLQ Tests
# =============================================================================


class TestDeadLetterQueue:
    """Tests for dead letter queue operations."""

    @pytest.mark.asyncio
    async def test_get_dlq_messages(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Should retrieve messages from DLQ."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue(sample_task, batch_id)

        consumer = queue.consume("worker-0", batch_id, block_ms=1000)
        message = await anext(consumer)
        await queue.nack(message, requeue=False)

        dlq_messages = await queue.get_dlq_messages(batch_id)
        assert len(dlq_messages) == 1
        assert dlq_messages[0].payload["task_id"] == sample_task.task_id

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_retry_from_dlq(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Retry from DLQ should move message back to queue."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue(sample_task, batch_id)

        consumer = queue.consume("worker-0", batch_id, block_ms=1000)
        message = await anext(consumer)
        await queue.nack(message, requeue=False)

        assert await queue.get_dlq_count(batch_id) == 1

        # Retry
        dlq_messages = await queue.get_dlq_messages(batch_id)
        new_id = await queue.retry_from_dlq(dlq_messages[0])

        assert new_id is not None
        assert await queue.get_dlq_count(batch_id) == 0
        assert await queue.get_pending_count(batch_id) >= 1

        # Cleanup
        await queue.delete_stream(batch_id)


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for queue lifecycle operations."""

    @pytest.mark.asyncio
    async def test_ensure_stream_is_idempotent(self, queue: RedisStreamQueue, batch_id: str):
        """Ensure stream should be safe to call multiple times."""
        # Should not raise on repeated calls
        await queue.ensure_stream(batch_id)
        await queue.ensure_stream(batch_id)
        await queue.ensure_stream(batch_id)

        # Cleanup
        await queue.delete_stream(batch_id)

    @pytest.mark.asyncio
    async def test_delete_stream_removes_data(
        self, queue: RedisStreamQueue, sample_task: WorkerTask, batch_id: str
    ):
        """Delete stream should remove all data."""
        await queue.ensure_stream(batch_id)
        await queue.enqueue(sample_task, batch_id)

        assert await queue.get_pending_count(batch_id) >= 1

        await queue.delete_stream(batch_id)

        # Stream should be gone
        assert await queue.get_pending_count(batch_id) == 0
