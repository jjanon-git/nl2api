"""
Tests for TaskQueue protocol and InMemoryQueue implementation.

These tests verify the contract defined by the TaskQueue protocol
using the InMemoryQueue implementation as a reference.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from src.evalkit.contracts.worker import WorkerTask
from src.evalkit.distributed.config import QueueBackend, QueueConfig
from src.evalkit.distributed.models import QueueMessage
from src.evalkit.distributed.queue import InMemoryQueue, TaskQueue, create_queue
from src.evalkit.distributed.queue.protocol import QueueAckError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def queue() -> InMemoryQueue:
    """Create a fresh in-memory queue for each test."""
    return InMemoryQueue(max_retries=3)


@pytest.fixture
def sample_task() -> WorkerTask:
    """Create a sample worker task."""
    return WorkerTask(
        task_id="task-001",
        test_case_id="tc-001",
        batch_id="batch-001",
    )


@pytest.fixture
def sample_tasks() -> list[WorkerTask]:
    """Create multiple sample worker tasks."""
    return [
        WorkerTask(
            task_id=f"task-{i:03d}",
            test_case_id=f"tc-{i:03d}",
            batch_id="batch-001",
        )
        for i in range(10)
    ]


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestTaskQueueProtocol:
    """Tests that InMemoryQueue satisfies the TaskQueue protocol."""

    def test_implements_protocol(self, queue: InMemoryQueue):
        """InMemoryQueue should implement TaskQueue protocol."""
        assert isinstance(queue, TaskQueue)

    def test_has_required_methods(self, queue: InMemoryQueue):
        """InMemoryQueue should have all required protocol methods."""
        # Enqueue
        assert hasattr(queue, "enqueue")
        assert hasattr(queue, "enqueue_batch")

        # Consume
        assert hasattr(queue, "consume")
        assert hasattr(queue, "ack")
        assert hasattr(queue, "nack")

        # Status
        assert hasattr(queue, "get_pending_count")
        assert hasattr(queue, "get_processing_count")
        assert hasattr(queue, "get_dlq_count")

        # Stalled recovery
        assert hasattr(queue, "get_stalled_messages")
        assert hasattr(queue, "claim_stalled")

        # DLQ
        assert hasattr(queue, "get_dlq_messages")
        assert hasattr(queue, "retry_from_dlq")
        assert hasattr(queue, "delete_from_dlq")

        # Lifecycle
        assert hasattr(queue, "ensure_stream")
        assert hasattr(queue, "delete_stream")
        assert hasattr(queue, "close")


# =============================================================================
# Enqueue Tests
# =============================================================================


class TestEnqueue:
    """Tests for enqueue operations."""

    @pytest.mark.asyncio
    async def test_enqueue_single_task(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Enqueue should add a task and return a message ID."""
        message_id = await queue.enqueue(sample_task, "batch-001")

        assert message_id is not None
        assert isinstance(message_id, str)
        assert len(message_id) > 0

    @pytest.mark.asyncio
    async def test_enqueue_increments_pending_count(
        self, queue: InMemoryQueue, sample_task: WorkerTask
    ):
        """Enqueue should increment pending count."""
        assert await queue.get_pending_count("batch-001") == 0

        await queue.enqueue(sample_task, "batch-001")
        assert await queue.get_pending_count("batch-001") == 1

        await queue.enqueue(sample_task, "batch-001")
        assert await queue.get_pending_count("batch-001") == 2

    @pytest.mark.asyncio
    async def test_enqueue_batch(self, queue: InMemoryQueue, sample_tasks: list[WorkerTask]):
        """Enqueue batch should add multiple tasks efficiently."""
        message_ids = await queue.enqueue_batch(sample_tasks, "batch-001")

        assert len(message_ids) == len(sample_tasks)
        assert await queue.get_pending_count("batch-001") == len(sample_tasks)

        # All IDs should be unique
        assert len(set(message_ids)) == len(message_ids)

    @pytest.mark.asyncio
    async def test_enqueue_to_different_batches(
        self, queue: InMemoryQueue, sample_task: WorkerTask
    ):
        """Tasks in different batches should be isolated."""
        await queue.enqueue(sample_task, "batch-001")
        await queue.enqueue(sample_task, "batch-002")
        await queue.enqueue(sample_task, "batch-002")

        assert await queue.get_pending_count("batch-001") == 1
        assert await queue.get_pending_count("batch-002") == 2

    @pytest.mark.asyncio
    async def test_enqueue_after_close_fails(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Enqueue should fail after queue is closed."""
        await queue.close()

        with pytest.raises(Exception):  # QueueError
            await queue.enqueue(sample_task, "batch-001")


# =============================================================================
# Consume Tests
# =============================================================================


class TestConsume:
    """Tests for consume operations."""

    @pytest.mark.asyncio
    async def test_consume_yields_message(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Consume should yield enqueued messages."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)

        assert message is not None
        assert isinstance(message, QueueMessage)
        assert message.payload["task_id"] == sample_task.task_id

    @pytest.mark.asyncio
    async def test_consume_marks_message_as_processing(
        self, queue: InMemoryQueue, sample_task: WorkerTask
    ):
        """Consumed message should be marked as processing."""
        await queue.enqueue(sample_task, "batch-001")

        assert await queue.get_processing_count("batch-001") == 0

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        _ = await anext(consumer)

        # Message should be moved from pending to processing
        assert await queue.get_pending_count("batch-001") == 0
        assert await queue.get_processing_count("batch-001") == 1

    @pytest.mark.asyncio
    async def test_consume_sets_claimed_by(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Consumed message should have claimed_by set."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)

        assert message.claimed_by == "worker-0"
        assert message.claimed_at is not None

    @pytest.mark.asyncio
    async def test_consume_multiple_messages(
        self, queue: InMemoryQueue, sample_tasks: list[WorkerTask]
    ):
        """Consume should yield all enqueued messages."""
        await queue.enqueue_batch(sample_tasks[:3], "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        messages = []

        for _ in range(3):
            msg = await anext(consumer)
            messages.append(msg)

        assert len(messages) == 3
        task_ids = {m.payload["task_id"] for m in messages}
        expected_ids = {t.task_id for t in sample_tasks[:3]}
        assert task_ids == expected_ids

    @pytest.mark.asyncio
    async def test_consume_from_empty_queue_blocks(self, queue: InMemoryQueue):
        """Consume from empty queue should not yield immediately."""
        consumer = queue.consume("worker-0", "batch-001", block_ms=50)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(consumer), timeout=0.1)


# =============================================================================
# Ack Tests
# =============================================================================


class TestAck:
    """Tests for message acknowledgment."""

    @pytest.mark.asyncio
    async def test_ack_removes_from_processing(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Ack should remove message from processing."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)

        assert await queue.get_processing_count("batch-001") == 1

        await queue.ack(message)

        assert await queue.get_processing_count("batch-001") == 0

    @pytest.mark.asyncio
    async def test_ack_marks_as_completed(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Ack should mark message as completed."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)
        await queue.ack(message)

        # Verify via internal state (testing helper)
        state = await queue.get_all_messages("batch-001")
        assert message.message_id in state["completed"]

    @pytest.mark.asyncio
    async def test_ack_unknown_message_fails(self, queue: InMemoryQueue):
        """Ack for unknown message should fail."""
        fake_message = QueueMessage(
            message_id="unknown",
            payload={},
            stream_name="memory:batch-001",
        )

        with pytest.raises(QueueAckError):
            await queue.ack(fake_message)


# =============================================================================
# Nack Tests
# =============================================================================


class TestNack:
    """Tests for negative acknowledgment."""

    @pytest.mark.asyncio
    async def test_nack_with_requeue(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Nack with requeue should put message back in queue."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)

        await queue.nack(message, requeue=True)

        # Message should be back in pending (with new ID)
        assert await queue.get_pending_count("batch-001") == 1
        assert await queue.get_processing_count("batch-001") == 0

    @pytest.mark.asyncio
    async def test_nack_increments_attempt(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Nack should increment attempt count on requeue."""
        await queue.enqueue(sample_task, "batch-001")

        # First attempt
        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        msg1 = await anext(consumer)
        assert msg1.attempt == 1
        await queue.nack(msg1, requeue=True)

        # Second attempt
        msg2 = await anext(consumer)
        assert msg2.attempt == 2
        await queue.nack(msg2, requeue=True)

        # Third attempt
        msg3 = await anext(consumer)
        assert msg3.attempt == 3

    @pytest.mark.asyncio
    async def test_nack_max_retries_to_dlq(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Nack after max retries should move to DLQ."""
        queue = InMemoryQueue(max_retries=2)  # Only 2 attempts
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)

        # First attempt - requeue
        msg1 = await anext(consumer)
        await queue.nack(msg1, requeue=True)

        # Second attempt - should go to DLQ (max_retries=2)
        msg2 = await anext(consumer)
        assert msg2.attempt == 2
        await queue.nack(msg2, requeue=True)

        # Should be in DLQ now
        assert await queue.get_pending_count("batch-001") == 0
        assert await queue.get_dlq_count("batch-001") == 1

    @pytest.mark.asyncio
    async def test_nack_without_requeue_to_dlq(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Nack without requeue should go directly to DLQ."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)

        await queue.nack(message, requeue=False)

        assert await queue.get_pending_count("batch-001") == 0
        assert await queue.get_dlq_count("batch-001") == 1


# =============================================================================
# Stalled Task Recovery Tests
# =============================================================================


class TestStalledTaskRecovery:
    """Tests for stalled task detection and recovery."""

    @pytest.mark.asyncio
    async def test_get_stalled_messages_empty(self, queue: InMemoryQueue):
        """No stalled messages when nothing is processing."""
        stalled = await queue.get_stalled_messages("batch-001", min_idle_ms=0)
        assert stalled == []

    @pytest.mark.asyncio
    async def test_get_stalled_messages_finds_old(
        self, queue: InMemoryQueue, sample_task: WorkerTask
    ):
        """Should find messages that have been processing too long."""

        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)

        # Artificially age the message by modifying internal state
        stream = queue._get_stream("batch-001")
        old_time = datetime.now(UTC) - timedelta(minutes=5)
        aged_message = message.model_copy(update={"claimed_at": old_time})
        stream.processing[message.message_id] = aged_message

        # Now it should be detected as stalled
        stalled = await queue.get_stalled_messages("batch-001", min_idle_ms=60000)
        assert len(stalled) == 1
        assert stalled[0].message_id == message.message_id

    @pytest.mark.asyncio
    async def test_claim_stalled_reassigns(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Claim stalled should reassign to new consumer."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)

        assert message.claimed_by == "worker-0"

        # Claim for different worker
        claimed = await queue.claim_stalled(message, "worker-1")

        assert claimed.claimed_by == "worker-1"
        assert claimed.message_id == message.message_id


# =============================================================================
# DLQ Tests
# =============================================================================


class TestDeadLetterQueue:
    """Tests for dead letter queue operations."""

    @pytest.mark.asyncio
    async def test_get_dlq_messages(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Should retrieve messages from DLQ."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)
        await queue.nack(message, requeue=False)  # Send to DLQ

        dlq_messages = await queue.get_dlq_messages("batch-001")
        assert len(dlq_messages) == 1
        assert dlq_messages[0].payload["task_id"] == sample_task.task_id

    @pytest.mark.asyncio
    async def test_retry_from_dlq(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Retry from DLQ should move message back to queue."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)
        await queue.nack(message, requeue=False)

        assert await queue.get_dlq_count("batch-001") == 1
        assert await queue.get_pending_count("batch-001") == 0

        # Retry
        dlq_messages = await queue.get_dlq_messages("batch-001")
        new_id = await queue.retry_from_dlq(dlq_messages[0])

        assert new_id is not None
        assert await queue.get_dlq_count("batch-001") == 0
        assert await queue.get_pending_count("batch-001") == 1

    @pytest.mark.asyncio
    async def test_delete_from_dlq(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Delete from DLQ should permanently remove message."""
        await queue.enqueue(sample_task, "batch-001")

        consumer = queue.consume("worker-0", "batch-001", block_ms=100)
        message = await anext(consumer)
        await queue.nack(message, requeue=False)

        dlq_messages = await queue.get_dlq_messages("batch-001")
        await queue.delete_from_dlq(dlq_messages[0])

        assert await queue.get_dlq_count("batch-001") == 0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for queue lifecycle operations."""

    @pytest.mark.asyncio
    async def test_ensure_stream_creates(self, queue: InMemoryQueue):
        """Ensure stream should create stream if not exists."""
        await queue.ensure_stream("new-batch")
        # Should not raise, stream now exists
        assert await queue.get_pending_count("new-batch") == 0

    @pytest.mark.asyncio
    async def test_delete_stream_removes(self, queue: InMemoryQueue, sample_task: WorkerTask):
        """Delete stream should remove all data."""
        await queue.enqueue(sample_task, "batch-001")
        assert await queue.get_pending_count("batch-001") == 1

        await queue.delete_stream("batch-001")

        # Stream should be gone
        assert await queue.get_pending_count("batch-001") == 0

    @pytest.mark.asyncio
    async def test_close(self, queue: InMemoryQueue):
        """Close should mark queue as closed."""
        await queue.close()
        # Subsequent operations should fail
        with pytest.raises(Exception):
            await queue.enqueue(
                WorkerTask(test_case_id="tc-001"),
                "batch-001",
            )


# =============================================================================
# Factory Tests
# =============================================================================


class TestQueueFactory:
    """Tests for queue factory."""

    @pytest.mark.asyncio
    async def test_create_memory_queue(self):
        """Factory should create in-memory queue."""
        config = QueueConfig(backend=QueueBackend.MEMORY)
        queue = await create_queue(config)

        assert isinstance(queue, InMemoryQueue)
        await queue.close()

    @pytest.mark.asyncio
    async def test_create_default_is_redis(self):
        """Default should be Redis (will fail without Redis running)."""
        config = QueueConfig()
        assert config.backend == QueueBackend.REDIS

    @pytest.mark.asyncio
    async def test_azure_sb_not_implemented(self):
        """Azure Service Bus should raise NotImplementedError."""
        config = QueueConfig(backend=QueueBackend.AZURE_SB)

        with pytest.raises(NotImplementedError):
            await create_queue(config)
