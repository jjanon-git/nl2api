"""
Task Queue Protocol.

Defines the abstract interface for task queues used in distributed evaluation.
Implementations include in-memory (testing), Redis Streams (production),
and Azure Service Bus (future).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from src.evalkit.contracts.worker import WorkerTask
from src.evalkit.distributed.models import QueueMessage


@runtime_checkable
class TaskQueue(Protocol):
    """
    Abstract task queue interface for distributed evaluation.

    Implementations must provide:
    - Reliable message delivery (at-least-once semantics)
    - Consumer groups for load balancing
    - Dead letter queue for permanent failures
    - Stalled task detection and recovery

    Thread Safety:
        Implementations should be safe for concurrent use from multiple
        async tasks within a single process.
    """

    # =========================================================================
    # Enqueue Operations
    # =========================================================================

    @abstractmethod
    async def enqueue(self, task: WorkerTask, batch_id: str) -> str:
        """
        Add a single task to the queue.

        Args:
            task: The worker task to enqueue
            batch_id: Batch identifier (used for stream routing)

        Returns:
            Message ID assigned by the queue

        Raises:
            QueueError: If enqueue fails
        """
        ...

    @abstractmethod
    async def enqueue_batch(
        self,
        tasks: list[WorkerTask],
        batch_id: str,
    ) -> list[str]:
        """
        Add multiple tasks to the queue efficiently.

        Args:
            tasks: List of worker tasks to enqueue
            batch_id: Batch identifier (used for stream routing)

        Returns:
            List of message IDs in same order as input tasks

        Raises:
            QueueError: If enqueue fails (partial failure possible)
        """
        ...

    # =========================================================================
    # Consume Operations
    # =========================================================================

    @abstractmethod
    async def consume(
        self,
        consumer_id: str,
        batch_id: str,
        block_ms: int = 5000,
    ) -> AsyncIterator[QueueMessage]:
        """
        Consume messages from the queue.

        This is an async generator that yields messages as they become available.
        Messages must be acknowledged (ack) or negatively acknowledged (nack)
        after processing.

        Args:
            consumer_id: Unique identifier for this consumer (worker)
            batch_id: Batch to consume from
            block_ms: How long to block waiting for new messages

        Yields:
            QueueMessage instances as they become available

        Note:
            The generator handles consumer group registration automatically.
            Yielded messages are marked as "pending" until ack/nack.
        """
        ...

    @abstractmethod
    async def ack(self, message: QueueMessage) -> None:
        """
        Acknowledge successful processing of a message.

        Removes the message from the pending list, marking it as complete.

        Args:
            message: The message to acknowledge

        Raises:
            QueueError: If acknowledgment fails
        """
        ...

    @abstractmethod
    async def nack(
        self,
        message: QueueMessage,
        requeue: bool = True,
        error: str | None = None,
    ) -> None:
        """
        Negatively acknowledge a message (processing failed).

        Args:
            message: The message to negatively acknowledge
            requeue: If True, message will be requeued for retry (if attempts remain).
                     If False, message goes directly to DLQ.
            error: Optional error message to record

        Behavior:
            - If requeue=True and attempts < max_attempts: Message is requeued
            - If requeue=True and attempts >= max_attempts: Message goes to DLQ
            - If requeue=False: Message goes directly to DLQ

        Raises:
            QueueError: If nack fails
        """
        ...

    # =========================================================================
    # Queue Status
    # =========================================================================

    @abstractmethod
    async def get_pending_count(self, batch_id: str) -> int:
        """
        Get count of pending (unprocessed) messages.

        Args:
            batch_id: Batch to check

        Returns:
            Number of messages waiting to be processed
        """
        ...

    @abstractmethod
    async def get_processing_count(self, batch_id: str) -> int:
        """
        Get count of messages currently being processed.

        These are messages that have been delivered but not yet ack/nack'd.

        Args:
            batch_id: Batch to check

        Returns:
            Number of messages currently being processed
        """
        ...

    @abstractmethod
    async def get_dlq_count(self, batch_id: str) -> int:
        """
        Get count of messages in the dead letter queue.

        Args:
            batch_id: Batch to check

        Returns:
            Number of messages in DLQ
        """
        ...

    # =========================================================================
    # Stalled Task Recovery
    # =========================================================================

    @abstractmethod
    async def get_stalled_messages(
        self,
        batch_id: str,
        min_idle_ms: int = 60000,
    ) -> list[QueueMessage]:
        """
        Get messages that have been pending for too long (stalled).

        These are messages that were delivered to a consumer but never
        acknowledged, suggesting the consumer may have crashed.

        Args:
            batch_id: Batch to check
            min_idle_ms: Minimum idle time to consider a message stalled

        Returns:
            List of stalled messages
        """
        ...

    @abstractmethod
    async def claim_stalled(
        self,
        message: QueueMessage,
        new_consumer_id: str,
    ) -> QueueMessage:
        """
        Claim a stalled message for a different consumer.

        Used to recover from worker failures by reassigning tasks.

        Args:
            message: The stalled message to claim
            new_consumer_id: The consumer taking over

        Returns:
            Updated QueueMessage with new consumer assignment

        Raises:
            QueueError: If claim fails (message may have been ack'd already)
        """
        ...

    # =========================================================================
    # Dead Letter Queue
    # =========================================================================

    @abstractmethod
    async def get_dlq_messages(
        self,
        batch_id: str,
        limit: int = 100,
    ) -> list[QueueMessage]:
        """
        Get messages from the dead letter queue.

        Args:
            batch_id: Batch to check
            limit: Maximum messages to return

        Returns:
            List of failed messages
        """
        ...

    @abstractmethod
    async def retry_from_dlq(
        self,
        message: QueueMessage,
    ) -> str:
        """
        Move a message from DLQ back to the main queue.

        Resets the attempt counter for a fresh retry.

        Args:
            message: DLQ message to retry

        Returns:
            New message ID

        Raises:
            QueueError: If retry fails
        """
        ...

    @abstractmethod
    async def delete_from_dlq(
        self,
        message: QueueMessage,
    ) -> None:
        """
        Permanently delete a message from the DLQ.

        Use with caution - this is a destructive operation.

        Args:
            message: DLQ message to delete

        Raises:
            QueueError: If deletion fails
        """
        ...

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @abstractmethod
    async def ensure_stream(self, batch_id: str) -> None:
        """
        Ensure the stream and consumer group exist for a batch.

        Creates the stream and consumer group if they don't exist.
        Idempotent - safe to call multiple times.

        Args:
            batch_id: Batch to create stream for
        """
        ...

    @abstractmethod
    async def delete_stream(self, batch_id: str) -> None:
        """
        Delete a stream and its DLQ.

        Use for cleanup after batch completion.

        Args:
            batch_id: Batch to delete
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Close the queue connection.

        Should be called when the queue is no longer needed.
        """
        ...


class QueueError(Exception):
    """Base exception for queue operations."""

    pass


class QueueConnectionError(QueueError):
    """Failed to connect to queue backend."""

    pass


class QueueEnqueueError(QueueError):
    """Failed to enqueue message."""

    pass


class QueueConsumeError(QueueError):
    """Failed to consume message."""

    pass


class QueueAckError(QueueError):
    """Failed to acknowledge message."""

    pass


__all__ = [
    "TaskQueue",
    "QueueError",
    "QueueConnectionError",
    "QueueEnqueueError",
    "QueueConsumeError",
    "QueueAckError",
]
