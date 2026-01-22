"""
In-Memory Task Queue Implementation.

A simple in-memory implementation of TaskQueue for unit testing.
NOT suitable for production use - no persistence, no distributed support.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

from src.contracts.core import _now_utc, _generate_id
from src.contracts.worker import WorkerTask
from src.evaluation.distributed.models import QueueMessage
from src.evaluation.distributed.queue.protocol import (
    TaskQueue,
    QueueError,
    QueueAckError,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    """State for a single stream (batch)."""

    # Main queue: messages waiting to be consumed
    pending: asyncio.Queue[QueueMessage] = field(
        default_factory=lambda: asyncio.Queue()
    )

    # Processing: messages claimed by consumers (message_id -> message)
    processing: dict[str, QueueMessage] = field(default_factory=dict)

    # Dead letter queue
    dlq: list[QueueMessage] = field(default_factory=list)

    # Completed message IDs (for idempotency)
    completed: set[str] = field(default_factory=set)

    # Consumer tracking
    consumers: set[str] = field(default_factory=set)

    # Stream metadata
    created_at: datetime = field(default_factory=_now_utc)
    message_count: int = 0


class InMemoryQueue:
    """
    In-memory implementation of TaskQueue for testing.

    Features:
    - Simulates consumer groups (multiple consumers can pull from same queue)
    - Tracks pending/processing/dlq states
    - Supports stalled task detection (based on claimed_at timestamp)
    - Thread-safe within a single process (uses asyncio primitives)

    Limitations:
    - No persistence (lost on restart)
    - Single process only (not distributed)
    - No true blocking (uses polling with sleep)
    """

    def __init__(
        self,
        max_retries: int = 3,
        consumer_group: str = "eval-workers",
    ):
        """
        Initialize the in-memory queue.

        Args:
            max_retries: Maximum delivery attempts before DLQ
            consumer_group: Default consumer group name
        """
        self._max_retries = max_retries
        self._consumer_group = consumer_group
        self._streams: dict[str, StreamState] = defaultdict(StreamState)
        self._lock = asyncio.Lock()
        self._closed = False

    def _get_stream(self, batch_id: str) -> StreamState:
        """Get or create stream state for a batch."""
        return self._streams[batch_id]

    # =========================================================================
    # Enqueue Operations
    # =========================================================================

    async def enqueue(self, task: WorkerTask, batch_id: str) -> str:
        """Add a single task to the queue."""
        if self._closed:
            raise QueueError("Queue is closed")

        message_id = _generate_id()
        stream = self._get_stream(batch_id)

        message = QueueMessage(
            message_id=message_id,
            payload=task.model_dump(),
            stream_name=f"memory:{batch_id}",
            consumer_group=self._consumer_group,
            max_attempts=self._max_retries,
            enqueued_at=_now_utc(),
        )

        await stream.pending.put(message)
        stream.message_count += 1

        logger.debug(f"Enqueued task {task.task_id} as message {message_id}")
        return message_id

    async def enqueue_batch(
        self,
        tasks: list[WorkerTask],
        batch_id: str,
    ) -> list[str]:
        """Add multiple tasks to the queue efficiently."""
        message_ids = []
        for task in tasks:
            msg_id = await self.enqueue(task, batch_id)
            message_ids.append(msg_id)
        return message_ids

    # =========================================================================
    # Consume Operations
    # =========================================================================

    async def consume(
        self,
        consumer_id: str,
        batch_id: str,
        block_ms: int = 5000,
    ) -> AsyncIterator[QueueMessage]:
        """
        Consume messages from the queue.

        Yields messages as they become available. Uses polling with sleep
        to simulate blocking behavior.
        """
        if self._closed:
            raise QueueError("Queue is closed")

        stream = self._get_stream(batch_id)
        stream.consumers.add(consumer_id)
        poll_interval = min(block_ms / 1000, 0.1)  # Max 100ms between polls

        try:
            while not self._closed:
                try:
                    # Non-blocking get with timeout
                    message = await asyncio.wait_for(
                        stream.pending.get(),
                        timeout=poll_interval,
                    )

                    # Mark as processing
                    claimed_message = message.with_claimed_by(consumer_id)

                    async with self._lock:
                        stream.processing[message.message_id] = claimed_message

                    logger.debug(
                        f"Consumer {consumer_id} claimed message {message.message_id}"
                    )
                    yield claimed_message

                except asyncio.TimeoutError:
                    # No message available, continue polling
                    continue

        finally:
            stream.consumers.discard(consumer_id)

    async def ack(self, message: QueueMessage) -> None:
        """Acknowledge successful processing of a message."""
        stream = self._get_stream(self._batch_from_stream(message.stream_name))

        async with self._lock:
            if message.message_id in stream.processing:
                del stream.processing[message.message_id]
                stream.completed.add(message.message_id)
                logger.debug(f"Acked message {message.message_id}")
            else:
                raise QueueAckError(
                    f"Message {message.message_id} not found in processing"
                )

    async def nack(
        self,
        message: QueueMessage,
        requeue: bool = True,
        error: str | None = None,
    ) -> None:
        """Negatively acknowledge a message (processing failed)."""
        batch_id = self._batch_from_stream(message.stream_name)
        stream = self._get_stream(batch_id)

        async with self._lock:
            # Remove from processing
            if message.message_id in stream.processing:
                del stream.processing[message.message_id]

            if requeue and message.is_retriable:
                # Requeue with incremented attempt
                retry_message = QueueMessage(
                    message_id=_generate_id(),  # New ID for retry
                    payload=message.payload,
                    stream_name=message.stream_name,
                    consumer_group=message.consumer_group,
                    attempt=message.attempt + 1,
                    max_attempts=message.max_attempts,
                    enqueued_at=_now_utc(),
                )
                await stream.pending.put(retry_message)
                logger.debug(
                    f"Requeued message {message.message_id} as {retry_message.message_id} "
                    f"(attempt {retry_message.attempt}/{retry_message.max_attempts})"
                )
            else:
                # Move to DLQ
                dlq_message = message.model_copy(
                    update={
                        "claimed_by": None,
                        "claimed_at": None,
                    }
                )
                stream.dlq.append(dlq_message)
                logger.debug(
                    f"Moved message {message.message_id} to DLQ "
                    f"(error: {error or 'none'})"
                )

    # =========================================================================
    # Queue Status
    # =========================================================================

    async def get_pending_count(self, batch_id: str) -> int:
        """Get count of pending (unprocessed) messages."""
        stream = self._get_stream(batch_id)
        return stream.pending.qsize()

    async def get_processing_count(self, batch_id: str) -> int:
        """Get count of messages currently being processed."""
        stream = self._get_stream(batch_id)
        return len(stream.processing)

    async def get_dlq_count(self, batch_id: str) -> int:
        """Get count of messages in the dead letter queue."""
        stream = self._get_stream(batch_id)
        return len(stream.dlq)

    # =========================================================================
    # Stalled Task Recovery
    # =========================================================================

    async def get_stalled_messages(
        self,
        batch_id: str,
        min_idle_ms: int = 60000,
    ) -> list[QueueMessage]:
        """Get messages that have been pending for too long (stalled)."""
        stream = self._get_stream(batch_id)
        now = _now_utc()
        stalled = []

        async with self._lock:
            for message in stream.processing.values():
                if message.claimed_at:
                    # Ensure both datetimes are timezone-aware for comparison
                    claimed_at = message.claimed_at
                    if claimed_at.tzinfo is None:
                        # Assume naive datetime is UTC
                        from datetime import timezone
                        claimed_at = claimed_at.replace(tzinfo=timezone.utc)
                    idle_ms = (now - claimed_at).total_seconds() * 1000
                    if idle_ms >= min_idle_ms:
                        stalled.append(message)

        return stalled

    async def claim_stalled(
        self,
        message: QueueMessage,
        new_consumer_id: str,
    ) -> QueueMessage:
        """Claim a stalled message for a different consumer."""
        batch_id = self._batch_from_stream(message.stream_name)
        stream = self._get_stream(batch_id)

        async with self._lock:
            if message.message_id not in stream.processing:
                raise QueueError(
                    f"Message {message.message_id} not found in processing"
                )

            # Update the message with new consumer
            claimed_message = message.with_claimed_by(new_consumer_id)
            stream.processing[message.message_id] = claimed_message

            logger.info(
                f"Claimed stalled message {message.message_id} "
                f"from {message.claimed_by} to {new_consumer_id}"
            )
            return claimed_message

    # =========================================================================
    # Dead Letter Queue
    # =========================================================================

    async def get_dlq_messages(
        self,
        batch_id: str,
        limit: int = 100,
    ) -> list[QueueMessage]:
        """Get messages from the dead letter queue."""
        stream = self._get_stream(batch_id)
        return stream.dlq[:limit]

    async def retry_from_dlq(
        self,
        message: QueueMessage,
    ) -> str:
        """Move a message from DLQ back to the main queue."""
        batch_id = self._batch_from_stream(message.stream_name)
        stream = self._get_stream(batch_id)

        async with self._lock:
            # Remove from DLQ
            stream.dlq = [m for m in stream.dlq if m.message_id != message.message_id]

            # Create new message with reset attempts
            retry_message = QueueMessage(
                message_id=_generate_id(),
                payload=message.payload,
                stream_name=message.stream_name,
                consumer_group=message.consumer_group,
                attempt=1,  # Reset attempts
                max_attempts=message.max_attempts,
                enqueued_at=_now_utc(),
            )
            await stream.pending.put(retry_message)

            logger.info(
                f"Retried DLQ message {message.message_id} as {retry_message.message_id}"
            )
            return retry_message.message_id

    async def delete_from_dlq(
        self,
        message: QueueMessage,
    ) -> None:
        """Permanently delete a message from the DLQ."""
        batch_id = self._batch_from_stream(message.stream_name)
        stream = self._get_stream(batch_id)

        async with self._lock:
            stream.dlq = [m for m in stream.dlq if m.message_id != message.message_id]
            logger.info(f"Deleted message {message.message_id} from DLQ")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def ensure_stream(self, batch_id: str) -> None:
        """Ensure the stream exists for a batch (no-op for in-memory)."""
        # Stream is created lazily via defaultdict
        _ = self._get_stream(batch_id)

    async def delete_stream(self, batch_id: str) -> None:
        """Delete a stream and its DLQ."""
        if batch_id in self._streams:
            del self._streams[batch_id]
            logger.info(f"Deleted stream for batch {batch_id}")

    async def close(self) -> None:
        """Close the queue connection."""
        self._closed = True
        logger.info("In-memory queue closed")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _batch_from_stream(self, stream_name: str) -> str:
        """Extract batch_id from stream name."""
        # Format: "memory:{batch_id}"
        if ":" in stream_name:
            return stream_name.split(":", 1)[1]
        return stream_name

    # =========================================================================
    # Testing Helpers
    # =========================================================================

    async def get_all_messages(self, batch_id: str) -> dict:
        """Get all messages for a batch (testing only)."""
        stream = self._get_stream(batch_id)
        return {
            "pending": stream.pending.qsize(),
            "processing": list(stream.processing.values()),
            "dlq": stream.dlq,
            "completed": stream.completed,  # Return the set, not length
        }

    def reset(self) -> None:
        """Reset all state (testing only)."""
        self._streams.clear()
        self._closed = False


__all__ = ["InMemoryQueue"]
