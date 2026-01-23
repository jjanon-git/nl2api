"""
Redis Streams Task Queue Implementation.

Production-ready implementation of TaskQueue using Redis Streams with:
- Consumer groups for load balancing
- XPENDING/XCLAIM for stalled task recovery
- Dead letter queue for permanent failures
- Atomic operations with Lua scripts where needed
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from src.contracts.core import _now_utc
from src.contracts.worker import WorkerTask
from src.evaluation.distributed.config import QueueConfig
from src.evaluation.distributed.models import QueueMessage
from src.evaluation.distributed.queue.protocol import (
    QueueAckError,
    QueueConnectionError,
    QueueConsumeError,
    QueueEnqueueError,
    QueueError,
)

# Telemetry imports (optional)
try:
    from src.common.telemetry import get_meter

    meter = get_meter(__name__)
    TELEMETRY_AVAILABLE = True

    # Queue metrics
    _enqueue_counter = meter.create_counter(
        "eval_queue_enqueued",
        description="Number of tasks enqueued",
    )
    _ack_counter = meter.create_counter(
        "eval_queue_acked",
        description="Number of tasks acknowledged (completed)",
    )
    _nack_counter = meter.create_counter(
        "eval_queue_nacked",
        description="Number of tasks nacked (requeued or DLQ)",
    )
    _dlq_counter = meter.create_counter(
        "eval_queue_dlq",
        description="Number of tasks moved to DLQ",
    )
except ImportError:
    TELEMETRY_AVAILABLE = False
    _enqueue_counter = None
    _ack_counter = None
    _nack_counter = None
    _dlq_counter = None

logger = logging.getLogger(__name__)


class RedisStreamQueue:
    """
    Redis Streams implementation of TaskQueue.

    Uses Redis Streams with consumer groups for distributed task processing:
    - Each batch gets its own stream: eval:tasks:{batch_id}
    - All workers join the same consumer group: eval-workers
    - XREADGROUP for consuming (blocks until message available)
    - XACK for acknowledgment
    - XPENDING/XCLAIM for stalled task recovery
    - Separate stream for DLQ: eval:dlq:{batch_id}
    """

    def __init__(
        self,
        client: Redis,
        config: QueueConfig,
    ):
        """
        Initialize Redis Stream queue.

        Use RedisStreamQueue.create() factory method instead of __init__ directly.
        """
        self._client = client
        self._config = config
        self._closed = False

    @classmethod
    async def create(cls, config: QueueConfig) -> RedisStreamQueue:
        """
        Factory method to create and connect a RedisStreamQueue.

        Args:
            config: Queue configuration

        Returns:
            Connected RedisStreamQueue instance

        Raises:
            QueueConnectionError: If connection fails
        """
        try:
            client = redis.from_url(
                config.redis_url,
                db=config.redis_db,
                decode_responses=True,
            )
            # Test connection
            await client.ping()
            logger.info(f"Connected to Redis at {config.redis_url}")
            return cls(client, config)
        except Exception as e:
            raise QueueConnectionError(f"Failed to connect to Redis: {e}") from e

    def _stream_name(self, batch_id: str) -> str:
        """Get stream name for a batch."""
        return self._config.stream_name(batch_id)

    def _dlq_name(self, batch_id: str) -> str:
        """Get DLQ stream name for a batch."""
        return self._config.dlq_name(batch_id)

    # =========================================================================
    # Enqueue Operations
    # =========================================================================

    async def enqueue(self, task: WorkerTask, batch_id: str) -> str:
        """Add a single task to the queue."""
        if self._closed:
            raise QueueError("Queue is closed")

        try:
            stream = self._stream_name(batch_id)
            message_id = await self._client.xadd(
                stream,
                {
                    "task_id": task.task_id,
                    "test_case_id": task.test_case_id,
                    "batch_id": batch_id,
                    "payload": task.model_dump_json(),
                    "attempt": "1",
                    "enqueued_at": _now_utc().isoformat(),
                },
                maxlen=self._config.stream_max_len,
            )
            logger.debug(f"Enqueued task {task.task_id} as message {message_id}")

            # Record metric
            if _enqueue_counter:
                _enqueue_counter.add(1, {"batch_id": batch_id})

            return message_id
        except Exception as e:
            raise QueueEnqueueError(f"Failed to enqueue task: {e}") from e

    async def enqueue_batch(
        self,
        tasks: list[WorkerTask],
        batch_id: str,
    ) -> list[str]:
        """Add multiple tasks to the queue efficiently using pipeline."""
        if self._closed:
            raise QueueError("Queue is closed")

        if not tasks:
            return []

        try:
            stream = self._stream_name(batch_id)
            message_ids = []

            # Use pipeline for efficiency
            async with self._client.pipeline(transaction=False) as pipe:
                for task in tasks:
                    pipe.xadd(
                        stream,
                        {
                            "task_id": task.task_id,
                            "test_case_id": task.test_case_id,
                            "batch_id": batch_id,
                            "payload": task.model_dump_json(),
                            "attempt": "1",
                            "enqueued_at": _now_utc().isoformat(),
                        },
                        maxlen=self._config.stream_max_len,
                    )
                results = await pipe.execute()

            message_ids = [str(r) for r in results]
            logger.info(f"Enqueued {len(tasks)} tasks to batch {batch_id}")

            # Record metric
            if _enqueue_counter:
                _enqueue_counter.add(len(tasks), {"batch_id": batch_id})

            return message_ids

        except Exception as e:
            raise QueueEnqueueError(f"Failed to enqueue batch: {e}") from e

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
        Consume messages from the queue using XREADGROUP.

        Yields messages as they become available. Each message must be
        acknowledged (ack) or negatively acknowledged (nack) after processing.
        """
        if self._closed:
            raise QueueError("Queue is closed")

        stream = self._stream_name(batch_id)
        group = self._config.consumer_group

        # Ensure consumer group exists
        await self.ensure_stream(batch_id)

        try:
            while not self._closed:
                try:
                    # XREADGROUP with blocking
                    result = await self._client.xreadgroup(
                        groupname=group,
                        consumername=consumer_id,
                        streams={stream: ">"},  # Only new messages
                        count=1,
                        block=block_ms,
                    )

                    if not result:
                        # Timeout, no message available
                        continue

                    # Parse result: [(stream_name, [(message_id, fields), ...])]
                    for stream_name, messages in result:
                        for message_id, fields in messages:
                            queue_message = self._fields_to_message(
                                message_id=message_id,
                                fields=fields,
                                stream_name=stream_name,
                                consumer_id=consumer_id,
                            )
                            logger.debug(f"Consumer {consumer_id} received message {message_id}")
                            yield queue_message

                except ResponseError as e:
                    if "NOGROUP" in str(e):
                        # Group doesn't exist, recreate
                        await self.ensure_stream(batch_id)
                        continue
                    raise

        except asyncio.CancelledError:
            logger.info(f"Consumer {consumer_id} cancelled")
            raise
        except Exception as e:
            raise QueueConsumeError(f"Failed to consume: {e}") from e

    async def ack(self, message: QueueMessage) -> None:
        """Acknowledge successful processing using XACK."""
        try:
            result = await self._client.xack(
                message.stream_name,
                message.consumer_group,
                message.message_id,
            )
            if result == 0:
                logger.warning(
                    f"XACK returned 0 for message {message.message_id} "
                    "(may have already been acked)"
                )
            else:
                logger.debug(f"Acked message {message.message_id}")

                # Record metric
                batch_id = self._batch_from_stream(message.stream_name)
                if _ack_counter:
                    _ack_counter.add(1, {"batch_id": batch_id})
        except Exception as e:
            raise QueueAckError(f"Failed to ack message: {e}") from e

    async def nack(
        self,
        message: QueueMessage,
        requeue: bool = True,
        error: str | None = None,
    ) -> None:
        """Negatively acknowledge a message."""
        try:
            # Always ack to remove from pending (we'll requeue separately if needed)
            await self._client.xack(
                message.stream_name,
                message.consumer_group,
                message.message_id,
            )

            if requeue and message.is_retriable:
                # Requeue with incremented attempt
                batch_id = self._batch_from_stream(message.stream_name)
                stream = self._stream_name(batch_id)

                await self._client.xadd(
                    stream,
                    {
                        "task_id": message.payload.get("task_id", ""),
                        "test_case_id": message.payload.get("test_case_id", ""),
                        "batch_id": batch_id,
                        "payload": (
                            message.payload.get("payload", "")
                            if isinstance(message.payload.get("payload"), str)
                            else WorkerTask.model_validate(message.payload).model_dump_json()
                        ),
                        "attempt": str(message.attempt + 1),
                        "enqueued_at": _now_utc().isoformat(),
                        "error": error or "",
                    },
                    maxlen=self._config.stream_max_len,
                )
                logger.debug(
                    f"Requeued message {message.message_id} "
                    f"(attempt {message.attempt + 1}/{message.max_attempts})"
                )

                # Record nack metric (requeued)
                if _nack_counter:
                    _nack_counter.add(1, {"batch_id": batch_id, "action": "requeue"})
            else:
                # Move to DLQ
                await self._move_to_dlq(message, error)

        except Exception as e:
            raise QueueError(f"Failed to nack message: {e}") from e

    async def _move_to_dlq(self, message: QueueMessage, error: str | None) -> None:
        """Move a message to the dead letter queue."""
        batch_id = self._batch_from_stream(message.stream_name)
        dlq = self._dlq_name(batch_id)

        await self._client.xadd(
            dlq,
            {
                "original_message_id": message.message_id,
                "task_id": message.payload.get("task_id", ""),
                "test_case_id": message.payload.get("test_case_id", ""),
                "batch_id": batch_id,
                "payload": (
                    message.payload.get("payload", "")
                    if isinstance(message.payload.get("payload"), str)
                    else WorkerTask.model_validate(message.payload).model_dump_json()
                ),
                "attempt": str(message.attempt),
                "error": error or "max_retries_exceeded",
                "moved_at": _now_utc().isoformat(),
            },
            maxlen=self._config.dlq_max_len,
        )
        logger.info(f"Moved message {message.message_id} to DLQ (error: {error})")

        # Record DLQ metrics
        if _dlq_counter:
            _dlq_counter.add(1, {"batch_id": batch_id})
        if _nack_counter:
            _nack_counter.add(1, {"batch_id": batch_id, "action": "dlq"})

    # =========================================================================
    # Queue Status
    # =========================================================================

    async def get_pending_count(self, batch_id: str) -> int:
        """Get count of pending messages using XLEN."""
        try:
            stream = self._stream_name(batch_id)
            length = await self._client.xlen(stream)
            return length
        except Exception:
            return 0

    async def get_processing_count(self, batch_id: str) -> int:
        """Get count of messages being processed using XPENDING."""
        try:
            stream = self._stream_name(batch_id)
            group = self._config.consumer_group

            # XPENDING returns summary: [pending_count, min_id, max_id, [[consumer, count], ...]]
            result = await self._client.xpending(stream, group)
            if result and isinstance(result, dict):
                return result.get("pending", 0)
            elif result and isinstance(result, (list, tuple)) and len(result) > 0:
                return result[0] if isinstance(result[0], int) else 0
            return 0
        except ResponseError:
            # Group doesn't exist
            return 0
        except Exception as e:
            logger.warning(f"Failed to get processing count: {e}")
            return 0

    async def get_dlq_count(self, batch_id: str) -> int:
        """Get count of messages in DLQ."""
        try:
            dlq = self._dlq_name(batch_id)
            return await self._client.xlen(dlq)
        except Exception:
            return 0

    # =========================================================================
    # Stalled Task Recovery
    # =========================================================================

    async def get_stalled_messages(
        self,
        batch_id: str,
        min_idle_ms: int = 60000,
    ) -> list[QueueMessage]:
        """Get messages that have been pending for too long using XPENDING."""
        try:
            stream = self._stream_name(batch_id)
            group = self._config.consumer_group

            # XPENDING with range to get individual entries
            pending = await self._client.xpending_range(
                stream,
                group,
                min="-",
                max="+",
                count=100,
                idle=min_idle_ms,
            )

            stalled = []
            for entry in pending:
                # entry is dict: {'message_id': ..., 'consumer': ..., 'time_since_delivered': ..., 'times_delivered': ...}
                message_id = entry.get("message_id", entry.get("id", ""))
                consumer = entry.get("consumer", "")
                idle_time = entry.get("time_since_delivered", entry.get("idle", 0))
                entry.get("times_delivered", entry.get("deliveries", 1))

                if idle_time >= min_idle_ms:
                    # Fetch the actual message data
                    messages = await self._client.xrange(stream, message_id, message_id)
                    if messages:
                        _, fields = messages[0]
                        queue_message = self._fields_to_message(
                            message_id=message_id,
                            fields=fields,
                            stream_name=stream,
                            consumer_id=consumer,
                        )
                        stalled.append(queue_message)

            return stalled

        except ResponseError:
            return []
        except Exception as e:
            logger.warning(f"Failed to get stalled messages: {e}")
            return []

    async def claim_stalled(
        self,
        message: QueueMessage,
        new_consumer_id: str,
    ) -> QueueMessage:
        """Claim a stalled message for a different consumer using XCLAIM."""
        try:
            result = await self._client.xclaim(
                message.stream_name,
                message.consumer_group,
                new_consumer_id,
                min_idle_time=0,  # Claim regardless of idle time
                message_ids=[message.message_id],
            )

            if not result:
                raise QueueError(
                    f"Failed to claim message {message.message_id} (may have been acked or deleted)"
                )

            # result is list of (message_id, fields)
            message_id, fields = result[0]
            claimed_message = self._fields_to_message(
                message_id=message_id,
                fields=fields,
                stream_name=message.stream_name,
                consumer_id=new_consumer_id,
            )

            logger.info(
                f"Claimed message {message.message_id} "
                f"from {message.claimed_by} to {new_consumer_id}"
            )
            return claimed_message

        except Exception as e:
            raise QueueError(f"Failed to claim message: {e}") from e

    # =========================================================================
    # Dead Letter Queue
    # =========================================================================

    async def get_dlq_messages(
        self,
        batch_id: str,
        limit: int = 100,
    ) -> list[QueueMessage]:
        """Get messages from the DLQ."""
        try:
            dlq = self._dlq_name(batch_id)
            messages = await self._client.xrange(dlq, count=limit)

            result = []
            for message_id, fields in messages:
                queue_message = self._fields_to_message(
                    message_id=message_id,
                    fields=fields,
                    stream_name=dlq,
                    consumer_id=None,
                )
                result.append(queue_message)

            return result

        except Exception as e:
            logger.warning(f"Failed to get DLQ messages: {e}")
            return []

    async def retry_from_dlq(self, message: QueueMessage) -> str:
        """Move a message from DLQ back to the main queue."""
        try:
            batch_id = self._batch_from_stream(message.stream_name)
            stream = self._stream_name(batch_id)
            dlq = self._dlq_name(batch_id)

            # Add back to main queue with reset attempts
            new_id = await self._client.xadd(
                stream,
                {
                    "task_id": message.payload.get("task_id", ""),
                    "test_case_id": message.payload.get("test_case_id", ""),
                    "batch_id": batch_id,
                    "payload": message.payload.get("payload", ""),
                    "attempt": "1",  # Reset attempts
                    "enqueued_at": _now_utc().isoformat(),
                    "retried_from": message.message_id,
                },
                maxlen=self._config.stream_max_len,
            )

            # Remove from DLQ
            await self._client.xdel(dlq, message.message_id)

            logger.info(f"Retried DLQ message {message.message_id} as {new_id}")
            return new_id

        except Exception as e:
            raise QueueError(f"Failed to retry from DLQ: {e}") from e

    async def delete_from_dlq(self, message: QueueMessage) -> None:
        """Permanently delete a message from the DLQ."""
        try:
            dlq = message.stream_name
            await self._client.xdel(dlq, message.message_id)
            logger.info(f"Deleted message {message.message_id} from DLQ")
        except Exception as e:
            raise QueueError(f"Failed to delete from DLQ: {e}") from e

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def ensure_stream(self, batch_id: str) -> None:
        """Ensure the stream and consumer group exist."""
        stream = self._stream_name(batch_id)
        group = self._config.consumer_group

        try:
            # Create consumer group (also creates stream if needed)
            await self._client.xgroup_create(
                stream,
                group,
                id="0",  # Start from beginning
                mkstream=True,
            )
            logger.debug(f"Created consumer group {group} for stream {stream}")
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                # BUSYGROUP means group already exists, which is fine
                raise

    async def delete_stream(self, batch_id: str) -> None:
        """Delete a stream and its DLQ."""
        try:
            stream = self._stream_name(batch_id)
            dlq = self._dlq_name(batch_id)

            await self._client.delete(stream, dlq)
            logger.info(f"Deleted streams for batch {batch_id}")
        except Exception as e:
            raise QueueError(f"Failed to delete stream: {e}") from e

    async def close(self) -> None:
        """Close the Redis connection."""
        self._closed = True
        await self._client.aclose()
        logger.info("Redis queue closed")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _fields_to_message(
        self,
        message_id: str,
        fields: dict[str, Any],
        stream_name: str,
        consumer_id: str | None,
    ) -> QueueMessage:
        """Convert Redis stream fields to QueueMessage."""
        from datetime import datetime

        # Parse payload - it may be JSON string or dict
        payload = fields.get("payload", "{}")
        if isinstance(payload, str):
            import json

            try:
                payload_dict = json.loads(payload)
            except json.JSONDecodeError:
                payload_dict = {"raw": payload}
        else:
            payload_dict = payload

        # Merge top-level fields into payload for compatibility
        payload_dict.update(
            {
                "task_id": fields.get("task_id", payload_dict.get("task_id")),
                "test_case_id": fields.get("test_case_id", payload_dict.get("test_case_id")),
                "batch_id": fields.get("batch_id", payload_dict.get("batch_id")),
            }
        )

        attempt = int(fields.get("attempt", "1"))
        enqueued_str = fields.get("enqueued_at", "")

        enqueued_at = _now_utc()
        if enqueued_str:
            try:
                enqueued_at = datetime.fromisoformat(enqueued_str)
            except ValueError:
                pass

        return QueueMessage(
            message_id=str(message_id),
            payload=payload_dict,
            stream_name=stream_name,
            consumer_group=self._config.consumer_group,
            attempt=attempt,
            max_attempts=self._config.max_retries,
            enqueued_at=enqueued_at,
            claimed_by=consumer_id,
            claimed_at=_now_utc() if consumer_id else None,
        )

    def _batch_from_stream(self, stream_name: str) -> str:
        """Extract batch_id from stream name."""
        # Format: "eval:tasks:{batch_id}" or "eval:dlq:{batch_id}"
        parts = stream_name.split(":")
        if len(parts) >= 3:
            return parts[-1]
        return stream_name


__all__ = ["RedisStreamQueue"]
