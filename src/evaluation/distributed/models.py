"""
Distributed Evaluation Models.

Data models for the distributed evaluation infrastructure.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.core import _generate_id, _now_utc


class QueueMessageStatus(str, Enum):
    """Status of a queue message."""

    PENDING = "pending"  # In queue, not yet claimed
    PROCESSING = "processing"  # Claimed by a worker
    COMPLETED = "completed"  # Successfully processed
    FAILED = "failed"  # Failed, will retry
    DEAD = "dead"  # Max retries exceeded, in DLQ


class QueueMessage(BaseModel):
    """
    A message in the task queue.

    Wraps a WorkerTask with queue-specific metadata for tracking
    delivery, acknowledgment, and retry state.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    message_id: str = Field(
        default_factory=_generate_id,
        description="Unique message ID (from Redis Stream)",
    )

    # Payload
    payload: dict[str, Any] = Field(
        ...,
        description="Task data (serialized WorkerTask)",
    )

    # Queue metadata
    stream_name: str = Field(
        ...,
        description="Redis stream this message belongs to",
    )
    consumer_group: str = Field(
        default="eval-workers",
        description="Consumer group name",
    )

    # Delivery tracking
    attempt: int = Field(
        default=1,
        ge=1,
        description="Current delivery attempt (1-based)",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum delivery attempts before DLQ",
    )

    # Timestamps
    enqueued_at: datetime = Field(
        default_factory=_now_utc,
        description="When message was added to queue",
    )
    claimed_at: datetime | None = Field(
        default=None,
        description="When message was claimed by a worker",
    )
    last_delivery_at: datetime | None = Field(
        default=None,
        description="When message was last delivered",
    )

    # Worker tracking
    claimed_by: str | None = Field(
        default=None,
        description="Worker ID that claimed this message",
    )

    @property
    def is_retriable(self) -> bool:
        """Whether this message can be retried."""
        return self.attempt < self.max_attempts

    @property
    def task_id(self) -> str | None:
        """Extract task_id from payload."""
        return self.payload.get("task_id")

    @property
    def test_case_id(self) -> str | None:
        """Extract test_case_id from payload."""
        return self.payload.get("test_case_id")

    @property
    def batch_id(self) -> str | None:
        """Extract batch_id from payload."""
        return self.payload.get("batch_id")

    def with_attempt(self, attempt: int) -> QueueMessage:
        """Create a copy with updated attempt count."""
        return self.model_copy(update={"attempt": attempt})

    def with_claimed_by(self, worker_id: str) -> QueueMessage:
        """Create a copy marked as claimed by a worker."""
        return self.model_copy(
            update={
                "claimed_by": worker_id,
                "claimed_at": _now_utc(),
                "last_delivery_at": _now_utc(),
            }
        )


class BatchProgress(BaseModel):
    """Progress tracking for a distributed batch."""

    model_config = ConfigDict(frozen=True)

    batch_id: str
    total: int = Field(ge=0)
    completed: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    pending: int = Field(default=0, ge=0)
    in_dlq: int = Field(default=0, ge=0)

    # Timing
    started_at: datetime | None = None
    estimated_completion: datetime | None = None

    @property
    def progress_pct(self) -> float:
        """Completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    @property
    def pass_rate(self) -> float:
        """Pass rate of completed tests."""
        if self.completed == 0:
            return 0.0
        return (self.passed / self.completed) * 100

    @property
    def is_complete(self) -> bool:
        """Whether all tests have been processed."""
        return self.completed >= self.total


class WorkerStatus(BaseModel):
    """Status of a worker."""

    model_config = ConfigDict(frozen=True)

    worker_id: str
    status: str = Field(default="running")  # running, idle, stopped, error
    tasks_processed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    last_heartbeat: datetime = Field(default_factory=_now_utc)
    current_task_id: str | None = None
    current_batch_id: str | None = None
    error_message: str | None = None

    @property
    def is_healthy(self) -> bool:
        """Whether worker is considered healthy."""
        return self.status in ("running", "idle")


__all__ = [
    "QueueMessageStatus",
    "QueueMessage",
    "BatchProgress",
    "WorkerStatus",
]
