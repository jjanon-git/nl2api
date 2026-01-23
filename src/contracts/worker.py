"""
Worker Models

Models for batch processing: worker tasks, batch jobs, and worker configuration.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from src.contracts.core import TaskPriority, TaskStatus, _generate_id, _now_utc

# =============================================================================
# Worker Task
# =============================================================================


class WorkerTask(BaseModel):
    """
    Message sent to workers via Azure Service Bus.

    Contains everything needed for idempotent processing.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    task_id: str = Field(
        default_factory=_generate_id,
        description="Unique task identifier",
    )
    test_case_id: str = Field(..., description="Test case to evaluate")
    batch_id: str | None = Field(default=None, description="Parent batch")

    # Processing hints
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    created_at: datetime = Field(default_factory=_now_utc)

    # Idempotency
    idempotency_key: str | None = Field(
        default=None,
        description="Key for deduplication (auto-generated if not provided)",
    )

    # Retry tracking (updated by worker, not in original message)
    attempt_count: int = Field(default=0, ge=0)
    last_error: str | None = Field(default=None)

    @model_validator(mode="after")
    def ensure_idempotency_key(self) -> WorkerTask:
        """Generate idempotency key if not provided."""
        if self.idempotency_key is None:
            object.__setattr__(
                self,
                "idempotency_key",
                f"{self.test_case_id}:{self.task_id}",
            )
        return self


# =============================================================================
# Batch Job
# =============================================================================


class BatchJob(BaseModel):
    """
    Tracks a batch of test cases submitted together.

    Stored in Table Storage for status tracking.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    batch_id: str = Field(
        default_factory=_generate_id,
        description="Unique batch identifier",
    )

    # Tracking
    total_tests: int = Field(..., ge=1, description="Number of tests in batch")
    completed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    status: TaskStatus = Field(default=TaskStatus.PENDING)

    # Timestamps
    created_at: datetime = Field(default_factory=_now_utc)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    # Metadata
    submitted_by: str | None = Field(default=None, description="User/system that submitted")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    tags: tuple[str, ...] = Field(default_factory=tuple)

    @computed_field
    @property
    def progress_pct(self) -> float:
        """Completion percentage."""
        if self.total_tests == 0:
            return 0.0
        return ((self.completed_count + self.failed_count) / self.total_tests) * 100

    @computed_field
    @property
    def partition_key(self) -> str:
        """Azure Table Storage partition key (by date for efficient queries)."""
        return self.created_at.strftime("%Y-%m-%d")

    @computed_field
    @property
    def row_key(self) -> str:
        """Azure Table Storage row key."""
        return self.batch_id


# =============================================================================
# Worker Configuration
# =============================================================================


class WorkerConfig(BaseModel):
    """Configuration for worker instances."""

    model_config = ConfigDict(frozen=True)

    # Concurrency
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    prefetch_count: int = Field(default=10, ge=1, le=100)

    # Timeouts
    task_timeout_seconds: int = Field(default=300, ge=30)
    visibility_timeout_seconds: int = Field(default=300, ge=60)

    # Circuit breaker
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_recovery_timeout_seconds: int = Field(default=30, ge=5)

    # Retry
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_backoff_base_seconds: float = Field(default=1.0, ge=0.1)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "WorkerTask",
    "BatchJob",
    "WorkerConfig",
]
