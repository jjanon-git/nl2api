"""
Distributed Evaluation Configuration.

Configuration models for queue, worker, coordinator, and alerting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueueBackend(str, Enum):
    """Supported queue backends."""

    MEMORY = "memory"  # In-memory (testing)
    REDIS = "redis"  # Redis Streams (production)
    AZURE_SB = "azure_sb"  # Azure Service Bus (future)


class EvalMode(str, Enum):
    """Evaluation modes for workers."""

    SIMULATED = "simulated"  # Always pass (infrastructure testing)
    RESOLVER = "resolver"  # Entity resolution only
    ROUTING = "routing"  # Routing only
    TOOL_ONLY = "tool_only"  # Single agent
    ORCHESTRATOR = "orchestrator"  # Full pipeline


@dataclass(frozen=True)
class QueueConfig:
    """Configuration for task queue."""

    # Backend selection
    backend: QueueBackend = QueueBackend.REDIS

    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    # Stream configuration
    stream_prefix: str = "eval:tasks"
    consumer_group: str = "eval-workers"
    dlq_prefix: str = "eval:dlq"

    # Message handling
    max_retries: int = 3
    visibility_timeout_seconds: int = 300  # 5 minutes
    claim_min_idle_ms: int = 60000  # 1 minute before claiming stalled

    # Batch configuration
    enqueue_batch_size: int = 100  # Tasks per XADD pipeline

    # Cleanup
    stream_max_len: int = 100000  # Max messages per stream (MAXLEN)
    dlq_max_len: int = 10000  # Max messages in DLQ

    def stream_name(self, batch_id: str) -> str:
        """Get stream name for a batch."""
        return f"{self.stream_prefix}:{batch_id}"

    def dlq_name(self, batch_id: str) -> str:
        """Get DLQ stream name for a batch."""
        return f"{self.dlq_prefix}:{batch_id}"


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for evaluation workers."""

    # Identity
    worker_id: str = "worker-0"

    # Processing
    eval_mode: EvalMode = EvalMode.ORCHESTRATOR
    max_concurrent_tasks: int = 1  # Tasks processed concurrently per worker

    # Timeouts
    task_timeout_seconds: int = 300  # 5 minutes per task
    shutdown_timeout_seconds: int = 60  # Grace period for shutdown

    # Health
    heartbeat_interval_seconds: int = 30
    health_check_interval_seconds: int = 10

    # Rate limiting (per worker, not global)
    requests_per_minute: int | None = None  # None = no limit

    # Retry behavior
    retry_delay_base_seconds: float = 1.0
    retry_delay_max_seconds: float = 30.0
    retry_jitter: bool = True


@dataclass(frozen=True)
class CoordinatorConfig:
    """Configuration for batch coordinator."""

    # Monitoring
    progress_poll_interval_seconds: int = 5
    stalled_task_check_interval_seconds: int = 60

    # Timeouts
    batch_timeout_seconds: int = 3600  # 1 hour max batch duration
    coordinator_heartbeat_seconds: int = 30

    # Stalled task recovery
    stalled_task_threshold_seconds: int = 300  # 5 minutes
    max_claim_attempts: int = 3

    # Batch API (Phase 7)
    batch_api_enabled: bool = False
    batch_api_poll_interval_seconds: int = 60
    batch_api_max_wait_hours: int = 24


@dataclass(frozen=True)
class AlertConfig:
    """Configuration for CLI-based alerting."""

    # Worker health
    worker_heartbeat_timeout_seconds: int = 60
    min_workers: int = 1

    # Queue health
    max_pending_tasks: int = 1000
    max_dlq_size: int = 10

    # Batch health
    max_batch_duration_seconds: int = 3600  # 1 hour
    max_failure_rate: float = 0.1  # 10%

    # Rate limiting
    max_rate_limit_wait_seconds: int = 30

    # Alert suppression
    alert_cooldown_seconds: int = 300  # Don't repeat same alert within 5 min


@dataclass
class DistributedConfig:
    """Combined configuration for distributed evaluation."""

    queue: QueueConfig = field(default_factory=QueueConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # Telemetry
    telemetry_enabled: bool = True

    # Debug
    verbose: bool = False


__all__ = [
    "QueueBackend",
    "EvalMode",
    "QueueConfig",
    "WorkerConfig",
    "CoordinatorConfig",
    "AlertConfig",
    "DistributedConfig",
]
