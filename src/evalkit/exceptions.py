"""
EvalKit Exception Hierarchy

Provides structured exception types for the evaluation framework.
All evalkit-specific exceptions inherit from EvalKitError.

Usage:
    from src.evalkit.exceptions import StorageError, EvaluationError

    try:
        result = await repo.get(id)
    except StorageError as e:
        logger.error(f"Storage operation failed: {e}")
"""

from __future__ import annotations


class EvalKitError(Exception):
    """
    Base exception for all EvalKit errors.

    All custom exceptions in the evalkit module should inherit from this class
    to enable catching all evalkit-specific errors with a single except clause.

    Attributes:
        message: Human-readable error description
        code: Optional machine-readable error code for programmatic handling
    """

    def __init__(self, message: str, code: str | None = None) -> None:
        self.message = message
        self.code = code
        super().__init__(message)

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(EvalKitError):
    """Base class for configuration-related errors."""

    pass


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid or malformed."""

    def __init__(self, field: str, value: str, reason: str) -> None:
        super().__init__(
            f"Invalid configuration for '{field}': {value}. {reason}",
            code="CONFIG_INVALID",
        )
        self.field = field
        self.value = value
        self.reason = reason


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    def __init__(self, field: str, hint: str | None = None) -> None:
        message = f"Missing required configuration: '{field}'"
        if hint:
            message += f". {hint}"
        super().__init__(message, code="CONFIG_MISSING")
        self.field = field


# =============================================================================
# Storage Errors
# =============================================================================


class StorageError(EvalKitError):
    """Base class for storage/repository errors."""

    pass


class StorageConnectionError(StorageError):
    """Failed to connect to storage backend."""

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(
            f"Failed to connect to {backend} storage: {reason}",
            code="STORAGE_CONNECTION",
        )
        self.backend = backend
        self.reason = reason


class StorageQueryError(StorageError):
    """Failed to execute a storage query."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            f"Storage query failed for '{operation}': {reason}",
            code="STORAGE_QUERY",
        )
        self.operation = operation
        self.reason = reason


class StorageWriteError(StorageError):
    """Failed to write to storage."""

    def __init__(self, entity: str, reason: str) -> None:
        super().__init__(
            f"Failed to write {entity}: {reason}",
            code="STORAGE_WRITE",
        )
        self.entity = entity
        self.reason = reason


class EntityNotFoundError(StorageError):
    """Requested entity does not exist."""

    def __init__(self, entity_type: str, entity_id: str) -> None:
        super().__init__(
            f"{entity_type} not found: {entity_id}",
            code="STORAGE_NOT_FOUND",
        )
        self.entity_type = entity_type
        self.entity_id = entity_id


# =============================================================================
# Queue Errors (moved from protocol.py for consolidation)
# =============================================================================


class QueueError(EvalKitError):
    """Base exception for queue operations."""

    pass


class QueueConnectionError(QueueError):
    """Failed to connect to queue backend."""

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(
            f"Failed to connect to {backend} queue: {reason}",
            code="QUEUE_CONNECTION",
        )
        self.backend = backend
        self.reason = reason


class QueueEnqueueError(QueueError):
    """Failed to enqueue message."""

    def __init__(self, batch_id: str, reason: str) -> None:
        super().__init__(
            f"Failed to enqueue task for batch {batch_id}: {reason}",
            code="QUEUE_ENQUEUE",
        )
        self.batch_id = batch_id
        self.reason = reason


class QueueConsumeError(QueueError):
    """Failed to consume message."""

    def __init__(self, consumer_id: str, reason: str) -> None:
        super().__init__(
            f"Consumer {consumer_id} failed to consume: {reason}",
            code="QUEUE_CONSUME",
        )
        self.consumer_id = consumer_id
        self.reason = reason


class QueueAckError(QueueError):
    """Failed to acknowledge message."""

    def __init__(self, message_id: str, reason: str) -> None:
        super().__init__(
            f"Failed to acknowledge message {message_id}: {reason}",
            code="QUEUE_ACK",
        )
        self.message_id = message_id
        self.reason = reason


# =============================================================================
# Evaluation Errors
# =============================================================================


class EvaluationError(EvalKitError):
    """Base class for evaluation-related errors."""

    pass


class EvaluationTimeoutError(EvaluationError):
    """Evaluation exceeded time limit."""

    def __init__(self, test_case_id: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Evaluation timed out for test case {test_case_id} after {timeout_seconds}s",
            code="EVAL_TIMEOUT",
        )
        self.test_case_id = test_case_id
        self.timeout_seconds = timeout_seconds


class StageEvaluationError(EvaluationError):
    """A specific evaluation stage failed."""

    def __init__(self, stage_name: str, test_case_id: str, reason: str) -> None:
        super().__init__(
            f"Stage '{stage_name}' failed for test case {test_case_id}: {reason}",
            code="EVAL_STAGE_FAILED",
        )
        self.stage_name = stage_name
        self.test_case_id = test_case_id
        self.reason = reason


class LLMJudgeError(EvaluationError):
    """LLM-as-Judge evaluation failed."""

    def __init__(self, reason: str, model: str | None = None) -> None:
        message = f"LLM judge evaluation failed: {reason}"
        if model:
            message = f"LLM judge ({model}) evaluation failed: {reason}"
        super().__init__(message, code="EVAL_LLM_JUDGE")
        self.reason = reason
        self.model = model


# =============================================================================
# Distributed Processing Errors
# =============================================================================


class DistributedError(EvalKitError):
    """Base class for distributed processing errors."""

    pass


class WorkerError(DistributedError):
    """Worker process encountered an error."""

    def __init__(self, worker_id: str, reason: str) -> None:
        super().__init__(
            f"Worker {worker_id} error: {reason}",
            code="DISTRIBUTED_WORKER",
        )
        self.worker_id = worker_id
        self.reason = reason


class CoordinatorError(DistributedError):
    """Coordinator encountered an error."""

    def __init__(self, batch_id: str, reason: str) -> None:
        super().__init__(
            f"Coordinator error for batch {batch_id}: {reason}",
            code="DISTRIBUTED_COORDINATOR",
        )
        self.batch_id = batch_id
        self.reason = reason


class MessageProcessingError(DistributedError):
    """Failed to process a queue message."""

    def __init__(self, message_id: str, reason: str) -> None:
        super().__init__(
            f"Failed to process message {message_id}: {reason}",
            code="DISTRIBUTED_MESSAGE",
        )
        self.message_id = message_id
        self.reason = reason


# =============================================================================
# External Service Errors
# =============================================================================


class ExternalServiceError(EvalKitError):
    """Base class for external service errors."""

    pass


class CircuitOpenError(ExternalServiceError):
    """Circuit breaker is open, rejecting requests."""

    def __init__(self, service: str, retry_after: float | None = None) -> None:
        message = f"Circuit breaker open for {service}"
        if retry_after:
            message += f", retry after {retry_after:.1f}s"
        super().__init__(message, code="CIRCUIT_OPEN")
        self.service = service
        self.retry_after = retry_after


class RetryExhaustedError(ExternalServiceError):
    """All retry attempts exhausted."""

    def __init__(self, operation: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"Exhausted {attempts} retries for '{operation}': {last_error}",
            code="RETRY_EXHAUSTED",
        )
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error


# =============================================================================
# Telemetry Errors
# =============================================================================


class TelemetryError(EvalKitError):
    """Base class for telemetry/observability errors."""

    pass


class MetricsError(TelemetryError):
    """Failed to record or export metrics."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Metrics error: {reason}", code="TELEMETRY_METRICS")
        self.reason = reason


class TracingError(TelemetryError):
    """Failed to create or export traces."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Tracing error: {reason}", code="TELEMETRY_TRACING")
        self.reason = reason


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "EvalKitError",
    # Configuration
    "ConfigurationError",
    "InvalidConfigError",
    "MissingConfigError",
    # Storage
    "StorageError",
    "StorageConnectionError",
    "StorageQueryError",
    "StorageWriteError",
    "EntityNotFoundError",
    # Queue
    "QueueError",
    "QueueConnectionError",
    "QueueEnqueueError",
    "QueueConsumeError",
    "QueueAckError",
    # Evaluation
    "EvaluationError",
    "EvaluationTimeoutError",
    "StageEvaluationError",
    "LLMJudgeError",
    # Distributed
    "DistributedError",
    "WorkerError",
    "CoordinatorError",
    "MessageProcessingError",
    # External Services
    "ExternalServiceError",
    "CircuitOpenError",
    "RetryExhaustedError",
    # Telemetry
    "TelemetryError",
    "MetricsError",
    "TracingError",
]
