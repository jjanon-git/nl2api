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
    """Configuration-related errors."""

    pass


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


# =============================================================================
# Queue Errors
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
    """Evaluation-related errors."""

    pass


# =============================================================================
# Distributed Processing Errors
# =============================================================================


class DistributedError(EvalKitError):
    """Distributed processing errors."""

    pass


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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "EvalKitError",
    # Configuration
    "ConfigurationError",
    # Storage
    "StorageError",
    "StorageConnectionError",
    "StorageQueryError",
    "StorageWriteError",
    # Queue
    "QueueError",
    "QueueConnectionError",
    "QueueEnqueueError",
    "QueueConsumeError",
    "QueueAckError",
    # Evaluation
    "EvaluationError",
    # Distributed
    "DistributedError",
    # External Services
    "ExternalServiceError",
    "CircuitOpenError",
]
