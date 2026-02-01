"""
Unit tests for EvalKit exception hierarchy.

Tests the exception classes, inheritance, and error code handling.
"""

from __future__ import annotations

import pytest

from src.evalkit.exceptions import (
    CircuitOpenError,
    ConfigurationError,
    DistributedError,
    EvalKitError,
    EvaluationError,
    ExternalServiceError,
    QueueAckError,
    QueueConnectionError,
    QueueConsumeError,
    QueueEnqueueError,
    QueueError,
    StorageConnectionError,
    StorageError,
    StorageQueryError,
    StorageWriteError,
)


class TestEvalKitErrorBase:
    """Tests for the base EvalKitError class."""

    def test_base_error_with_message_only(self):
        """Test creating error with just a message."""
        error = EvalKitError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code is None

    def test_base_error_with_code(self):
        """Test creating error with message and code."""
        error = EvalKitError("Something went wrong", code="ERR_001")
        assert str(error) == "[ERR_001] Something went wrong"
        assert error.code == "ERR_001"

    def test_base_error_is_exception(self):
        """Test that EvalKitError inherits from Exception."""
        error = EvalKitError("test")
        assert isinstance(error, Exception)

    def test_base_error_can_be_raised_and_caught(self):
        """Test that the error can be raised and caught."""
        with pytest.raises(EvalKitError) as exc_info:
            raise EvalKitError("test error")
        assert "test error" in str(exc_info.value)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_evalkit_error(self):
        """Test that all custom exceptions inherit from EvalKitError."""
        exception_classes = [
            ConfigurationError,
            StorageError,
            StorageConnectionError,
            StorageQueryError,
            StorageWriteError,
            QueueError,
            QueueConnectionError,
            QueueEnqueueError,
            QueueConsumeError,
            QueueAckError,
            EvaluationError,
            DistributedError,
            ExternalServiceError,
            CircuitOpenError,
        ]

        for exc_class in exception_classes:
            # Create a minimal instance
            try:
                if exc_class == StorageConnectionError:
                    instance = exc_class("postgres", "reason")
                elif exc_class == StorageQueryError:
                    instance = exc_class("operation", "reason")
                elif exc_class == StorageWriteError:
                    instance = exc_class("entity", "reason")
                elif exc_class == QueueConnectionError:
                    instance = exc_class("redis", "reason")
                elif exc_class == QueueEnqueueError:
                    instance = exc_class("batch-123", "reason")
                elif exc_class == QueueConsumeError:
                    instance = exc_class("worker-1", "reason")
                elif exc_class == QueueAckError:
                    instance = exc_class("msg-123", "reason")
                elif exc_class == CircuitOpenError:
                    instance = exc_class("service")
                else:
                    instance = exc_class("test message")
            except TypeError:
                pytest.fail(f"{exc_class.__name__} could not be instantiated")

            assert isinstance(instance, EvalKitError), (
                f"{exc_class.__name__} does not inherit from EvalKitError"
            )

    def test_storage_exceptions_inherit_from_storage_error(self):
        """Test storage exception hierarchy."""
        assert issubclass(StorageConnectionError, StorageError)
        assert issubclass(StorageQueryError, StorageError)
        assert issubclass(StorageWriteError, StorageError)

    def test_queue_exceptions_inherit_from_queue_error(self):
        """Test queue exception hierarchy."""
        assert issubclass(QueueConnectionError, QueueError)
        assert issubclass(QueueEnqueueError, QueueError)
        assert issubclass(QueueConsumeError, QueueError)
        assert issubclass(QueueAckError, QueueError)


class TestSpecificExceptions:
    """Tests for specific exception types."""

    def test_circuit_open_error(self):
        """Test CircuitOpenError stores retry_after."""
        error = CircuitOpenError("llm-provider", retry_after=60.0)
        assert error.service == "llm-provider"
        assert error.retry_after == 60.0
        assert error.code == "CIRCUIT_OPEN"
        assert "60.0" in str(error)

    def test_storage_query_error(self):
        """Test StorageQueryError stores operation info."""
        error = StorageQueryError("select", "connection failed")
        assert error.operation == "select"
        assert error.reason == "connection failed"
        assert error.code == "STORAGE_QUERY"

    def test_storage_write_error(self):
        """Test StorageWriteError stores entity info."""
        error = StorageWriteError("scorecard", "disk full")
        assert error.entity == "scorecard"
        assert error.reason == "disk full"
        assert error.code == "STORAGE_WRITE"

    def test_queue_enqueue_error(self):
        """Test QueueEnqueueError stores batch info."""
        error = QueueEnqueueError("batch-123", "queue full")
        assert error.batch_id == "batch-123"
        assert error.reason == "queue full"
        assert error.code == "QUEUE_ENQUEUE"


class TestExceptionCatching:
    """Tests for catching exceptions at different hierarchy levels."""

    def test_catch_all_evalkit_errors(self):
        """Test that EvalKitError catches all derived exceptions."""
        exceptions_to_test = [
            StorageConnectionError("pg", "failed"),
            QueueEnqueueError("batch", "failed"),
            CircuitOpenError("svc"),
        ]

        for exc in exceptions_to_test:
            caught = False
            try:
                raise exc
            except EvalKitError:
                caught = True
            assert caught, f"Failed to catch {type(exc).__name__} as EvalKitError"

    def test_catch_storage_errors(self):
        """Test that StorageError catches all storage-related exceptions."""
        storage_exceptions = [
            StorageConnectionError("pg", "failed"),
            StorageQueryError("select", "failed"),
            StorageWriteError("scorecard", "failed"),
        ]

        for exc in storage_exceptions:
            caught = False
            try:
                raise exc
            except StorageError:
                caught = True
            assert caught, f"Failed to catch {type(exc).__name__} as StorageError"

    def test_specific_exception_not_caught_by_sibling(self):
        """Test that sibling exceptions don't catch each other."""
        with pytest.raises(StorageError):
            try:
                raise StorageConnectionError("pg", "failed")
            except QueueError:
                pytest.fail("QueueError should not catch StorageConnectionError")
                raise


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_queue_exceptions_importable_from_protocol(self):
        """Test that queue exceptions can still be imported from protocol."""
        from src.evalkit.distributed.queue.protocol import (
            QueueError,
        )

        # Should be the same classes
        from src.evalkit.exceptions import QueueError as QueueErrorFromExceptions

        assert QueueError is QueueErrorFromExceptions

    def test_circuit_open_error_importable_from_circuit_breaker(self):
        """Test that CircuitOpenError can still be imported from circuit_breaker."""
        from src.evalkit.common.resilience.circuit_breaker import CircuitOpenError

        # Should be the same class
        from src.evalkit.exceptions import CircuitOpenError as CircuitOpenFromExceptions

        assert CircuitOpenError is CircuitOpenFromExceptions
