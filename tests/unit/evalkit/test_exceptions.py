"""
Unit tests for EvalKit exception hierarchy.

Tests the exception classes, inheritance, and error code handling.
"""

from __future__ import annotations

import pytest

from src.evalkit.exceptions import (
    CircuitOpenError,
    ConfigurationError,
    CoordinatorError,
    DistributedError,
    EntityNotFoundError,
    EvalKitError,
    EvaluationError,
    EvaluationTimeoutError,
    ExternalServiceError,
    InvalidConfigError,
    LLMJudgeError,
    MessageProcessingError,
    MetricsError,
    MissingConfigError,
    QueueAckError,
    QueueConnectionError,
    QueueConsumeError,
    QueueEnqueueError,
    QueueError,
    RetryExhaustedError,
    StageEvaluationError,
    StorageConnectionError,
    StorageError,
    StorageQueryError,
    StorageWriteError,
    TelemetryError,
    TracingError,
    WorkerError,
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
            InvalidConfigError,
            MissingConfigError,
            StorageError,
            StorageConnectionError,
            StorageQueryError,
            StorageWriteError,
            EntityNotFoundError,
            QueueError,
            QueueConnectionError,
            QueueEnqueueError,
            QueueConsumeError,
            QueueAckError,
            EvaluationError,
            EvaluationTimeoutError,
            StageEvaluationError,
            LLMJudgeError,
            DistributedError,
            WorkerError,
            CoordinatorError,
            MessageProcessingError,
            ExternalServiceError,
            CircuitOpenError,
            RetryExhaustedError,
            TelemetryError,
            MetricsError,
            TracingError,
        ]

        for exc_class in exception_classes:
            # Create a minimal instance
            try:
                if exc_class == InvalidConfigError:
                    instance = exc_class("field", "value", "reason")
                elif exc_class == MissingConfigError:
                    instance = exc_class("field")
                elif exc_class == StorageConnectionError:
                    instance = exc_class("postgres", "reason")
                elif exc_class == StorageQueryError:
                    instance = exc_class("operation", "reason")
                elif exc_class == StorageWriteError:
                    instance = exc_class("entity", "reason")
                elif exc_class == EntityNotFoundError:
                    instance = exc_class("TestCase", "123")
                elif exc_class == QueueConnectionError:
                    instance = exc_class("redis", "reason")
                elif exc_class == QueueEnqueueError:
                    instance = exc_class("batch-123", "reason")
                elif exc_class == QueueConsumeError:
                    instance = exc_class("worker-1", "reason")
                elif exc_class == QueueAckError:
                    instance = exc_class("msg-123", "reason")
                elif exc_class == EvaluationTimeoutError:
                    instance = exc_class("tc-123", 30.0)
                elif exc_class == StageEvaluationError:
                    instance = exc_class("syntax", "tc-123", "reason")
                elif exc_class == LLMJudgeError:
                    instance = exc_class("reason")
                elif exc_class == WorkerError:
                    instance = exc_class("worker-1", "reason")
                elif exc_class == CoordinatorError:
                    instance = exc_class("batch-123", "reason")
                elif exc_class == MessageProcessingError:
                    instance = exc_class("msg-123", "reason")
                elif exc_class == CircuitOpenError:
                    instance = exc_class("service")
                elif exc_class == RetryExhaustedError:
                    instance = exc_class("operation", 3, "last error")
                elif exc_class == MetricsError:
                    instance = exc_class("reason")
                elif exc_class == TracingError:
                    instance = exc_class("reason")
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
        assert issubclass(EntityNotFoundError, StorageError)

    def test_queue_exceptions_inherit_from_queue_error(self):
        """Test queue exception hierarchy."""
        assert issubclass(QueueConnectionError, QueueError)
        assert issubclass(QueueEnqueueError, QueueError)
        assert issubclass(QueueConsumeError, QueueError)
        assert issubclass(QueueAckError, QueueError)

    def test_evaluation_exceptions_inherit_from_evaluation_error(self):
        """Test evaluation exception hierarchy."""
        assert issubclass(EvaluationTimeoutError, EvaluationError)
        assert issubclass(StageEvaluationError, EvaluationError)
        assert issubclass(LLMJudgeError, EvaluationError)


class TestSpecificExceptions:
    """Tests for specific exception types."""

    def test_invalid_config_error(self):
        """Test InvalidConfigError stores field information."""
        error = InvalidConfigError("database_url", "not-a-url", "Must be a valid URL")
        assert error.field == "database_url"
        assert error.value == "not-a-url"
        assert error.reason == "Must be a valid URL"
        assert error.code == "CONFIG_INVALID"
        assert "database_url" in str(error)

    def test_missing_config_error(self):
        """Test MissingConfigError with hint."""
        error = MissingConfigError("api_key", hint="Set via ANTHROPIC_API_KEY env var")
        assert error.field == "api_key"
        assert error.code == "CONFIG_MISSING"
        assert "api_key" in str(error)
        assert "ANTHROPIC_API_KEY" in str(error)

    def test_entity_not_found_error(self):
        """Test EntityNotFoundError stores entity info."""
        error = EntityNotFoundError("TestCase", "tc-abc-123")
        assert error.entity_type == "TestCase"
        assert error.entity_id == "tc-abc-123"
        assert error.code == "STORAGE_NOT_FOUND"
        assert "TestCase" in str(error)
        assert "tc-abc-123" in str(error)

    def test_evaluation_timeout_error(self):
        """Test EvaluationTimeoutError stores timeout info."""
        error = EvaluationTimeoutError("tc-123", 30.0)
        assert error.test_case_id == "tc-123"
        assert error.timeout_seconds == 30.0
        assert error.code == "EVAL_TIMEOUT"

    def test_circuit_open_error(self):
        """Test CircuitOpenError stores retry_after."""
        error = CircuitOpenError("llm-provider", retry_after=60.0)
        assert error.service == "llm-provider"
        assert error.retry_after == 60.0
        assert error.code == "CIRCUIT_OPEN"
        assert "60.0" in str(error)

    def test_retry_exhausted_error(self):
        """Test RetryExhaustedError stores attempt info."""
        error = RetryExhaustedError("api_call", 5, "Connection timeout")
        assert error.operation == "api_call"
        assert error.attempts == 5
        assert error.last_error == "Connection timeout"
        assert error.code == "RETRY_EXHAUSTED"


class TestExceptionCatching:
    """Tests for catching exceptions at different hierarchy levels."""

    def test_catch_all_evalkit_errors(self):
        """Test that EvalKitError catches all derived exceptions."""
        exceptions_to_test = [
            StorageConnectionError("pg", "failed"),
            QueueEnqueueError("batch", "failed"),
            EvaluationTimeoutError("tc", 10.0),
            WorkerError("w1", "crashed"),
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
            EntityNotFoundError("TestCase", "123"),
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
