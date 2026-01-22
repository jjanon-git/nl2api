"""Tests for log sanitization."""

from __future__ import annotations

import logging
import re

import pytest

from src.common.logging.sanitizer import (
    SanitizingFilter,
    configure_sanitized_logging,
    get_sanitized_logger,
)


class TestSanitizingFilter:
    """Test suite for SanitizingFilter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.filter = SanitizingFilter()

    def test_redacts_anthropic_api_key(self) -> None:
        """Test that Anthropic API keys are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Using API key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "sk-ant-api03" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_openai_api_key(self) -> None:
        """Test that OpenAI API keys are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="OpenAI key: sk-proj-abcdefghijklmnopqrstuvwxyz",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "sk-proj" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_generic_api_key(self) -> None:
        """Test that generic API key patterns are redacted."""
        test_cases = [
            "api_key=abcdefghij1234567890klmnop",
            "API-KEY: 'my-super-secret-key-12345'",
            'apikey="test_key_abcdefghijklmnop"',
        ]
        for msg in test_cases:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )
            self.filter.filter(record)
            assert "[REDACTED]" in record.msg, f"Failed to redact: {msg}"

    def test_redacts_password(self) -> None:
        """Test that passwords are redacted."""
        test_cases = [
            "password=mysecretpassword123",
            "Password: 'hunter2hunter2'",
            'pwd="verysecure123"',
        ]
        for msg in test_cases:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )
            self.filter.filter(record)
            assert "[REDACTED]" in record.msg, f"Failed to redact: {msg}"

    def test_redacts_bearer_token(self) -> None:
        """Test that Bearer tokens are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_postgres_connection_string(self) -> None:
        """Test that PostgreSQL connection strings with passwords are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Connecting to postgresql://user:password123@localhost:5432/db",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "password123" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_redis_url(self) -> None:
        """Test that Redis URLs with passwords are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Redis URL: redis://:mysecretpassword@localhost:6379",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "mysecretpassword" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_preserves_non_sensitive_data(self) -> None:
        """Test that non-sensitive data is preserved."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Processing query for Apple Inc. with RIC AAPL.O",
            args=(),
            exc_info=None,
        )
        original_msg = record.msg
        self.filter.filter(record)
        assert record.msg == original_msg

    def test_sanitizes_args_tuple(self) -> None:
        """Test that log arguments (tuple) are sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Key: %s, Value: %s",
            args=("api_key=secretkey1234567890abc", "normal_value"),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "secretkey1234567890abc" not in str(record.args)
        assert "[REDACTED]" in str(record.args)
        assert "normal_value" in str(record.args)

    def test_sanitizes_args_dict(self) -> None:
        """Test that log arguments (dict) are sanitized."""
        # Create record without args first, then set args manually
        # This avoids Python 3.14 LogRecord init validation issues
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Data: %(key)s",
            args=None,
            exc_info=None,
        )
        record.args = {"key": "password=secretpassword123"}
        self.filter.filter(record)
        assert "secretpassword123" not in str(record.args)
        assert "[REDACTED]" in str(record.args)

    def test_custom_redaction_placeholder(self) -> None:
        """Test custom redaction placeholder."""
        custom_filter = SanitizingFilter(redaction_placeholder="***HIDDEN***")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Key: sk-ant-api03-abcdefghijklmnop123456789",
            args=(),
            exc_info=None,
        )
        custom_filter.filter(record)
        assert "***HIDDEN***" in record.msg
        assert "[REDACTED]" not in record.msg

    def test_additional_patterns(self) -> None:
        """Test adding custom patterns."""
        custom_pattern = ("CUSTOM_ID", re.compile(r"custom-id-\d{10}"))
        custom_filter = SanitizingFilter(additional_patterns=[custom_pattern])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="ID: custom-id-1234567890",
            args=(),
            exc_info=None,
        )
        custom_filter.filter(record)
        assert "custom-id-1234567890" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_always_returns_true(self) -> None:
        """Test that filter always returns True (allows all records)."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = self.filter.filter(record)
        assert result is True


class TestGetSanitizedLogger:
    """Test suite for get_sanitized_logger."""

    def test_returns_logger_with_filter(self) -> None:
        """Test that returned logger has SanitizingFilter."""
        logger = get_sanitized_logger("test.sanitized")
        has_filter = any(isinstance(f, SanitizingFilter) for f in logger.filters)
        assert has_filter

    def test_does_not_duplicate_filters(self) -> None:
        """Test that calling twice doesn't add duplicate filters."""
        logger = get_sanitized_logger("test.no_dup")
        get_sanitized_logger("test.no_dup")  # Call again
        filter_count = sum(1 for f in logger.filters if isinstance(f, SanitizingFilter))
        assert filter_count == 1


class TestConfigureSanitizedLogging:
    """Test suite for configure_sanitized_logging."""

    def teardown_method(self) -> None:
        """Clean up root logger after test."""
        root = logging.getLogger()
        # Remove sanitizing filters we added
        root.filters = [f for f in root.filters if not isinstance(f, SanitizingFilter)]
        for handler in root.handlers:
            handler.filters = [
                f for f in handler.filters if not isinstance(f, SanitizingFilter)
            ]

    def test_adds_filter_to_root_logger(self) -> None:
        """Test that configure adds filter to root logger."""
        configure_sanitized_logging()
        root = logging.getLogger()
        has_filter = any(isinstance(f, SanitizingFilter) for f in root.filters)
        assert has_filter
