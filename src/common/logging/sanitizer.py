"""
Log Sanitization

Provides filters and utilities for redacting sensitive information from logs.
"""

from __future__ import annotations

import logging
import re
from re import Pattern
from typing import Any

# Patterns for sensitive data that should be redacted
SENSITIVE_PATTERNS: list[tuple[str, Pattern[str]]] = [
    # API keys (various formats)
    (
        "API_KEY",
        re.compile(r"(api[_-]?key|apikey)\s*[=:]\s*['\"]?[\w\-]{20,}['\"]?", re.IGNORECASE),
    ),
    (
        "AUTH_TOKEN",
        re.compile(r"(auth[_-]?token|bearer)\s*[=:]\s*['\"]?[\w\-\.]{20,}['\"]?", re.IGNORECASE),
    ),
    (
        "SECRET",
        re.compile(
            r"(secret|password|passwd|pwd)\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", re.IGNORECASE
        ),
    ),
    # Anthropic API key format: sk-ant-api03-...
    ("ANTHROPIC_KEY", re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,}", re.IGNORECASE)),
    # OpenAI API key format: sk-... or sk-proj-...
    ("OPENAI_KEY", re.compile(r"sk-[a-zA-Z0-9\-]{20,}", re.IGNORECASE)),
    # Azure connection strings
    (
        "AZURE_CONN",
        re.compile(
            r"DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[^;]+", re.IGNORECASE
        ),
    ),
    # PostgreSQL connection strings with password
    ("PG_CONN", re.compile(r"postgresql://[^:]+:[^@]+@", re.IGNORECASE)),
    # Redis URLs with password
    ("REDIS_URL", re.compile(r"redis://:[^@]+@", re.IGNORECASE)),
    # Bearer tokens in headers
    ("BEARER", re.compile(r"Bearer\s+[a-zA-Z0-9\-_\.]+", re.IGNORECASE)),
    # Basic auth headers (base64 encoded credentials)
    ("BASIC_AUTH", re.compile(r"Basic\s+[a-zA-Z0-9+/=]{20,}", re.IGNORECASE)),
    # X-API-Key headers
    ("X_API_KEY", re.compile(r"X-API-Key['\"]?\s*[=:]\s*['\"]?[\w\-]{16,}['\"]?", re.IGNORECASE)),
]

# Placeholder for redacted content
REDACTION_PLACEHOLDER = "[REDACTED]"


class SanitizingFilter(logging.Filter):
    """
    A logging filter that redacts sensitive information from log messages.

    Applies pattern matching to detect and redact:
    - API keys and tokens
    - Passwords and secrets
    - Connection strings with credentials
    - Bearer tokens

    Usage:
        logger = logging.getLogger(__name__)
        logger.addFilter(SanitizingFilter())
    """

    def __init__(
        self,
        name: str = "",
        additional_patterns: list[tuple[str, Pattern[str]]] | None = None,
        redaction_placeholder: str = REDACTION_PLACEHOLDER,
    ):
        """
        Initialize the sanitizing filter.

        Args:
            name: Filter name (passed to parent)
            additional_patterns: Extra patterns to redact beyond defaults
            redaction_placeholder: Text to replace sensitive data with
        """
        super().__init__(name)
        self._patterns = list(SENSITIVE_PATTERNS)
        if additional_patterns:
            self._patterns.extend(additional_patterns)
        self._placeholder = redaction_placeholder

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and sanitize the log record.

        Args:
            record: The log record to filter

        Returns:
            True (always allows the record, but sanitizes it)
        """
        # Sanitize the message
        if record.msg:
            record.msg = self._sanitize(str(record.msg))

        # Sanitize arguments if they're strings
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._sanitize_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._sanitize_value(arg) for arg in record.args)

        return True

    def _sanitize(self, text: str) -> str:
        """
        Sanitize a text string by redacting sensitive patterns.

        Args:
            text: The text to sanitize

        Returns:
            Sanitized text with sensitive data redacted
        """
        result = text
        for pattern_name, pattern in self._patterns:
            result = pattern.sub(f"{pattern_name}={self._placeholder}", result)
        return result

    def _sanitize_value(self, value: Any) -> Any:
        """
        Sanitize a single value (used for log arguments).

        Args:
            value: The value to sanitize

        Returns:
            Sanitized value if it's a string, otherwise unchanged
        """
        if isinstance(value, str):
            return self._sanitize(value)
        return value


def configure_sanitized_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    additional_patterns: list[tuple[str, Pattern[str]]] | None = None,
) -> None:
    """
    Configure the root logger with sanitization enabled.

    This adds the SanitizingFilter to the root logger, ensuring all
    log output is sanitized.

    Args:
        level: Logging level
        format_string: Log format string (uses default if not specified)
        additional_patterns: Extra patterns to redact
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, format=format_string)

    # Add sanitizing filter to root logger
    root_logger = logging.getLogger()
    sanitizing_filter = SanitizingFilter(additional_patterns=additional_patterns)
    root_logger.addFilter(sanitizing_filter)

    # Also add to all handlers
    for handler in root_logger.handlers:
        handler.addFilter(sanitizing_filter)


def get_sanitized_logger(name: str) -> logging.Logger:
    """
    Get a logger with sanitization filter attached.

    Use this when you want a specific logger to have sanitization
    without configuring it globally.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger with SanitizingFilter attached
    """
    logger = logging.getLogger(name)

    # Only add filter if not already present
    has_sanitizing_filter = any(isinstance(f, SanitizingFilter) for f in logger.filters)
    if not has_sanitizing_filter:
        logger.addFilter(SanitizingFilter())

    return logger
