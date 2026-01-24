"""
Metrics emission backends for NL2API.

Supports multiple emission backends:
- LoggingEmitter: Structured JSON logging
- FileEmitter: JSONL file output
- OTELEmitter: OpenTelemetry metrics export
- CompositeEmitter: Multiple backends simultaneously

Usage:
    from src.nl2api.observability import configure_emitter, emit_metrics, FileEmitter

    # Configure file emission
    configure_emitter(FileEmitter("/var/log/nl2api/metrics.jsonl"))

    # Configure OTEL emission
    configure_emitter(OTELEmitter())

    # Emit metrics
    await emit_metrics(request_metrics)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evalkit.common.telemetry.metrics import NL2APIMetrics
    from src.nl2api.observability.metrics import RequestMetrics

logger = logging.getLogger(__name__)

# Dedicated logger for metrics (can be configured separately)
metrics_logger = logging.getLogger("nl2api.metrics")


class MetricsEmitter(ABC):
    """
    Base class for metrics emission backends.

    Subclasses implement the `emit` method to send metrics
    to their specific backend.
    """

    @abstractmethod
    async def emit(self, metrics: RequestMetrics) -> None:
        """
        Emit metrics to the backend.

        Args:
            metrics: RequestMetrics to emit

        Note:
            Implementations should be resilient to failures
            and not raise exceptions that would affect request processing.
        """
        ...

    async def close(self) -> None:
        """Close the emitter and release resources."""
        pass


class LoggingEmitter(MetricsEmitter):
    """
    Emit metrics to Python logging as structured JSON.

    This is the default emitter. Metrics are logged at INFO level
    to a dedicated logger ("nl2api.metrics") for easy filtering.
    """

    def __init__(
        self,
        logger_name: str = "nl2api.metrics",
        include_full_json: bool = True,
    ):
        """
        Initialize logging emitter.

        Args:
            logger_name: Name of logger to use
            include_full_json: Whether to include full JSON in logs
        """
        self._logger = logging.getLogger(logger_name)
        self._include_full_json = include_full_json

    async def emit(self, metrics: RequestMetrics) -> None:
        """Emit metrics to logging."""
        try:
            # Summary log for quick debugging
            self._logger.info(f"Request: {metrics.to_log_summary()}")

            # Full JSON for log aggregation/analysis
            if self._include_full_json:
                self._logger.debug(
                    "request_metrics_json",
                    extra={"metrics": metrics.to_dict()},
                )
        except Exception as e:
            logger.warning(f"LoggingEmitter failed: {e}")


class FileEmitter(MetricsEmitter):
    """
    Emit metrics to a JSONL (JSON Lines) file.

    Each metric is written as a single JSON line, making it easy
    to process with standard tools (jq, pandas, etc.).
    """

    def __init__(
        self,
        file_path: str | Path,
        create_dirs: bool = True,
        buffer_size: int = 10,
        flush_interval_seconds: float = 5.0,
    ):
        """
        Initialize file emitter.

        Args:
            file_path: Path to output JSONL file
            create_dirs: Create parent directories if needed
            buffer_size: Number of metrics to buffer before writing
            flush_interval_seconds: Max time between flushes
        """
        self._file_path = Path(file_path)
        self._buffer: list[str] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._lock = asyncio.Lock()

        if create_dirs:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

    async def emit(self, metrics: RequestMetrics) -> None:
        """Emit metrics to file."""
        try:
            json_line = metrics.to_json()

            async with self._lock:
                self._buffer.append(json_line)

                if len(self._buffer) >= self._buffer_size:
                    await self._flush()
        except Exception as e:
            logger.warning(f"FileEmitter failed: {e}")

    async def _flush(self) -> None:
        """Flush buffer to file."""
        if not self._buffer:
            return

        try:
            with open(self._file_path, "a") as f:
                for line in self._buffer:
                    f.write(line + "\n")
            self._buffer.clear()
        except Exception as e:
            logger.warning(f"FileEmitter flush failed: {e}")

    async def close(self) -> None:
        """Flush remaining buffer and close."""
        async with self._lock:
            await self._flush()


class CompositeEmitter(MetricsEmitter):
    """
    Emit metrics to multiple backends simultaneously.

    Failures in one backend don't affect others.
    """

    def __init__(self, emitters: list[MetricsEmitter]):
        """
        Initialize composite emitter.

        Args:
            emitters: List of emitters to send metrics to
        """
        self._emitters = emitters

    async def emit(self, metrics: RequestMetrics) -> None:
        """Emit to all configured backends."""
        # Fire all emitters concurrently
        tasks = [emitter.emit(metrics) for emitter in self._emitters]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                emitter_type = type(self._emitters[i]).__name__
                logger.warning(f"CompositeEmitter: {emitter_type} failed: {result}")

    async def close(self) -> None:
        """Close all emitters."""
        for emitter in self._emitters:
            try:
                await emitter.close()
            except Exception as e:
                logger.warning(f"Error closing emitter: {e}")


class OTELEmitter(MetricsEmitter):
    """
    Emit metrics to OpenTelemetry.

    Uses the unified telemetry module to send metrics to the configured
    OTEL collector (Jaeger for traces, Prometheus for metrics).
    """

    def __init__(self) -> None:
        """Initialize OTEL emitter."""
        self._nl2api_metrics = None

    def _get_metrics(self) -> NL2APIMetrics:
        """Lazy load NL2API metrics to avoid import cycles."""
        if self._nl2api_metrics is None:
            from src.evalkit.common.telemetry import get_nl2api_metrics

            self._nl2api_metrics = get_nl2api_metrics()
        return self._nl2api_metrics

    async def emit(self, metrics: RequestMetrics) -> None:
        """Emit metrics to OpenTelemetry."""
        try:
            nl2api_metrics = self._get_metrics()
            nl2api_metrics.record_request(metrics)
        except Exception as e:
            logger.warning(f"OTELEmitter failed: {e}")


class NullEmitter(MetricsEmitter):
    """No-op emitter for testing or disabled metrics."""

    async def emit(self, metrics: RequestMetrics) -> None:
        """Do nothing."""
        pass


# === Global Emitter Configuration ===

_emitter: MetricsEmitter | None = None
_metrics_enabled: bool = True


def configure_emitter(emitter: MetricsEmitter) -> None:
    """
    Configure the global metrics emitter.

    Args:
        emitter: Emitter instance to use globally

    Example:
        configure_emitter(CompositeEmitter([
            LoggingEmitter(),
            FileEmitter("/var/log/nl2api/metrics.jsonl"),
        ]))
    """
    global _emitter
    _emitter = emitter
    logger.info(f"Configured metrics emitter: {type(emitter).__name__}")


def set_metrics_enabled(enabled: bool) -> None:
    """Enable or disable metrics emission."""
    global _metrics_enabled
    _metrics_enabled = enabled


def get_emitter() -> MetricsEmitter:
    """
    Get the configured emitter.

    Returns LoggingEmitter if none configured.
    """
    global _emitter
    if _emitter is None:
        _emitter = LoggingEmitter()
    return _emitter


async def emit_metrics(metrics: RequestMetrics) -> None:
    """
    Emit metrics using the configured emitter.

    This is the primary entry point for emitting metrics.
    It's designed to be fire-and-forget - errors are logged
    but not raised.

    Args:
        metrics: RequestMetrics to emit
    """
    if not _metrics_enabled:
        return

    try:
        emitter = get_emitter()
        await emitter.emit(metrics)
    except Exception as e:
        # Never let metrics emission fail the request
        logger.warning(f"Failed to emit metrics: {e}")


def create_emitter_from_config(
    log_enabled: bool = True,
    file_path: str | None = None,
    otel_enabled: bool = False,
    include_full_json: bool = True,
) -> MetricsEmitter:
    """
    Create an emitter from configuration options.

    Args:
        log_enabled: Enable logging emitter
        file_path: Path for file emitter (None = disabled)
        otel_enabled: Enable OpenTelemetry emitter
        include_full_json: Include full JSON in log output

    Returns:
        Configured MetricsEmitter
    """
    emitters: list[MetricsEmitter] = []

    if log_enabled:
        emitters.append(LoggingEmitter(include_full_json=include_full_json))

    if file_path:
        emitters.append(FileEmitter(file_path))

    if otel_enabled:
        emitters.append(OTELEmitter())

    if not emitters:
        return NullEmitter()

    if len(emitters) == 1:
        return emitters[0]

    return CompositeEmitter(emitters)
