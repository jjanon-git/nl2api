"""
OpenTelemetry Metrics for Batch Evaluation

Provides observability metrics for batch runs including:
- Test counts (total, passed, failed)
- Batch duration
- Per-test latency
- Score distribution
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CONTRACTS import BatchJob, Scorecard

# Try to import OpenTelemetry, gracefully degrade if not available
_otel_available = False
try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    _otel_available = True
except ImportError:
    pass


logger = logging.getLogger(__name__)


class BatchMetrics:
    """
    OpenTelemetry metrics for batch evaluation.

    Gracefully handles missing OpenTelemetry dependencies.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize metrics.

        Args:
            enabled: Whether metrics are enabled (requires OTel to be installed)
        """
        self.enabled = enabled and _otel_available
        self._meter = None
        self._tests_total = None
        self._tests_passed = None
        self._tests_failed = None
        self._batch_duration = None
        self._test_duration = None
        self._test_score = None

        if self.enabled:
            self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Set up OpenTelemetry metrics."""
        if not _otel_available:
            return

        # Get or create meter
        self._meter = metrics.get_meter("evalplatform")

        # Counters
        self._tests_total = self._meter.create_counter(
            name="eval.batch.tests.total",
            description="Total tests in batch",
            unit="1",
        )
        self._tests_passed = self._meter.create_counter(
            name="eval.batch.tests.passed",
            description="Passed tests",
            unit="1",
        )
        self._tests_failed = self._meter.create_counter(
            name="eval.batch.tests.failed",
            description="Failed tests",
            unit="1",
        )

        # Histograms
        self._batch_duration = self._meter.create_histogram(
            name="eval.batch.duration_seconds",
            description="Batch duration",
            unit="s",
        )
        self._test_duration = self._meter.create_histogram(
            name="eval.test.duration_ms",
            description="Per-test latency",
            unit="ms",
        )
        self._test_score = self._meter.create_histogram(
            name="eval.test.score",
            description="Score distribution",
            unit="1",
        )

    def record_test_result(
        self,
        scorecard: "Scorecard",
        batch_id: str,
        tags: list[str] | None = None,
    ) -> None:
        """
        Record metrics for a single test result.

        Args:
            scorecard: The scorecard from evaluation
            batch_id: Batch identifier
            tags: Optional tags for filtering
        """
        if not self.enabled:
            return

        attributes = {
            "batch_id": batch_id,
        }
        if tags:
            attributes["tags"] = ",".join(tags)

        # Record counters
        self._tests_total.add(1, attributes)
        if scorecard.overall_passed:
            self._tests_passed.add(1, attributes)
        else:
            self._tests_failed.add(1, attributes)

        # Record histograms
        self._test_duration.record(scorecard.total_latency_ms, attributes)
        self._test_score.record(scorecard.overall_score, attributes)

    def record_batch_complete(
        self,
        batch_job: "BatchJob",
        duration_seconds: float,
    ) -> None:
        """
        Record metrics for batch completion.

        Args:
            batch_job: The completed batch job
            duration_seconds: Total batch duration in seconds
        """
        if not self.enabled:
            return

        attributes = {
            "batch_id": batch_job.batch_id,
        }
        if batch_job.tags:
            attributes["tags"] = ",".join(batch_job.tags)

        self._batch_duration.record(duration_seconds, attributes)


def setup_console_exporter() -> None:
    """
    Configure OpenTelemetry to export metrics to console.

    Useful for debugging and local development.
    """
    if not _otel_available:
        logger.warning("OpenTelemetry not installed, metrics disabled")
        return

    exporter = ConsoleMetricExporter()
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=10000)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)


def setup_otlp_exporter(endpoint: str | None = None) -> None:
    """
    Configure OpenTelemetry to export metrics via OTLP.

    Args:
        endpoint: OTLP collector endpoint. If None, uses OTEL_EXPORTER_OTLP_ENDPOINT env var.
    """
    if not _otel_available:
        logger.warning("OpenTelemetry not installed, metrics disabled")
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=10000)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
    except ImportError:
        logger.warning("OTLP exporter not installed, metrics disabled")


# Global metrics instance
_metrics: BatchMetrics | None = None


def get_metrics() -> BatchMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = BatchMetrics()
    return _metrics
