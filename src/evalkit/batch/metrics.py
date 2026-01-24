"""
OpenTelemetry Metrics for Batch Evaluation

Provides observability metrics for batch runs including:
- Test counts (total, passed, failed)
- Batch duration
- Per-test latency
- Score distribution

Uses the unified telemetry module (src.common.telemetry) for OTEL setup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.evalkit.common.telemetry import (
    EvalMetrics,
    get_eval_metrics,
    init_telemetry,
)

if TYPE_CHECKING:
    from CONTRACTS import BatchJob, Scorecard

logger = logging.getLogger(__name__)


class BatchMetrics:
    """
    OpenTelemetry metrics for batch evaluation.

    Wraps the unified EvalMetrics from the telemetry module.
    Gracefully handles missing OpenTelemetry dependencies.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize metrics.

        Args:
            enabled: Whether metrics are enabled
        """
        self.enabled = enabled
        self._eval_metrics: EvalMetrics | None = None

        if self.enabled:
            self._eval_metrics = get_eval_metrics()

    def record_test_result(
        self,
        scorecard: Scorecard,
        batch_id: str,
        tags: list[str] | None = None,
        client_type: str | None = None,
        client_version: str | None = None,
        eval_mode: str | None = None,
    ) -> None:
        """
        Record metrics for a single test result.

        Args:
            scorecard: The scorecard from evaluation
            batch_id: Batch identifier
            tags: Optional tags for filtering
            client_type: Client type for multi-client tracking
            client_version: Client version for multi-client tracking
            eval_mode: Evaluation mode
        """
        if not self.enabled or not self._eval_metrics:
            return

        self._eval_metrics.record_test_result(
            scorecard,
            batch_id,
            tags,
            client_type=client_type,
            client_version=client_version,
            eval_mode=eval_mode,
        )

    def record_batch_complete(
        self,
        batch_job: BatchJob,
        duration_seconds: float,
        client_type: str | None = None,
        client_version: str | None = None,
        eval_mode: str | None = None,
        pack_name: str | None = None,
    ) -> None:
        """
        Record metrics for batch completion.

        Args:
            batch_job: The completed batch job
            duration_seconds: Total batch duration in seconds
            client_type: Client type for multi-client tracking
            client_version: Client version for multi-client tracking
            eval_mode: Evaluation mode
            pack_name: Evaluation pack name (e.g., "nl2api", "rag")
        """
        if not self.enabled or not self._eval_metrics:
            return

        self._eval_metrics.record_batch_complete(
            batch_job,
            duration_seconds,
            client_type=client_type,
            client_version=client_version,
            eval_mode=eval_mode,
            pack_name=pack_name,
        )


def setup_console_exporter() -> None:
    """
    Configure OpenTelemetry to export metrics to console.

    Useful for debugging and local development.

    Note: This is a legacy function. Prefer using init_telemetry() from
    src.common.telemetry for new code.
    """
    logger.info("Console exporter requested - use init_telemetry() for OTEL setup")


def setup_otlp_exporter(endpoint: str | None = None) -> None:
    """
    Configure OpenTelemetry to export metrics via OTLP.

    Args:
        endpoint: OTLP collector endpoint. If None, uses OTEL_EXPORTER_OTLP_ENDPOINT env var.

    Note: This is a legacy function. Prefer using init_telemetry() from
    src.common.telemetry for new code.
    """
    init_telemetry(otlp_endpoint=endpoint)


# Global metrics instance
_metrics: BatchMetrics | None = None


def get_metrics() -> BatchMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = BatchMetrics()
    return _metrics
