"""
NL2API and Evaluation Metrics.

Provides pre-defined metrics for:
- NL2API request processing
- Evaluation batch runs and accuracy
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.evalkit.common.telemetry.setup import get_meter, is_telemetry_enabled

if TYPE_CHECKING:
    from CONTRACTS import BatchJob, Scorecard
    from src.nl2api.observability.metrics import RequestMetrics

logger = logging.getLogger(__name__)


class NL2APIMetrics:
    """
    Metrics for NL2API request processing.

    Tracks:
    - Request counts by domain, status, cache hit
    - Request latency distribution
    - Token usage
    - LLM vs rule-based processing ratio
    """

    def __init__(self, meter_name: str = "nl2api"):
        """Initialize NL2API metrics."""
        self._meter = get_meter(meter_name)
        self._enabled = is_telemetry_enabled()

        # Request counter
        self._requests_total = self._meter.create_counter(
            name="requests",
            description="Total NL2API requests",
            unit="1",
        )

        # Latency histogram
        self._request_duration = self._meter.create_histogram(
            name="request_duration_ms",
            description="Request processing duration",
            unit="ms",
        )

        # Token usage
        self._tokens_total = self._meter.create_counter(
            name="tokens",
            description="Total tokens used",
            unit="1",
        )

        # Routing latency
        self._routing_duration = self._meter.create_histogram(
            name="routing_duration_ms",
            description="Query routing duration",
            unit="ms",
        )

        # Entity resolution latency
        self._entity_resolution_duration = self._meter.create_histogram(
            name="entity_resolution_duration_ms",
            description="Entity resolution duration",
            unit="ms",
        )

        # Context retrieval latency
        self._context_retrieval_duration = self._meter.create_histogram(
            name="context_retrieval_duration_ms",
            description="Context retrieval duration",
            unit="ms",
        )

        # Agent processing latency
        self._agent_duration = self._meter.create_histogram(
            name="agent_duration_ms",
            description="Agent processing duration",
            unit="ms",
        )

        # Tool calls generated
        self._tool_calls = self._meter.create_histogram(
            name="tool_calls_count",
            description="Number of tool calls generated per request",
            unit="1",
        )

    def record_request(self, metrics: RequestMetrics) -> None:
        """
        Record metrics from a completed request.

        Args:
            metrics: RequestMetrics from the completed request
        """
        if not self._enabled:
            return

        try:
            # Build attributes
            attrs: dict[str, Any] = {
                "domain": metrics.routing_domain or "unknown",
                "status": "success" if metrics.success else "error",
                "cached": str(metrics.routing_cached).lower(),
                "used_llm": str(metrics.agent_used_llm).lower(),
            }

            if metrics.error_type:
                attrs["error_type"] = metrics.error_type

            if metrics.needs_clarification:
                attrs["clarification"] = metrics.clarification_type or "unknown"

            # Record request count
            self._requests_total.add(1, attrs)

            # Record latencies
            if metrics.total_latency_ms > 0:
                self._request_duration.record(metrics.total_latency_ms, attrs)

            if metrics.routing_latency_ms > 0:
                self._routing_duration.record(metrics.routing_latency_ms, attrs)

            if metrics.entity_resolution_latency_ms > 0:
                self._entity_resolution_duration.record(metrics.entity_resolution_latency_ms, attrs)

            if metrics.context_latency_ms > 0:
                self._context_retrieval_duration.record(metrics.context_latency_ms, attrs)

            if metrics.agent_latency_ms > 0:
                self._agent_duration.record(metrics.agent_latency_ms, attrs)

            # Record tokens
            if metrics.total_tokens > 0:
                token_attrs = {
                    "domain": metrics.routing_domain or "unknown",
                    "type": "prompt",
                }
                self._tokens_total.add(metrics.agent_tokens_prompt, token_attrs)

                token_attrs["type"] = "completion"
                self._tokens_total.add(metrics.agent_tokens_completion, token_attrs)

            # Record tool calls
            self._tool_calls.record(metrics.tool_calls_count, attrs)

        except Exception as e:
            logger.warning(f"Failed to record NL2API metrics: {e}")


class EvalMetrics:
    """
    Metrics for evaluation runs.

    Tracks:
    - Test counts (total, passed, failed)
    - Batch duration
    - Per-test latency and scores
    - Accuracy trends over time
    """

    def __init__(self, meter_name: str = "nl2api"):
        """Initialize evaluation metrics."""
        self._meter = get_meter(meter_name)
        self._enabled = is_telemetry_enabled()

        # Test counters
        self._tests_total = self._meter.create_counter(
            name="eval_batch_tests",
            description="Total tests in batch evaluations",
            unit="1",
        )

        self._tests_passed = self._meter.create_counter(
            name="eval_batch_tests_passed",
            description="Passed tests in batch evaluations",
            unit="1",
        )

        self._tests_failed = self._meter.create_counter(
            name="eval_batch_tests_failed",
            description="Failed tests in batch evaluations",
            unit="1",
        )

        # Batch duration
        self._batch_duration = self._meter.create_histogram(
            name="eval_batch_duration_seconds",
            description="Batch evaluation duration",
            unit="s",
        )

        # Per-test metrics
        self._test_duration = self._meter.create_histogram(
            name="eval_test_duration_ms",
            description="Per-test evaluation latency",
            unit="ms",
        )

        self._test_score = self._meter.create_histogram(
            name="eval_test_score",
            description="Test score distribution (0.0 - 1.0)",
            unit="1",
        )

        # Stage-specific pass rates
        self._stage_passed = self._meter.create_counter(
            name="eval_stage_passed",
            description="Tests passed per evaluation stage",
            unit="1",
        )

        self._stage_failed = self._meter.create_counter(
            name="eval_stage_failed",
            description="Tests failed per evaluation stage",
            unit="1",
        )

        # Token usage metrics
        self._tokens_total = self._meter.create_counter(
            name="eval_tokens",
            description="Total tokens used in evaluations",
            unit="1",
        )

        # Cost metrics
        self._cost_total = self._meter.create_counter(
            name="eval_cost_usd",
            description="Estimated cost of evaluations in USD (scaled by 1M for precision)",
            unit="1",
        )

        # Worker metrics (for infrastructure dashboard)
        self._worker_active = self._meter.create_up_down_counter(
            name="eval_worker_active",
            description="Number of active evaluation workers",
            unit="1",
        )

        self._worker_tasks_processed = self._meter.create_counter(
            name="eval_worker_tasks_processed",
            description="Total tasks processed by workers",
            unit="1",
        )

        self._worker_tasks_failed = self._meter.create_counter(
            name="eval_worker_tasks_failed",
            description="Total tasks failed by workers",
            unit="1",
        )

        self._worker_task_duration = self._meter.create_histogram(
            name="eval_worker_task_duration_ms",
            description="Worker task processing duration",
            unit="ms",
        )

        # Queue metrics (for infrastructure dashboard)
        self._queue_enqueued = self._meter.create_counter(
            name="eval_queue_enqueued",
            description="Total items enqueued",
            unit="1",
        )

        self._queue_acked = self._meter.create_counter(
            name="eval_queue_acked",
            description="Total items acknowledged",
            unit="1",
        )

        self._queue_nacked = self._meter.create_counter(
            name="eval_queue_nacked",
            description="Total items not acknowledged (requeued or DLQ)",
            unit="1",
        )

        self._queue_dlq = self._meter.create_counter(
            name="eval_queue_dlq",
            description="Total items sent to dead letter queue",
            unit="1",
        )

    def record_test_result(
        self,
        scorecard: Scorecard,
        batch_id: str,
        tags: list[str] | None = None,
        client_type: str | None = None,
        client_version: str | None = None,
        eval_mode: str | None = None,
        source_type: str | None = None,
    ) -> None:
        """
        Record metrics for a single test result.

        Args:
            scorecard: The scorecard from evaluation
            batch_id: Batch identifier for grouping
            tags: Optional tags for filtering
            client_type: Client type for multi-client tracking
            client_version: Client version for multi-client tracking
            eval_mode: Evaluation mode
            source_type: Data source type (customer, sme, synthetic, hybrid)
        """
        if not self._enabled:
            return

        try:
            # Get pack_name from scorecard (defaults to "nl2api" for backwards compat)
            pack_name = getattr(scorecard, "pack_name", None) or "nl2api"

            attrs: dict[str, Any] = {"batch_id": batch_id, "pack_name": pack_name}
            if tags:
                attrs["tags"] = ",".join(tags)

            # Add client dimensions for multi-client tracking
            if client_type:
                attrs["client_type"] = client_type
            if client_version:
                attrs["client_version"] = client_version
            if eval_mode:
                attrs["eval_mode"] = eval_mode
            # Always set source_type to avoid "Value" in Grafana groupings
            attrs["source_type"] = source_type or "unknown"

            # Record test counts
            self._tests_total.add(1, attrs)

            if scorecard.overall_passed:
                self._tests_passed.add(1, attrs)
            else:
                self._tests_failed.add(1, attrs)

            # Record latency and score
            self._test_duration.record(scorecard.total_latency_ms, attrs)
            self._test_score.record(scorecard.overall_score, attrs)

            # Record per-stage results dynamically from stage_results
            # First try the new generic stage_results dict
            all_stage_results = scorecard.get_all_stage_results()
            for stage_name, stage_result in all_stage_results.items():
                if stage_result is None:
                    continue
                stage_attrs = {**attrs, "stage": stage_name}
                if stage_result.passed:
                    self._stage_passed.add(1, stage_attrs)
                else:
                    self._stage_failed.add(1, stage_attrs)

            # Record token usage
            if hasattr(scorecard, "input_tokens") and scorecard.input_tokens:
                token_attrs = {**attrs, "token_type": "input"}
                self._tokens_total.add(scorecard.input_tokens, token_attrs)

            if hasattr(scorecard, "output_tokens") and scorecard.output_tokens:
                token_attrs = {**attrs, "token_type": "output"}
                self._tokens_total.add(scorecard.output_tokens, token_attrs)

            # Record cost (scaled by 1M for counter precision with small values)
            if hasattr(scorecard, "estimated_cost_usd") and scorecard.estimated_cost_usd:
                # Scale to micro-dollars (USD * 1,000,000) for counter precision
                cost_micro_usd = int(scorecard.estimated_cost_usd * 1_000_000)
                if cost_micro_usd > 0:
                    self._cost_total.add(cost_micro_usd, attrs)

        except Exception as e:
            logger.warning(f"Failed to record eval test metrics: {e}")

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
            duration_seconds: Total batch duration
            client_type: Client type for multi-client tracking
            client_version: Client version for multi-client tracking
            eval_mode: Evaluation mode
            pack_name: Evaluation pack name (e.g., "nl2api", "rag")
        """
        if not self._enabled:
            return

        try:
            # Default to nl2api for backwards compatibility
            pack = pack_name or "nl2api"
            attrs: dict[str, Any] = {"batch_id": batch_job.batch_id, "pack_name": pack}
            if batch_job.tags:
                attrs["tags"] = ",".join(batch_job.tags)

            # Add client dimensions for multi-client tracking
            if client_type:
                attrs["client_type"] = client_type
            if client_version:
                attrs["client_version"] = client_version
            if eval_mode:
                attrs["eval_mode"] = eval_mode

            self._batch_duration.record(duration_seconds, attrs)

        except Exception as e:
            logger.warning(f"Failed to record batch completion metrics: {e}")

    def record_worker_status(
        self,
        worker_id: str,
        active: bool,
        tasks_processed: int = 0,
        tasks_failed: int = 0,
        task_duration_ms: float | None = None,
    ) -> None:
        """
        Record worker status metrics.

        Args:
            worker_id: Worker identifier
            active: Whether worker is active (1) or inactive (0)
            tasks_processed: Number of tasks successfully processed
            tasks_failed: Number of tasks that failed
            task_duration_ms: Duration of last task in milliseconds
        """
        if not self._enabled:
            return

        try:
            attrs = {"worker_id": worker_id}

            # Update active worker count
            self._worker_active.add(1 if active else -1, attrs)

            # Record task counts
            if tasks_processed > 0:
                self._worker_tasks_processed.add(tasks_processed, {"status": "success"})

            if tasks_failed > 0:
                self._worker_tasks_failed.add(tasks_failed, {})

            # Record task duration
            if task_duration_ms is not None:
                self._worker_task_duration.record(task_duration_ms, attrs)

        except Exception as e:
            logger.warning(f"Failed to record worker metrics: {e}")

    def record_queue_operation(
        self,
        operation: str,
        count: int = 1,
        action: str | None = None,
    ) -> None:
        """
        Record queue operation metrics.

        Args:
            operation: One of 'enqueue', 'ack', 'nack', 'dlq'
            count: Number of items
            action: For nack, the action taken ('requeue' or 'dlq')
        """
        if not self._enabled:
            return

        try:
            if operation == "enqueue":
                self._queue_enqueued.add(count, {})
            elif operation == "ack":
                self._queue_acked.add(count, {})
            elif operation == "nack":
                attrs = {"action": action} if action else {}
                self._queue_nacked.add(count, attrs)
            elif operation == "dlq":
                self._queue_dlq.add(count, {})

        except Exception as e:
            logger.warning(f"Failed to record queue metrics: {e}")


class AccuracyMetrics:
    """
    Metrics for accuracy testing runs.

    Tracks:
    - Accuracy test counts (total, correct, failed)
    - Accuracy rate by tier, category, domain
    - Per-query latency and confidence
    - Low confidence rate (would trigger clarification)
    """

    def __init__(self, meter_name: str = "nl2api"):
        """Initialize accuracy metrics."""
        self._meter = get_meter(meter_name)
        self._enabled = is_telemetry_enabled()

        # Test counters
        self._tests_total = self._meter.create_counter(
            name="accuracy_tests",
            description="Total accuracy test queries",
            unit="1",
        )

        self._tests_correct = self._meter.create_counter(
            name="accuracy_tests_correct",
            description="Correctly predicted queries",
            unit="1",
        )

        self._tests_incorrect = self._meter.create_counter(
            name="accuracy_tests_incorrect",
            description="Incorrectly predicted queries",
            unit="1",
        )

        self._tests_error = self._meter.create_counter(
            name="accuracy_tests_error",
            description="Queries that resulted in errors",
            unit="1",
        )

        self._low_confidence = self._meter.create_counter(
            name="accuracy_tests_low_confidence",
            description="Queries with low confidence (would trigger clarification)",
            unit="1",
        )

        # Batch metrics
        self._batch_duration = self._meter.create_histogram(
            name="accuracy_batch_duration_seconds",
            description="Accuracy batch evaluation duration",
            unit="s",
        )

        self._batch_accuracy = self._meter.create_histogram(
            name="accuracy_batch_rate",
            description="Accuracy rate per batch (0.0 - 1.0)",
            unit="1",
        )

        # Per-query metrics
        self._query_latency = self._meter.create_histogram(
            name="accuracy_query_latency_ms",
            description="Per-query evaluation latency",
            unit="ms",
        )

        self._query_confidence = self._meter.create_histogram(
            name="accuracy_query_confidence",
            description="Confidence score distribution (0.0 - 1.0)",
            unit="1",
        )

    def record_query_result(
        self,
        correct: bool,
        confidence: float,
        latency_ms: int,
        expected_domain: str,
        predicted_domain: str,
        tier: str = "",
        category: str = "",
        error: bool = False,
    ) -> None:
        """
        Record metrics for a single accuracy test query.

        Args:
            correct: Whether prediction was correct
            confidence: Model confidence (0.0 - 1.0)
            latency_ms: Query evaluation latency
            expected_domain: Expected domain
            predicted_domain: Predicted domain
            tier: Test tier (tier1, tier2, tier3)
            category: Query category (lookups, temporal, etc.)
            error: Whether an error occurred
        """
        if not self._enabled:
            return

        try:
            attrs: dict[str, Any] = {
                "expected_domain": expected_domain,
                "predicted_domain": predicted_domain,
            }
            if tier:
                attrs["tier"] = tier
            if category:
                attrs["category"] = category

            # Record counts
            self._tests_total.add(1, attrs)

            if error:
                self._tests_error.add(1, attrs)
            elif correct:
                self._tests_correct.add(1, attrs)
            else:
                self._tests_incorrect.add(1, attrs)

            # Low confidence threshold (would trigger clarification)
            if confidence <= 0.5:
                self._low_confidence.add(1, attrs)

            # Record latency and confidence
            self._query_latency.record(latency_ms, attrs)
            self._query_confidence.record(confidence, attrs)

        except Exception as e:
            logger.warning(f"Failed to record accuracy query metrics: {e}")

    def record_batch_complete(
        self,
        total_count: int,
        correct_count: int,
        duration_seconds: float,
        tier: str = "",
        model: str = "",
    ) -> None:
        """
        Record metrics for accuracy batch completion.

        Args:
            total_count: Total queries in batch
            correct_count: Correct predictions
            duration_seconds: Total batch duration
            tier: Test tier
            model: Model used for evaluation
        """
        if not self._enabled:
            return

        try:
            attrs: dict[str, Any] = {}
            if tier:
                attrs["tier"] = tier
            if model:
                attrs["model"] = model

            self._batch_duration.record(duration_seconds, attrs)

            if total_count > 0:
                accuracy_rate = correct_count / total_count
                self._batch_accuracy.record(accuracy_rate, attrs)

        except Exception as e:
            logger.warning(f"Failed to record accuracy batch metrics: {e}")


class RegressionAlertMetrics:
    """
    Metrics for regression detection alerts.

    Tracks:
    - Alert counts by severity and metric
    - Acknowledgment counts
    - Alert response times
    """

    def __init__(self, meter_name: str = "nl2api"):
        """Initialize regression alert metrics."""
        self._meter = get_meter(meter_name)
        self._enabled = is_telemetry_enabled()

        # Alert counters
        self._alerts_total = self._meter.create_counter(
            name="regression_alerts",
            description="Total regression alerts created",
            unit="1",
        )

        self._alerts_acknowledged = self._meter.create_counter(
            name="regression_alerts_acknowledged",
            description="Total regression alerts acknowledged",
            unit="1",
        )

        # Alert delta (magnitude of regression)
        self._alert_delta = self._meter.create_histogram(
            name="regression_alert_delta_pct",
            description="Regression delta percentage distribution",
            unit="1",
        )

    def record_alert_created(
        self,
        severity: str,
        metric_name: str,
        delta_pct: float | None = None,
        client_type: str | None = None,
    ) -> None:
        """
        Record metrics for a new regression alert.

        Args:
            severity: Alert severity (warning, critical)
            metric_name: Name of the metric that regressed
            delta_pct: Percentage change (optional)
            client_type: Client type (optional)
        """
        if not self._enabled:
            return

        try:
            attrs: dict[str, Any] = {
                "severity": severity,
                "metric_name": metric_name,
            }
            if client_type:
                attrs["client_type"] = client_type

            self._alerts_total.add(1, attrs)

            if delta_pct is not None:
                self._alert_delta.record(abs(delta_pct), attrs)

        except Exception as e:
            logger.warning(f"Failed to record regression alert metrics: {e}")

    def record_alert_acknowledged(
        self,
        severity: str,
        metric_name: str,
    ) -> None:
        """
        Record metrics for an acknowledged alert.

        Args:
            severity: Alert severity
            metric_name: Name of the metric
        """
        if not self._enabled:
            return

        try:
            attrs: dict[str, Any] = {
                "severity": severity,
                "metric_name": metric_name,
            }

            self._alerts_acknowledged.add(1, attrs)

        except Exception as e:
            logger.warning(f"Failed to record alert acknowledgment metrics: {e}")


# === Global Instances ===

_nl2api_metrics: NL2APIMetrics | None = None
_eval_metrics: EvalMetrics | None = None
_accuracy_metrics: AccuracyMetrics | None = None
_regression_alert_metrics: RegressionAlertMetrics | None = None


def get_nl2api_metrics() -> NL2APIMetrics:
    """Get the global NL2API metrics instance."""
    global _nl2api_metrics
    if _nl2api_metrics is None:
        _nl2api_metrics = NL2APIMetrics()
    return _nl2api_metrics


def get_eval_metrics() -> EvalMetrics:
    """Get the global evaluation metrics instance."""
    global _eval_metrics
    if _eval_metrics is None:
        _eval_metrics = EvalMetrics()
    return _eval_metrics


def get_accuracy_metrics() -> AccuracyMetrics:
    """Get the global accuracy metrics instance."""
    global _accuracy_metrics
    if _accuracy_metrics is None:
        _accuracy_metrics = AccuracyMetrics()
    return _accuracy_metrics


def get_regression_alert_metrics() -> RegressionAlertMetrics:
    """Get the global regression alert metrics instance."""
    global _regression_alert_metrics
    if _regression_alert_metrics is None:
        _regression_alert_metrics = RegressionAlertMetrics()
    return _regression_alert_metrics
