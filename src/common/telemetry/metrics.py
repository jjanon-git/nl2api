"""
NL2API and Evaluation Metrics.

Provides pre-defined metrics for:
- NL2API request processing
- Evaluation batch runs and accuracy
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from src.common.telemetry.setup import get_meter, is_telemetry_enabled

if TYPE_CHECKING:
    from src.nl2api.observability.metrics import RequestMetrics
    from CONTRACTS import Scorecard, BatchJob

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
            name="nl2api_requests_total",
            description="Total NL2API requests",
            unit="1",
        )

        # Latency histogram
        self._request_duration = self._meter.create_histogram(
            name="nl2api_request_duration_ms",
            description="Request processing duration",
            unit="ms",
        )

        # Token usage
        self._tokens_total = self._meter.create_counter(
            name="nl2api_tokens_total",
            description="Total tokens used",
            unit="1",
        )

        # Routing latency
        self._routing_duration = self._meter.create_histogram(
            name="nl2api_routing_duration_ms",
            description="Query routing duration",
            unit="ms",
        )

        # Entity resolution latency
        self._entity_resolution_duration = self._meter.create_histogram(
            name="nl2api_entity_resolution_duration_ms",
            description="Entity resolution duration",
            unit="ms",
        )

        # Context retrieval latency
        self._context_retrieval_duration = self._meter.create_histogram(
            name="nl2api_context_retrieval_duration_ms",
            description="Context retrieval duration",
            unit="ms",
        )

        # Agent processing latency
        self._agent_duration = self._meter.create_histogram(
            name="nl2api_agent_duration_ms",
            description="Agent processing duration",
            unit="ms",
        )

        # Tool calls generated
        self._tool_calls = self._meter.create_histogram(
            name="nl2api_tool_calls_count",
            description="Number of tool calls generated per request",
            unit="1",
        )

    def record_request(self, metrics: "RequestMetrics") -> None:
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
                self._entity_resolution_duration.record(
                    metrics.entity_resolution_latency_ms, attrs
                )

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
            name="eval_batch_tests_total",
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
            batch_id: Batch identifier for grouping
            tags: Optional tags for filtering
        """
        if not self._enabled:
            return

        try:
            attrs: dict[str, Any] = {"batch_id": batch_id}
            if tags:
                attrs["tags"] = ",".join(tags)

            # Record test counts
            self._tests_total.add(1, attrs)

            if scorecard.overall_passed:
                self._tests_passed.add(1, attrs)
            else:
                self._tests_failed.add(1, attrs)

            # Record latency and score
            self._test_duration.record(scorecard.total_latency_ms, attrs)
            self._test_score.record(scorecard.overall_score, attrs)

            # Record per-stage results
            for stage_result in scorecard.stage_results:
                stage_attrs = {**attrs, "stage": stage_result.stage_name}
                if stage_result.passed:
                    self._stage_passed.add(1, stage_attrs)
                else:
                    self._stage_failed.add(1, stage_attrs)

        except Exception as e:
            logger.warning(f"Failed to record eval test metrics: {e}")

    def record_batch_complete(
        self,
        batch_job: "BatchJob",
        duration_seconds: float,
    ) -> None:
        """
        Record metrics for batch completion.

        Args:
            batch_job: The completed batch job
            duration_seconds: Total batch duration
        """
        if not self._enabled:
            return

        try:
            attrs: dict[str, Any] = {"batch_id": batch_job.batch_id}
            if batch_job.tags:
                attrs["tags"] = ",".join(batch_job.tags)

            self._batch_duration.record(duration_seconds, attrs)

        except Exception as e:
            logger.warning(f"Failed to record batch completion metrics: {e}")


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
            name="accuracy_tests_total",
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


# === Global Instances ===

_nl2api_metrics: NL2APIMetrics | None = None
_eval_metrics: EvalMetrics | None = None
_accuracy_metrics: AccuracyMetrics | None = None


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
