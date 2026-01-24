"""
Regression Detection

Compares evaluation results between batches to detect regressions.
Uses statistical tests to distinguish noise from real regressions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.common.storage.postgres.scorecard_repo import PostgresScorecardRepository

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Alert severity levels."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RegressionResult:
    """Result of a regression check for a single metric."""

    metric_name: str
    previous_value: float | None
    current_value: float
    threshold: float
    delta: float | None  # Absolute change
    delta_pct: float | None  # Percentage change
    is_regression: bool
    severity: Severity | None
    p_value: float | None  # Statistical significance


class RegressionDetector:
    """
    Detects regressions between evaluation batches.

    Uses configurable thresholds and statistical tests to determine
    if performance changes are significant regressions.
    """

    # Default thresholds for different metrics
    THRESHOLDS: dict[str, dict[str, float]] = {
        "pass_rate": {
            "warning": -0.02,  # 2% drop
            "critical": -0.05,  # 5% drop
        },
        "avg_score": {
            "warning": -0.02,
            "critical": -0.05,
        },
        "p95_latency_ms": {
            "warning": 1.5,  # 50% increase
            "critical": 2.0,  # 100% increase
        },
        "avg_latency_ms": {
            "warning": 1.3,  # 30% increase
            "critical": 1.5,  # 50% increase
        },
    }

    # Minimum sample size for statistical tests
    MIN_SAMPLE_SIZE = 20

    def __init__(
        self,
        scorecard_repo: PostgresScorecardRepository,
        thresholds: dict[str, dict[str, float]] | None = None,
    ):
        """
        Initialize regression detector.

        Args:
            scorecard_repo: Repository for fetching scorecard data
            thresholds: Optional custom thresholds (overrides defaults)
        """
        self.scorecard_repo = scorecard_repo
        self.thresholds = thresholds or self.THRESHOLDS

    async def detect_regressions(
        self,
        current_batch_id: str,
        previous_batch_id: str | None = None,
        client_type: str | None = None,
    ) -> list[RegressionResult]:
        """
        Detect regressions between current batch and previous batch.

        Args:
            current_batch_id: ID of the current (new) batch
            previous_batch_id: ID of the previous batch (auto-detected if None)
            client_type: Client type to compare (for finding previous batch)

        Returns:
            List of RegressionResult for each checked metric
        """
        results = []

        # Get current batch metrics
        current_metrics = await self._get_batch_metrics(current_batch_id)
        if not current_metrics:
            logger.warning(f"No metrics found for batch {current_batch_id}")
            return results

        # Get previous batch metrics
        if previous_batch_id:
            previous_metrics = await self._get_batch_metrics(previous_batch_id)
        else:
            # Find previous batch for same client type
            previous_metrics = await self._get_previous_batch_metrics(current_batch_id, client_type)

        # Compare metrics
        for metric_name in ["pass_rate", "avg_score", "avg_latency_ms"]:
            result = self._compare_metric(
                metric_name,
                previous_metrics.get(metric_name) if previous_metrics else None,
                current_metrics.get(metric_name),
                current_metrics.get("total_tests", 0),
                previous_metrics.get("total_tests", 0) if previous_metrics else 0,
            )
            results.append(result)

        return results

    async def _get_batch_metrics(self, batch_id: str) -> dict[str, Any]:
        """Get aggregated metrics for a batch."""
        summary = await self.scorecard_repo.get_batch_summary(batch_id)

        if summary is None or summary.get("total", 0) == 0:
            return {}

        pass_rate = summary["passed"] / summary["total"] if summary["total"] > 0 else 0

        # Get latency metrics
        scorecards = await self.scorecard_repo.get_by_batch(batch_id)
        latencies = [sc.total_latency_ms for sc in scorecards if sc.total_latency_ms > 0]

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            "total_tests": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "pass_rate": pass_rate,
            "avg_score": summary["avg_score"],
            "avg_latency_ms": avg_latency,
        }

    async def _get_previous_batch_metrics(
        self,
        current_batch_id: str,
        client_type: str | None,
    ) -> dict[str, Any] | None:
        """Find and get metrics for the previous batch of the same client type."""
        # This is a simplified implementation - in production, you'd query
        # for the most recent batch before the current one with the same client_type
        # For now, we return None to indicate no comparison available
        return None

    def _compare_metric(
        self,
        metric_name: str,
        previous_value: float | None,
        current_value: float | None,
        current_sample_size: int,
        previous_sample_size: int,
    ) -> RegressionResult:
        """Compare a single metric between batches."""
        if current_value is None:
            return RegressionResult(
                metric_name=metric_name,
                previous_value=previous_value,
                current_value=0.0,
                threshold=0.0,
                delta=None,
                delta_pct=None,
                is_regression=False,
                severity=None,
                p_value=None,
            )

        if previous_value is None:
            # No previous value to compare
            return RegressionResult(
                metric_name=metric_name,
                previous_value=None,
                current_value=current_value,
                threshold=0.0,
                delta=None,
                delta_pct=None,
                is_regression=False,
                severity=None,
                p_value=None,
            )

        # Calculate delta
        delta = current_value - previous_value
        delta_pct = (delta / previous_value * 100) if previous_value != 0 else 0

        # Get thresholds for this metric
        thresholds = self.thresholds.get(metric_name, {})
        warning_threshold = thresholds.get("warning", 0)
        critical_threshold = thresholds.get("critical", 0)

        # Determine if this is a regression
        is_regression = False
        severity = None

        if metric_name in ["pass_rate", "avg_score"]:
            # Lower is worse for these metrics (check if delta is below threshold)
            if delta <= critical_threshold:
                is_regression = True
                severity = Severity.CRITICAL
            elif delta <= warning_threshold:
                is_regression = True
                severity = Severity.WARNING
        else:
            # Higher is worse for latency metrics (check if ratio exceeds threshold)
            ratio = current_value / previous_value if previous_value > 0 else 1
            if ratio >= critical_threshold:
                is_regression = True
                severity = Severity.CRITICAL
            elif ratio >= warning_threshold:
                is_regression = True
                severity = Severity.WARNING

        # Calculate statistical significance (two-proportion z-test for pass_rate)
        p_value = None
        if metric_name == "pass_rate" and is_regression:
            p_value = self._calculate_p_value(
                previous_value,
                current_value,
                previous_sample_size,
                current_sample_size,
            )

        return RegressionResult(
            metric_name=metric_name,
            previous_value=previous_value,
            current_value=current_value,
            threshold=warning_threshold if severity == Severity.WARNING else critical_threshold,
            delta=delta,
            delta_pct=delta_pct,
            is_regression=is_regression,
            severity=severity,
            p_value=p_value,
        )

    def _calculate_p_value(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int,
    ) -> float | None:
        """
        Calculate p-value using two-proportion z-test.

        Tests if the difference between two proportions is statistically significant.

        Args:
            p1: First proportion (previous pass rate)
            p2: Second proportion (current pass rate)
            n1: First sample size
            n2: Second sample size

        Returns:
            p-value or None if sample size is too small
        """
        if n1 < self.MIN_SAMPLE_SIZE or n2 < self.MIN_SAMPLE_SIZE:
            return None

        # Pooled proportion
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        if p_pooled == 0 or p_pooled == 1:
            return None

        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

        if se == 0:
            return None

        # Z-score
        z = (p1 - p2) / se

        # Two-tailed p-value (using normal approximation)
        # This is a simplified calculation - for production, use scipy.stats
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        return p_value

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """
        Approximate normal CDF using error function approximation.

        For production, use scipy.stats.norm.cdf instead.
        """
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)
