"""
Unit tests for regression detection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evaluation.continuous.regression import (
    RegressionDetector,
    RegressionResult,
    Severity,
)


class TestRegressionDetector:
    """Tests for RegressionDetector class."""

    @pytest.fixture
    def mock_scorecard_repo(self):
        """Create a mock scorecard repository."""
        repo = MagicMock()
        repo.get_batch_summary = AsyncMock()
        repo.get_by_batch = AsyncMock()
        return repo

    @pytest.fixture
    def detector(self, mock_scorecard_repo):
        """Create a RegressionDetector instance."""
        return RegressionDetector(mock_scorecard_repo)

    def test_compare_metric_no_previous(self, detector):
        """Test comparison when no previous value exists."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=None,
            current_value=0.95,
            current_sample_size=100,
            previous_sample_size=0,
        )

        assert result.metric_name == "pass_rate"
        assert result.previous_value is None
        assert result.current_value == 0.95
        assert result.is_regression is False
        assert result.severity is None

    def test_compare_metric_no_regression(self, detector):
        """Test comparison with no regression."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=0.90,
            current_value=0.92,
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is False
        assert result.delta == pytest.approx(0.02, abs=0.001)

    def test_compare_metric_warning_regression(self, detector):
        """Test detection of warning-level regression."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=0.95,
            current_value=0.92,  # 3% drop
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is True
        assert result.severity == Severity.WARNING
        assert result.delta == pytest.approx(-0.03, abs=0.001)

    def test_compare_metric_critical_regression(self, detector):
        """Test detection of critical regression."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=0.95,
            current_value=0.88,  # 7% drop
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is True
        assert result.severity == Severity.CRITICAL
        assert result.delta == pytest.approx(-0.07, abs=0.001)

    def test_compare_metric_latency_warning(self, detector):
        """Test latency metric warning detection."""
        result = detector._compare_metric(
            metric_name="avg_latency_ms",
            previous_value=100.0,
            current_value=140.0,  # 40% increase (between 1.3 and 1.5 threshold)
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is True
        assert result.severity == Severity.WARNING

    def test_compare_metric_latency_critical(self, detector):
        """Test latency metric critical detection."""
        result = detector._compare_metric(
            metric_name="avg_latency_ms",
            previous_value=100.0,
            current_value=250.0,  # 150% increase
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is True
        assert result.severity == Severity.CRITICAL

    def test_p_value_calculation(self, detector):
        """Test p-value calculation for pass rate comparison."""
        # Large sample with significant difference
        p_value = detector._calculate_p_value(
            p1=0.95,
            p2=0.85,
            n1=100,
            n2=100,
        )

        assert p_value is not None
        assert p_value < 0.05  # Should be significant

    def test_p_value_small_sample(self, detector):
        """Test p-value returns None for small samples."""
        p_value = detector._calculate_p_value(
            p1=0.95,
            p2=0.85,
            n1=10,  # Too small
            n2=100,
        )

        assert p_value is None

    def test_normal_cdf(self):
        """Test normal CDF approximation."""
        # Known values
        assert RegressionDetector._normal_cdf(0) == pytest.approx(0.5, abs=0.01)
        assert RegressionDetector._normal_cdf(1.96) == pytest.approx(0.975, abs=0.01)
        assert RegressionDetector._normal_cdf(-1.96) == pytest.approx(0.025, abs=0.01)

    @pytest.mark.asyncio
    async def test_detect_regressions_empty_batch(self, detector, mock_scorecard_repo):
        """Test regression detection with empty batch."""
        mock_scorecard_repo.get_batch_summary.return_value = {"total": 0}

        results = await detector.detect_regressions("batch-123")

        assert results == []

    @pytest.mark.asyncio
    async def test_detect_regressions_with_data(self, detector, mock_scorecard_repo):
        """Test regression detection with actual data."""
        # Mock current batch
        mock_scorecard_repo.get_batch_summary.return_value = {
            "total": 100,
            "passed": 90,
            "failed": 10,
            "avg_score": 0.85,
        }

        # Mock scorecards for latency calculation
        mock_scorecard = MagicMock()
        mock_scorecard.total_latency_ms = 150
        mock_scorecard_repo.get_by_batch.return_value = [mock_scorecard] * 100

        results = await detector.detect_regressions("batch-123")

        # Should have results for pass_rate, avg_score, avg_latency_ms
        assert len(results) == 3
        metric_names = {r.metric_name for r in results}
        assert "pass_rate" in metric_names
        assert "avg_score" in metric_names
        assert "avg_latency_ms" in metric_names


class TestRegressionDetectorEdgeCases:
    """Edge case tests for RegressionDetector."""

    @pytest.fixture
    def mock_scorecard_repo(self):
        """Create a mock scorecard repository."""
        repo = MagicMock()
        repo.get_batch_summary = AsyncMock()
        repo.get_by_batch = AsyncMock()
        return repo

    @pytest.fixture
    def detector(self, mock_scorecard_repo):
        """Create a RegressionDetector instance."""
        return RegressionDetector(mock_scorecard_repo)

    def test_compare_metric_zero_previous_value(self, detector):
        """Test comparison when previous value is zero."""
        result = detector._compare_metric(
            metric_name="avg_latency_ms",
            previous_value=0.0,
            current_value=100.0,
            current_sample_size=100,
            previous_sample_size=100,
        )

        # Can't compute ratio with zero, should handle gracefully
        assert result.is_regression is False or result.delta == 100.0

    def test_compare_metric_equal_values(self, detector):
        """Test comparison when values are equal (no change)."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=0.90,
            current_value=0.90,
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is False
        assert result.delta == 0.0

    def test_compare_metric_exactly_at_warning_threshold(self, detector):
        """Test comparison at exact warning threshold boundary."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=0.90,
            current_value=0.88,  # Exactly -2% change
            current_sample_size=100,
            previous_sample_size=100,
        )

        # At exactly -2%, should trigger warning
        assert result.is_regression is True
        assert result.severity == Severity.WARNING

    def test_compare_metric_just_above_warning_threshold(self, detector):
        """Test comparison just above warning threshold (no regression)."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=0.90,
            current_value=0.882,  # -1.8% change (above -2% threshold)
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is False

    def test_compare_metric_improvement(self, detector):
        """Test that improvements are not flagged as regressions."""
        result = detector._compare_metric(
            metric_name="pass_rate",
            previous_value=0.80,
            current_value=0.95,  # 15% improvement
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is False
        assert result.delta > 0

    def test_compare_metric_latency_no_regression(self, detector):
        """Test latency metric with acceptable increase."""
        result = detector._compare_metric(
            metric_name="avg_latency_ms",
            previous_value=100.0,
            current_value=120.0,  # 20% increase (below 1.3x threshold)
            current_sample_size=100,
            previous_sample_size=100,
        )

        assert result.is_regression is False

    def test_p_value_both_samples_small(self, detector):
        """Test p-value returns None when both samples are small."""
        p_value = detector._calculate_p_value(
            p1=0.95,
            p2=0.85,
            n1=5,
            n2=5,
        )

        assert p_value is None

    def test_p_value_identical_proportions(self, detector):
        """Test p-value when proportions are identical."""
        p_value = detector._calculate_p_value(
            p1=0.90,
            p2=0.90,
            n1=100,
            n2=100,
        )

        # With identical proportions, p-value should be 1.0 (no significant difference)
        assert p_value is not None
        assert p_value > 0.5

    def test_normal_cdf_large_positive(self):
        """Test normal CDF with large positive z."""
        assert RegressionDetector._normal_cdf(5.0) == pytest.approx(1.0, abs=0.001)

    def test_normal_cdf_large_negative(self):
        """Test normal CDF with large negative z."""
        assert RegressionDetector._normal_cdf(-5.0) == pytest.approx(0.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_detect_regressions_missing_batch_summary(self, detector, mock_scorecard_repo):
        """Test regression detection when batch summary returns None."""
        mock_scorecard_repo.get_batch_summary.return_value = None

        results = await detector.detect_regressions("batch-123")

        assert results == []

    @pytest.mark.asyncio
    async def test_detect_regressions_all_failures(self, detector, mock_scorecard_repo):
        """Test regression detection when all tests fail."""
        mock_scorecard_repo.get_batch_summary.return_value = {
            "total": 100,
            "passed": 0,
            "failed": 100,
            "avg_score": 0.0,
        }

        mock_scorecard = MagicMock()
        mock_scorecard.total_latency_ms = 100
        mock_scorecard_repo.get_by_batch.return_value = [mock_scorecard] * 100

        results = await detector.detect_regressions("batch-123")

        # Should handle 0% pass rate
        pass_rate_result = next((r for r in results if r.metric_name == "pass_rate"), None)
        assert pass_rate_result is not None
        assert pass_rate_result.current_value == 0.0


class TestRegressionResult:
    """Tests for RegressionResult dataclass."""

    def test_regression_result_creation(self):
        """Test creating a RegressionResult."""
        result = RegressionResult(
            metric_name="pass_rate",
            previous_value=0.95,
            current_value=0.90,
            threshold=-0.02,
            delta=-0.05,
            delta_pct=-5.26,
            is_regression=True,
            severity=Severity.CRITICAL,
            p_value=0.01,
        )

        assert result.metric_name == "pass_rate"
        assert result.is_regression is True
        assert result.severity == Severity.CRITICAL

    def test_regression_result_without_previous(self):
        """Test RegressionResult without previous value."""
        result = RegressionResult(
            metric_name="pass_rate",
            previous_value=None,
            current_value=0.90,
            threshold=-0.02,
            delta=None,
            delta_pct=None,
            is_regression=False,
            severity=None,
            p_value=None,
        )

        assert result.previous_value is None
        assert result.is_regression is False
