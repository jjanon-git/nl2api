"""
Accuracy tests for DatastreamAgent.

Tests the quality of tool call generation for price, time series,
and calculated field queries using real LLM calls.
"""

from __future__ import annotations

import pytest

from tests.accuracy.conftest import requires_llm
from tests.accuracy.core.config import CATEGORY_THRESHOLDS, DEFAULT_MIN_ACCURACY


@pytest.mark.requires_llm
class TestDatastreamAccuracy:
    """Accuracy tests for DatastreamAgent."""

    # =========================================================================
    # Tier 1: Quick Sanity Checks
    # =========================================================================

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_single_field_lookups(self, evaluator, fixture_loader, emit_accuracy_report):
        """Quick accuracy check on single field lookups (price queries)."""
        cases = fixture_loader.load_by_subcategory("lookups", "single_field", limit=25)

        if not cases:
            pytest.skip("No single_field fixtures available")

        report = await evaluator.evaluate_batch(cases, category="lookups/single_field")

        # Emit to OTEL
        emit_accuracy_report(report, category="lookups/single_field", tier="tier1")

        # Log summary
        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("lookups", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold, (
            f"Single field lookup accuracy {report.accuracy:.1%} below {threshold:.0%} threshold. "
            f"Failed {report.failed_count}/{report.total_count} queries."
        )

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_temporal_queries(self, evaluator, fixture_loader, emit_accuracy_report):
        """Quick accuracy check on temporal (time series) queries."""
        cases = fixture_loader.load_category("temporal", limit=25)

        if not cases:
            pytest.skip("No temporal fixtures available")

        report = await evaluator.evaluate_batch(cases, category="temporal")

        emit_accuracy_report(report, category="temporal", tier="tier1")
        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("temporal", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold, (
            f"Temporal query accuracy {report.accuracy:.1%} below {threshold:.0%} threshold."
        )

    # =========================================================================
    # Tier 2: Standard Evaluation
    # =========================================================================

    @pytest.mark.tier2
    @pytest.mark.asyncio
    async def test_lookups_comprehensive(self, evaluator, fixture_loader, emit_accuracy_report):
        """Standard accuracy evaluation on all lookup query types."""
        cases = fixture_loader.load_category("lookups", limit=100)

        if not cases:
            pytest.skip("No lookups fixtures available")

        report = await evaluator.evaluate_batch(cases, category="lookups")

        emit_accuracy_report(report, category="lookups", tier="tier2")
        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("lookups", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold

    @pytest.mark.tier2
    @pytest.mark.asyncio
    async def test_comparisons(self, evaluator, fixture_loader, emit_accuracy_report):
        """Standard accuracy evaluation on comparison queries (multi-stock)."""
        cases = fixture_loader.load_category("comparisons", limit=50)

        if not cases:
            pytest.skip("No comparisons fixtures available")

        report = await evaluator.evaluate_batch(cases, category="comparisons")

        emit_accuracy_report(report, category="comparisons", tier="tier2")
        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("comparisons", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold

    # =========================================================================
    # Tier 3: Comprehensive Evaluation
    # =========================================================================

    @pytest.mark.tier3
    @pytest.mark.asyncio
    async def test_all_datastream_categories(self, evaluator, fixture_loader, emit_accuracy_report):
        """Comprehensive evaluation across all Datastream-relevant categories."""
        categories = ["lookups", "temporal", "comparisons"]
        overall_correct = 0
        overall_total = 0

        for category in categories:
            cases = fixture_loader.load_category(category)  # All fixtures

            if not cases:
                continue

            report = await evaluator.evaluate_batch(cases, category=category)

            emit_accuracy_report(report, category=category, tier="tier3")
            print(f"\n{category}:\n{report.summary()}")

            overall_correct += report.correct_count
            overall_total += report.total_count

        if overall_total == 0:
            pytest.skip("No fixtures available")

        overall_accuracy = overall_correct / overall_total
        print(f"\nOverall Datastream Accuracy: {overall_accuracy:.1%} ({overall_correct}/{overall_total})")

        assert overall_accuracy >= DEFAULT_MIN_ACCURACY, (
            f"Overall Datastream accuracy {overall_accuracy:.1%} below {DEFAULT_MIN_ACCURACY:.0%}"
        )

    @pytest.mark.tier3
    @pytest.mark.asyncio
    async def test_complex_queries(self, evaluator, fixture_loader, emit_accuracy_report):
        """Comprehensive evaluation on complex multi-step queries."""
        cases = fixture_loader.load_category("complex")

        if not cases:
            pytest.skip("No complex fixtures available")

        report = await evaluator.evaluate_batch(cases, category="complex")

        emit_accuracy_report(report, category="complex", tier="tier3")
        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("complex", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold
