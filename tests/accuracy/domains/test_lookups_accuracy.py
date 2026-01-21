"""
Accuracy tests for lookup queries (price, single field, multi-field).

This tests the domain/category rather than a specific agent,
measuring accuracy across all agents that handle lookup queries.
"""

from __future__ import annotations

import pytest

from tests.accuracy.conftest import requires_llm
from tests.accuracy.core.config import CATEGORY_THRESHOLDS, DEFAULT_MIN_ACCURACY


@pytest.mark.requires_llm
class TestLookupsAccuracy:
    """Accuracy tests for lookup query domain."""

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_single_field_lookups(self, evaluator, fixture_loader, emit_accuracy_report):
        """Quick check on single field lookups."""
        cases = fixture_loader.load_by_subcategory("lookups", "single_field", limit=30)

        if not cases:
            pytest.skip("No single_field fixtures")

        report = await evaluator.evaluate_batch(cases, category="lookups/single_field")
        emit_accuracy_report(report, category="lookups/single_field", tier="tier1")

        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("lookups", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_multi_field_lookups(self, evaluator, fixture_loader, emit_accuracy_report):
        """Quick check on multi-field lookups."""
        cases = fixture_loader.load_by_subcategory("lookups", "multi_field", limit=20)

        if not cases:
            pytest.skip("No multi_field fixtures")

        report = await evaluator.evaluate_batch(cases, category="lookups/multi_field")
        emit_accuracy_report(report, category="lookups/multi_field", tier="tier1")

        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("lookups", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold

    @pytest.mark.tier2
    @pytest.mark.asyncio
    async def test_all_lookup_subcategories(self, evaluator, fixture_loader, emit_accuracy_report):
        """Standard evaluation across all lookup subcategories."""
        subcategories = ["single_field", "multi_field", "calculated", "derived"]
        overall_correct = 0
        overall_total = 0

        for subcat in subcategories:
            cases = fixture_loader.load_by_subcategory("lookups", subcat, limit=50)

            if not cases:
                continue

            report = await evaluator.evaluate_batch(cases, category=f"lookups/{subcat}")
            emit_accuracy_report(report, category=f"lookups/{subcat}", tier="tier2")

            print(f"\n{subcat}:\n{report.summary()}")

            overall_correct += report.correct_count
            overall_total += report.total_count

        if overall_total == 0:
            pytest.skip("No lookup fixtures available")

        overall_accuracy = overall_correct / overall_total
        print(f"\nOverall Lookups Accuracy: {overall_accuracy:.1%}")

        threshold = CATEGORY_THRESHOLDS.get("lookups", DEFAULT_MIN_ACCURACY)
        assert overall_accuracy >= threshold

    @pytest.mark.tier3
    @pytest.mark.asyncio
    async def test_all_lookups(self, evaluator, fixture_loader, emit_accuracy_report):
        """Comprehensive evaluation on all lookup fixtures."""
        cases = fixture_loader.load_category("lookups")

        if not cases:
            pytest.skip("No lookups fixtures")

        report = await evaluator.evaluate_batch(cases, category="lookups", parallel=10)
        emit_accuracy_report(report, category="lookups", tier="tier3")

        print(f"\n{report.summary()}")

        # Show failed queries for debugging
        failed = report.get_failed_results()[:10]
        if failed:
            print(f"\nSample failed queries:")
            for r in failed:
                print(f"  - {r.query[:60]}...")
                print(f"    Expected: {[tc.tool_name for tc in r.expected_tool_calls]}")
                print(f"    Actual:   {[tc.tool_name for tc in r.actual_tool_calls]}")

        threshold = CATEGORY_THRESHOLDS.get("lookups", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold
