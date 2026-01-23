"""
Accuracy tests for screening queries (SCREEN expressions, rankings).

Tests the quality of SCREEN expression generation using real LLM calls.
"""

from __future__ import annotations

import pytest

from tests.accuracy.core.config import CATEGORY_THRESHOLDS, DEFAULT_MIN_ACCURACY


@pytest.mark.requires_llm
class TestScreeningAccuracy:
    """Accuracy tests for screening query domain."""

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_top_n_queries(self, evaluator, fixture_loader, emit_accuracy_report):
        """Quick check on top-N ranking queries."""
        cases = fixture_loader.load_by_subcategory("screening", "top_n", limit=20)

        if not cases:
            # Try category directly
            cases = fixture_loader.load_category("screening", limit=20)

        if not cases:
            pytest.skip("No screening fixtures")

        report = await evaluator.evaluate_batch(cases, category="screening/top_n")
        emit_accuracy_report(report, category="screening/top_n", tier="tier1")

        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("screening", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold

    @pytest.mark.tier2
    @pytest.mark.asyncio
    async def test_screening_expressions(self, evaluator, fixture_loader, emit_accuracy_report):
        """Standard evaluation on screening expressions."""
        cases = fixture_loader.load_category("screening", limit=100)

        if not cases:
            pytest.skip("No screening fixtures")

        report = await evaluator.evaluate_batch(cases, category="screening")
        emit_accuracy_report(report, category="screening", tier="tier2")

        print(f"\n{report.summary()}")

        threshold = CATEGORY_THRESHOLDS.get("screening", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold

    @pytest.mark.tier3
    @pytest.mark.asyncio
    async def test_all_screening(self, evaluator, fixture_loader, emit_accuracy_report):
        """Comprehensive evaluation on all screening fixtures."""
        cases = fixture_loader.load_category("screening")

        if not cases:
            pytest.skip("No screening fixtures")

        report = await evaluator.evaluate_batch(cases, category="screening")
        emit_accuracy_report(report, category="screening", tier="tier3")

        print(f"\n{report.summary()}")

        # Show failed queries
        failed = report.get_failed_results()[:5]
        if failed:
            print("\nSample failed screening queries:")
            for r in failed:
                print(f"  Query: {r.query[:80]}...")
                if r.missing_tools:
                    print(f"  Missing tools: {r.missing_tools}")
                if r.extra_tools:
                    print(f"  Extra tools: {r.extra_tools}")

        threshold = CATEGORY_THRESHOLDS.get("screening", DEFAULT_MIN_ACCURACY)
        assert report.accuracy >= threshold
