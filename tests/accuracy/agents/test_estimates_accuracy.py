"""
Accuracy tests for EstimatesAgent.

Tests the quality of tool call generation for I/B/E/S forecasts,
analyst recommendations, and consensus estimates using real LLM calls.
"""

from __future__ import annotations

import pytest

from tests.accuracy.conftest import requires_llm
from tests.accuracy.core.config import CATEGORY_THRESHOLDS, DEFAULT_MIN_ACCURACY


@pytest.mark.requires_llm
class TestEstimatesAccuracy:
    """Accuracy tests for EstimatesAgent."""

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_eps_queries(self, evaluator, fixture_loader, emit_accuracy_report):
        """Quick accuracy check on EPS estimate queries."""
        # Load fixtures tagged with estimates-related terms
        cases = fixture_loader.load_by_tag("estimates", limit=25)

        if not cases:
            # Fallback: try to find any cases mentioning EPS or estimates
            all_cases = list(fixture_loader.iterate_all())
            cases = [
                c for c in all_cases
                if "eps" in c.nl_query.lower() or "estimate" in c.nl_query.lower()
            ][:25]

        if not cases:
            pytest.skip("No estimates fixtures available")

        report = await evaluator.evaluate_batch(cases, category="estimates")

        emit_accuracy_report(report, category="estimates", tier="tier1")
        print(f"\n{report.summary()}")

        assert report.accuracy >= DEFAULT_MIN_ACCURACY, (
            f"Estimates query accuracy {report.accuracy:.1%} below {DEFAULT_MIN_ACCURACY:.0%}"
        )

    @pytest.mark.tier2
    @pytest.mark.asyncio
    async def test_estimates_comprehensive(self, evaluator, fixture_loader, emit_accuracy_report):
        """Standard accuracy evaluation on estimates queries."""
        cases = fixture_loader.load_by_tag("estimates", limit=100)

        if not cases:
            all_cases = list(fixture_loader.iterate_all())
            cases = [
                c for c in all_cases
                if any(term in c.nl_query.lower() for term in [
                    "eps", "estimate", "forecast", "analyst", "recommendation",
                    "consensus", "ibes", "mean", "median"
                ])
            ][:100]

        if not cases:
            pytest.skip("No estimates fixtures available")

        report = await evaluator.evaluate_batch(cases, category="estimates")

        emit_accuracy_report(report, category="estimates", tier="tier2")
        print(f"\n{report.summary()}")

        assert report.accuracy >= DEFAULT_MIN_ACCURACY

    @pytest.mark.tier3
    @pytest.mark.asyncio
    async def test_all_estimates_queries(self, evaluator, fixture_loader, emit_accuracy_report):
        """Comprehensive evaluation on all estimates-related queries."""
        all_cases = list(fixture_loader.iterate_all())
        cases = [
            c for c in all_cases
            if any(term in c.nl_query.lower() for term in [
                "eps", "estimate", "forecast", "analyst", "recommendation",
                "consensus", "ibes", "mean", "median", "target price",
                "earnings", "revenue estimate"
            ])
        ]

        if not cases:
            pytest.skip("No estimates fixtures available")

        report = await evaluator.evaluate_batch(cases, category="estimates")

        emit_accuracy_report(report, category="estimates", tier="tier3")
        print(f"\n{report.summary()}")

        assert report.accuracy >= DEFAULT_MIN_ACCURACY
