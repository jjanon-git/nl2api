"""
Routing Accuracy Tests

Tests whether the LLM router correctly routes queries to domain agents.
Uses real LLM calls to measure routing accuracy.

Run specific tiers:
    pytest tests/accuracy/routing/ -m tier1  # Quick check (~30s)
    pytest tests/accuracy/routing/ -m tier2  # Standard (~2min)
    pytest tests/accuracy/routing/ -m tier3  # Comprehensive (hours, batch API)
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from tests.accuracy.core.config import AccuracyConfig
from tests.accuracy.core.evaluator import (
    AccuracyReport,
    RoutingAccuracyEvaluator,
    RoutingTestCase,
)


def infer_domain_from_fixture(fixture) -> str | None:
    """
    Infer the expected domain from fixture tags and metadata.

    Returns the expected domain name or None if unclear.
    """
    tags = fixture.tags
    tags_lower = [t.lower() for t in tags]
    category = fixture.category.lower()

    # Check tags for domain hints
    for tag in tags_lower:
        if tag.startswith("datastream"):
            return "datastream"
        if tag.startswith("estimates"):
            return "estimates"
        if tag.startswith("fundamentals"):
            return "fundamentals"
        if tag.startswith("officers") or tag.startswith("officers-directors"):
            return "officers"
        if tag.startswith("screening"):
            return "screening"

    # Infer from category/tags
    estimates_keywords = [
        "eps",
        "consensus",
        "recommendations",
        "price-target",
        "forecast",
        "analyst",
        "estimate",
    ]
    fundamentals_keywords = [
        "balance-sheet",
        "cash-flow",
        "income-statement",
        "ratios",
        "revenue",
        "assets",
        "debt",
        "roe",
        "roa",
    ]
    officers_keywords = [
        "ceo",
        "cfo",
        "board",
        "executives",
        "compensation",
        "governance",
        "officers",
    ]
    screening_keywords = ["screening", "ranking", "top-", "filter", "find"]
    datastream_keywords = [
        "price",
        "ohlc",
        "volume",
        "time-series",
        "historical",
        "market-cap",
        "pe",
        "dividend-yield",
        "index",
    ]

    all_tags_str = " ".join(tags_lower)

    if any(kw in all_tags_str for kw in officers_keywords):
        return "officers"
    if any(kw in all_tags_str for kw in screening_keywords):
        return "screening"
    if any(kw in all_tags_str for kw in estimates_keywords):
        return "estimates"
    if any(kw in all_tags_str for kw in fundamentals_keywords):
        return "fundamentals"
    if any(kw in all_tags_str for kw in datastream_keywords):
        return "datastream"

    # Fallback based on category
    category_mapping = {
        "lookups": "datastream",
        "temporal": "datastream",
        "comparison": "datastream",
    }

    return category_mapping.get(category)


def load_routing_test_cases(
    limit: int | None = None,
    balanced: bool = True,
) -> list[RoutingTestCase]:
    """Load test cases with inferred routing expectations."""
    from tests.unit.nl2api.fixture_loader import FixtureLoader

    loader = FixtureLoader()
    fixtures = list(loader.iterate_all())

    # Group by domain for balanced sampling
    by_domain: dict[str, list[RoutingTestCase]] = defaultdict(list)
    skipped = 0

    for fixture in fixtures:
        expected_domain = infer_domain_from_fixture(fixture)

        if expected_domain is None:
            skipped += 1
            continue

        tc = RoutingTestCase(
            id=fixture.id,
            query=fixture.nl_query,
            expected_domain=expected_domain,
            category=fixture.category,
            tags=fixture.tags,
        )
        by_domain[expected_domain].append(tc)

    # Balanced sampling
    if balanced and limit:
        per_domain = limit // len(by_domain) if by_domain else limit
        test_cases = []
        for domain, cases in by_domain.items():
            test_cases.extend(cases[:per_domain])
        # Fill remaining slots
        remaining = limit - len(test_cases)
        for domain, cases in by_domain.items():
            if remaining <= 0:
                break
            extra = cases[per_domain : per_domain + remaining]
            test_cases.extend(extra)
            remaining -= len(extra)
    else:
        test_cases = []
        for domain, cases in by_domain.items():
            test_cases.extend(cases)
        if limit:
            test_cases = test_cases[:limit]

    return test_cases


def print_report(report: AccuracyReport, threshold: float):
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("ROUTING ACCURACY REPORT")
    print("=" * 60)

    print(f"\nOverall Accuracy: {report.accuracy:.1%}")
    print(f"  Correct: {report.correct_count}/{report.total_count}")
    print(f"  Failed: {report.failed_count}")
    print(f"  Errors: {report.error_count}")
    print(
        f"  Low Confidence (would clarify): {report.low_confidence_count} ({report.low_confidence_rate:.1%})"
    )

    if report.by_category:
        print("\nBy Category:")
        for cat, stats in sorted(report.by_category.items()):
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                print(f"  {cat:15} {acc:5.1%} ({stats['correct']}/{stats['total']})")

    print("\nConfusion Matrix:")
    domains = ["datastream", "estimates", "fundamentals", "officers", "screening"]
    print(f"{'':15}", end="")
    for d in domains:
        print(f"{d[:8]:>10}", end="")
    print()

    for expected in domains:
        print(f"{expected:15}", end="")
        for predicted in domains:
            count = report.confusion_matrix.get(expected, {}).get(predicted, 0)
            print(f"{count:10}", end="")
        print()

    print(f"\nThreshold: {threshold:.0%}")
    status = "PASS" if report.accuracy >= threshold else "FAIL"
    print(f"Result: {status}")
    print("=" * 60)


class TestRoutingAccuracy:
    """Routing accuracy tests with tier-based execution.

    Tier 1/2: Use realtime API for fast feedback (~30s for 50 samples).
    Tier 3: Use Batch API for comprehensive runs (50% cheaper, but ~8 hours).
    """

    @pytest.fixture
    def realtime_evaluator(self):
        """Create evaluator with realtime API (fast, for tier1/tier2)."""
        config = AccuracyConfig(model="claude-3-haiku-20240307", use_batch_api=False)
        return RoutingAccuracyEvaluator(config=config)

    @pytest.fixture
    def batch_evaluator(self):
        """Create evaluator with Batch API (slow but 50% cheaper, for tier3)."""
        config = AccuracyConfig(model="claude-3-haiku-20240307", use_batch_api=True)
        return RoutingAccuracyEvaluator(config=config)

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_routing_tier1(self, realtime_evaluator):
        """
        Tier 1: Quick routing check (50 samples, ~30s with realtime API).

        Run with: pytest tests/accuracy/routing/ -m tier1
        """
        test_cases = load_routing_test_cases(limit=50, balanced=True)
        threshold = 0.75  # Lower threshold for quick check

        def progress(current, total, result):
            if result and current % 10 == 0:
                print(f"  [{current}/{total}] {result.expected} -> {result.predicted}")

        report = await realtime_evaluator.evaluate_batch(
            test_cases, progress_callback=progress, tier="tier1"
        )
        print_report(report, threshold)

        assert report.accuracy >= threshold, (
            f"Routing accuracy {report.accuracy:.1%} below {threshold:.0%} threshold. "
            f"Failed: {report.failed_count}/{report.total_count}"
        )

    @pytest.mark.tier2
    @pytest.mark.asyncio
    async def test_routing_tier2(self, realtime_evaluator):
        """
        Tier 2: Standard routing evaluation (200 samples, ~2min with realtime API).

        Run with: pytest tests/accuracy/routing/ -m tier2
        """
        test_cases = load_routing_test_cases(limit=200, balanced=True)
        threshold = 0.80

        def progress(current, total, result):
            if result and current % 20 == 0:
                print(f"  [{current}/{total}]")

        report = await realtime_evaluator.evaluate_batch(
            test_cases, progress_callback=progress, tier="tier2"
        )
        print_report(report, threshold)

        assert report.accuracy >= threshold, (
            f"Routing accuracy {report.accuracy:.1%} below {threshold:.0%} threshold. "
            f"Failed: {report.failed_count}/{report.total_count}"
        )

    @pytest.mark.tier3
    @pytest.mark.asyncio
    async def test_routing_tier3(self, batch_evaluator):
        """
        Tier 3: Comprehensive routing evaluation (all samples).

        Uses Batch API (50% cheaper, but takes ~8 hours).
        Run with: pytest tests/accuracy/routing/ -m tier3
        """
        test_cases = load_routing_test_cases(limit=None, balanced=False)
        threshold = 0.80

        def progress(current, total, result):
            if current % 100 == 0:
                print(f"  [{current}/{total}]")

        report = await batch_evaluator.evaluate_batch(
            test_cases, progress_callback=progress, tier="tier3"
        )
        print_report(report, threshold)

        assert report.accuracy >= threshold, (
            f"Routing accuracy {report.accuracy:.1%} below {threshold:.0%} threshold. "
            f"Failed: {report.failed_count}/{report.total_count}"
        )

    @pytest.mark.asyncio
    async def test_ambiguous_queries_trigger_clarification(self, realtime_evaluator):
        """
        Test that ambiguous queries return low confidence.

        Queries like "EPS" without temporal context should trigger clarification.
        """
        ambiguous_cases = [
            RoutingTestCase(
                id="ambig_1",
                query="What is Apple's EPS?",
                expected_domain="fundamentals",  # Or estimates - genuinely ambiguous
                category="ambiguous",
            ),
            RoutingTestCase(
                id="ambig_2",
                query="Get Microsoft earnings per share",
                expected_domain="fundamentals",
                category="ambiguous",
            ),
            RoutingTestCase(
                id="ambig_3",
                query="Show me Tesla's EPS",
                expected_domain="fundamentals",
                category="ambiguous",
            ),
        ]

        report = await realtime_evaluator.evaluate_batch(ambiguous_cases)

        # These ambiguous queries should have low confidence (would trigger clarification)
        low_confidence_count = sum(1 for r in report.results if r.confidence <= 0.5)

        assert low_confidence_count >= 2, (
            f"Expected at least 2/3 ambiguous queries to have low confidence, "
            f"but only {low_confidence_count} did. "
            f"Confidences: {[r.confidence for r in report.results]}"
        )
