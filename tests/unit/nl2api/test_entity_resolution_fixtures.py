"""
Entity Resolution Fixture Tests.

Tests the EntityResolver against generated entity resolution test cases.
This tests a different component (EntityResolver) than the agent coverage tests,
so it's tracked separately from the main fixture coverage tests.

Coverage thresholds are defined in CoverageRegistry.ENTITY_RESOLUTION_COVERAGE.
"""

from __future__ import annotations

import pytest

from src.nl2api.resolution.mock_resolver import MockEntityResolver
from tests.unit.nl2api.fixture_loader import FixtureLoader, GeneratedTestCase
from tests.unit.nl2api.test_fixture_coverage import CoverageRegistry

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def loader() -> FixtureLoader:
    """Create a fixture loader."""
    return FixtureLoader()


@pytest.fixture
def resolver() -> MockEntityResolver:
    """Create a mock entity resolver for testing."""
    return MockEntityResolver()


# =============================================================================
# Helper Functions
# =============================================================================

def get_entity_resolution_cases(loader: FixtureLoader) -> list[GeneratedTestCase]:
    """Load all entity resolution test cases."""
    return loader.load_category("entity_resolution")


def get_subcategory_cases(
    loader: FixtureLoader, subcategory: str
) -> list[GeneratedTestCase]:
    """Load entity resolution test cases for a specific subcategory."""
    all_cases = get_entity_resolution_cases(loader)
    return [c for c in all_cases if c.subcategory == subcategory]


def extract_input_entity(case: GeneratedTestCase) -> str:
    """Extract the input entity from a test case."""
    return case.metadata.get("input_entity", "")


def extract_expected_ric(case: GeneratedTestCase) -> str | None:
    """Extract the expected RIC from a test case."""
    return case.metadata.get("expected_ric")


# =============================================================================
# Test Classes
# =============================================================================

class TestEntityResolutionFixtureDiscovery:
    """Tests for entity resolution fixture discovery."""

    def test_entity_resolution_category_exists(self, loader: FixtureLoader):
        """Verify entity_resolution category is in CATEGORIES."""
        assert "entity_resolution" in loader.CATEGORIES

    def test_entity_resolution_fixtures_load(self, loader: FixtureLoader):
        """Test that entity resolution fixtures can be loaded."""
        cases = get_entity_resolution_cases(loader)
        # May be empty if fixtures haven't been generated yet
        # This test just verifies the loading mechanism works
        assert isinstance(cases, list)

    @pytest.mark.skipif(
        not FixtureLoader().load_category("entity_resolution"),
        reason="Entity resolution fixtures not generated yet"
    )
    def test_fixture_structure(self, loader: FixtureLoader):
        """Verify fixture structure is correct."""
        cases = get_entity_resolution_cases(loader)
        assert len(cases) > 0, "Expected some entity resolution fixtures"

        # Check first case has required fields
        case = cases[0]
        assert case.id.startswith("entity_resolution_")
        assert case.category == "entity_resolution"
        assert case.subcategory in CoverageRegistry.ENTITY_RESOLUTION_COVERAGE
        assert "input_entity" in case.metadata

    @pytest.mark.skipif(
        not FixtureLoader().load_category("entity_resolution"),
        reason="Entity resolution fixtures not generated yet"
    )
    def test_all_subcategories_present(self, loader: FixtureLoader):
        """Verify all expected subcategories are present."""
        cases = get_entity_resolution_cases(loader)
        found_subcategories = {c.subcategory for c in cases}

        expected = set(CoverageRegistry.ENTITY_RESOLUTION_COVERAGE.keys())
        missing = expected - found_subcategories

        # Allow some missing subcategories if database doesn't have enough data
        assert len(missing) <= 3, f"Too many missing subcategories: {missing}"


@pytest.mark.skip(reason="MockEntityResolver (~150 companies) cannot meaningfully test coverage against fixtures from 2.9M entities. Use tests/accuracy/ with real database.")
class TestEntityResolutionCoverage:
    """Tests for entity resolution coverage tracking.

    NOTE: These tests are skipped because the MockEntityResolver only has ~150
    hardcoded companies, while the fixtures are generated from a database with
    2.9M entities. Coverage will always be ~0% with a mock.

    For real coverage testing, run accuracy tests which use the real EntityResolver
    against the PostgreSQL database:
        pytest tests/accuracy/ -m entity_resolution
    """

    @pytest.mark.asyncio
    async def test_exact_match_coverage(
        self, loader: FixtureLoader, resolver: MockEntityResolver
    ):
        """Test coverage on exact_match subcategory."""
        await self._test_subcategory_coverage(loader, resolver, "exact_match")

    @pytest.mark.asyncio
    async def test_ticker_lookup_coverage(
        self, loader: FixtureLoader, resolver: MockEntityResolver
    ):
        """Test coverage on ticker_lookup subcategory."""
        await self._test_subcategory_coverage(loader, resolver, "ticker_lookup")

    @pytest.mark.asyncio
    async def test_negative_cases_coverage(
        self, loader: FixtureLoader, resolver: MockEntityResolver
    ):
        """Test coverage on negative_cases subcategory."""
        await self._test_subcategory_coverage(loader, resolver, "negative_cases")

    async def _test_subcategory_coverage(
        self,
        loader: FixtureLoader,
        resolver: MockEntityResolver,
        subcategory: str,
        sample_size: int = 50
    ):
        """Test resolver coverage for a subcategory."""
        cases = get_subcategory_cases(loader, subcategory)

        if not cases:
            pytest.skip(f"No cases for subcategory {subcategory}")

        # Sample for speed
        sample = cases[:sample_size]

        correct = 0
        for case in sample:
            input_entity = extract_input_entity(case)
            expected_ric = extract_expected_ric(case)

            result = await resolver.resolve_single(input_entity)

            if expected_ric is None:
                # Negative case - should NOT resolve
                if result is None:
                    correct += 1
            else:
                # Positive case - should resolve to expected RIC
                if result is not None and result.identifier == expected_ric:
                    correct += 1

        rate = correct / len(sample)
        threshold = CoverageRegistry.get_entity_resolution_threshold(subcategory)

        # Report coverage
        print(f"\n{subcategory}: {rate:.1%} coverage (threshold: {threshold:.1%})")

        # Assert meets threshold (with some tolerance for test stability)
        assert rate >= threshold * 0.9, (
            f"{subcategory} coverage {rate:.1%} below threshold {threshold:.1%}"
        )


@pytest.mark.skip(reason="MockEntityResolver (~150 companies) cannot meaningfully test coverage against fixtures from 2.9M entities. Use tests/accuracy/ with real database.")
class TestEntityResolutionSummary:
    """Generate summary reports for entity resolution coverage.

    NOTE: Skipped for same reason as TestEntityResolutionCoverage - mock resolver
    cannot produce meaningful results against database-generated fixtures.
    """

    @pytest.mark.asyncio
    async def test_generate_coverage_summary(
        self, loader: FixtureLoader, resolver: MockEntityResolver
    ):
        """Generate and display coverage summary for all subcategories."""
        cases = get_entity_resolution_cases(loader)

        if not cases:
            pytest.skip("No entity resolution fixtures")

        # Group by subcategory
        by_subcategory: dict[str, list[GeneratedTestCase]] = {}
        for case in cases:
            subcat = case.subcategory
            if subcat not in by_subcategory:
                by_subcategory[subcat] = []
            by_subcategory[subcat].append(case)

        print("\n" + "=" * 60)
        print("ENTITY RESOLUTION COVERAGE SUMMARY")
        print("=" * 60)
        print(f"Total fixtures: {len(cases):,}")
        print()
        print(f"{'Subcategory':<20} {'Count':>8} {'Sampled':>8} {'Correct':>8} {'Rate':>8}")
        print("-" * 56)

        total_correct = 0
        total_sampled = 0

        for subcat, subcat_cases in sorted(by_subcategory.items()):
            sample = subcat_cases[:50]
            correct = 0

            for case in sample:
                input_entity = extract_input_entity(case)
                expected_ric = extract_expected_ric(case)

                result = await resolver.resolve_single(input_entity)

                if expected_ric is None:
                    if result is None:
                        correct += 1
                else:
                    if result is not None and result.identifier == expected_ric:
                        correct += 1

            rate = correct / len(sample) if sample else 0
            total_correct += correct
            total_sampled += len(sample)

            print(
                f"{subcat:<20} {len(subcat_cases):>8,} {len(sample):>8} "
                f"{correct:>8} {rate:>7.1%}"
            )

        overall_rate = total_correct / total_sampled if total_sampled else 0
        print("-" * 56)
        print(
            f"{'TOTAL':<20} {len(cases):>8,} {total_sampled:>8} "
            f"{total_correct:>8} {overall_rate:>7.1%}"
        )
        print("=" * 60)


@pytest.mark.skip(reason="MockEntityResolver (~150 companies) cannot meaningfully test baseline against fixtures from 2.9M entities. Use tests/accuracy/ with real database.")
class TestEntityResolutionBaseline:
    """Baseline tests to establish current resolver accuracy.

    NOTE: Skipped for same reason as TestEntityResolutionCoverage - mock resolver
    cannot produce meaningful results against database-generated fixtures.
    """

    @pytest.mark.asyncio
    async def test_baseline_accuracy(
        self, loader: FixtureLoader, resolver: MockEntityResolver
    ):
        """
        Establish baseline accuracy of current resolver.

        This test documents the current state and will be used as a
        regression marker as the resolver is improved.
        """
        cases = get_entity_resolution_cases(loader)

        if not cases:
            pytest.skip("No entity resolution fixtures")

        # Sample across all subcategories
        sample = cases[:200]

        correct = 0
        by_subcategory: dict[str, tuple[int, int]] = {}  # subcat -> (correct, total)

        for case in sample:
            input_entity = extract_input_entity(case)
            expected_ric = extract_expected_ric(case)
            subcat = case.subcategory

            if subcat not in by_subcategory:
                by_subcategory[subcat] = (0, 0)

            result = await resolver.resolve_single(input_entity)

            is_correct = False
            if expected_ric is None:
                is_correct = result is None
            else:
                is_correct = result is not None and result.identifier == expected_ric

            if is_correct:
                correct += 1
                by_subcategory[subcat] = (
                    by_subcategory[subcat][0] + 1,
                    by_subcategory[subcat][1] + 1
                )
            else:
                by_subcategory[subcat] = (
                    by_subcategory[subcat][0],
                    by_subcategory[subcat][1] + 1
                )

        overall_rate = correct / len(sample)

        # Print baseline results
        print(f"\nBaseline accuracy: {overall_rate:.1%} ({correct}/{len(sample)})")
        print("\nBy subcategory:")
        for subcat, (c, t) in sorted(by_subcategory.items()):
            rate = c / t if t > 0 else 0
            print(f"  {subcat}: {rate:.1%} ({c}/{t})")

        # This is expected to be low with the current resolver
        # The test should pass as long as we're not regressing
        # Update this threshold as the resolver improves
        assert overall_rate >= 0.05, (
            f"Baseline accuracy {overall_rate:.1%} is worse than expected minimum 5%"
        )
