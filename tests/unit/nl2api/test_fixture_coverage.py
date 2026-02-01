"""
Dynamic fixture coverage tests.

These tests automatically expand to cover all fixture categories and subcategories,
ensuring that as test data grows, test coverage grows with it.

Note: Some tests require full fixtures (FIXTURE_SAMPLE_SIZE=0) and are skipped
when sampling is enabled for faster unit test runs.
"""

from __future__ import annotations

import pytest

from src.nl2api.agents.datastream import DatastreamAgent
from src.nl2api.agents.screening import ScreeningAgent
from tests.unit.nl2api.fixture_loader import FIXTURE_SAMPLE_SIZE, FixtureLoader

# Skip marker for tests that need full fixtures
requires_full_fixtures = pytest.mark.skipif(
    FIXTURE_SAMPLE_SIZE > 0,
    reason=f"Requires full fixtures (FIXTURE_SAMPLE_SIZE={FIXTURE_SAMPLE_SIZE})",
)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    async def complete(self, messages, tools=None, temperature=0.0):
        return None


# =============================================================================
# Fixture Discovery - Automatically finds all categories and subcategories
# =============================================================================


def discover_fixture_categories() -> list[str]:
    """Discover all fixture categories from the filesystem."""
    loader = FixtureLoader()
    return [cat for cat in loader.CATEGORIES if loader.load_category(cat)]


def discover_subcategories() -> dict[str, set[str]]:
    """Discover all subcategories for each category."""
    loader = FixtureLoader()
    result = {}
    for category in discover_fixture_categories():
        cases = loader.load_category(category)
        subcats = {c.subcategory for c in cases if c.subcategory}
        if subcats:
            result[category] = subcats
    return result


def discover_tags() -> set[str]:
    """Discover all unique tags across all fixtures."""
    loader = FixtureLoader()
    tags = set()
    for case in loader.iterate_all():
        tags.update(case.tags)
    return tags


# =============================================================================
# Coverage Registry - Tracks which categories/subcategories have coverage
# =============================================================================


class CoverageRegistry:
    """
    Registry that tracks fixture coverage requirements.

    Add entries here to enforce coverage for specific categories/subcategories.
    Tests will fail if coverage drops below the specified threshold.
    """

    # Format: (category, subcategory or None, min_detection_rate, agent_class)
    # These thresholds should be adjusted as agents improve
    REQUIRED_COVERAGE = [
        # DatastreamAgent coverage
        ("lookups", "single_field", 0.3, DatastreamAgent),
        ("lookups", "multi_field", 0.15, DatastreamAgent),  # Lower threshold - complex queries
        ("temporal", "historical_price", 0.4, DatastreamAgent),
        ("comparisons", "two_stock", 0.3, DatastreamAgent),
        # ScreeningAgent coverage
        ("screening", "index_constituents", 0.3, ScreeningAgent),
        ("screening", "top_n", 0.5, ScreeningAgent),
        # NOTE: entity_resolution coverage is tracked separately in
        # test_entity_resolution_fixtures.py since it tests the EntityResolver
        # component rather than domain agents.
    ]

    # Entity resolution coverage thresholds (for use by test_entity_resolution_fixtures.py)
    # NOTE: These thresholds are for MockEntityResolver (~150 hardcoded companies)
    # against fixtures generated from 2.9M entities. Coverage will be very low.
    # For real accuracy testing, use tests/accuracy/ with the real database resolver.
    ENTITY_RESOLUTION_COVERAGE = {
        "exact_match": 0.01,  # Mock has ~150 companies vs 2.9M in fixtures
        "ticker_lookup": 0.01,  # Mock only knows ~150 tickers
        "alias_match": 0.01,  # Limited alias coverage in mock
        "suffix_variations": 0.01,  # Mock doesn't handle all suffixes
        "fuzzy_misspellings": 0.00,  # Mock has no fuzzy matching
        "abbreviations": 0.05,  # Some common ones hardcoded
        "international": 0.01,  # ~10 international companies in mock
        "ambiguous": 0.00,  # Non-deterministic
        "ticker_collisions": 0.00,  # No exchange context
        "edge_case_names": 0.01,  # Limited coverage
        "negative_cases": 0.30,  # Over-matches common words
    }

    @classmethod
    def get_requirements_for_agent(cls, agent_class: type) -> list[tuple]:
        """Get coverage requirements for a specific agent."""
        return [r for r in cls.REQUIRED_COVERAGE if r[3] == agent_class]

    @classmethod
    def get_entity_resolution_threshold(cls, subcategory: str) -> float:
        """Get expected coverage threshold for entity resolution subcategory."""
        return cls.ENTITY_RESOLUTION_COVERAGE.get(subcategory, 0.0)


# =============================================================================
# Dynamic Category Tests - Auto-generated from fixture structure
# =============================================================================


@pytest.fixture
def loader() -> FixtureLoader:
    return FixtureLoader()


class TestFixtureDiscovery:
    """Tests that verify fixture structure is as expected."""

    def test_all_expected_categories_exist(self, loader: FixtureLoader):
        """Verify all expected categories have fixtures."""
        expected = {"lookups", "temporal", "comparisons", "screening", "complex"}
        actual = set(discover_fixture_categories())

        missing = expected - actual
        assert not missing, f"Missing expected categories: {missing}"

    @requires_full_fixtures
    def test_minimum_fixture_count(self, loader: FixtureLoader):
        """Verify we have a minimum number of fixtures."""
        total = sum(len(loader.load_category(c)) for c in discover_fixture_categories())

        # Should have at least 10,000 fixtures
        assert total >= 10000, f"Expected at least 10,000 fixtures, got {total}"

    def test_each_category_has_fixtures(self, loader: FixtureLoader):
        """Verify each discovered category has at least some fixtures."""
        for category in discover_fixture_categories():
            cases = loader.load_category(category)
            assert len(cases) > 0, f"Category {category} has no fixtures"

    def test_subcategories_discovered(self, loader: FixtureLoader):
        """Verify subcategories are discovered correctly."""
        subcats = discover_subcategories()

        # Should have subcategories for major categories
        assert "lookups" in subcats
        assert "screening" in subcats
        assert len(subcats["lookups"]) >= 1
        assert len(subcats["screening"]) >= 1


# =============================================================================
# Parameterized Coverage Tests - One test per category
# =============================================================================

# Dynamically generate test parameters from discovered categories
CATEGORY_PARAMS = [(cat,) for cat in discover_fixture_categories()]


@pytest.mark.parametrize("category", [c[0] for c in CATEGORY_PARAMS])
class TestCategoryCanHandle:
    """Test that agents can handle queries from each category."""

    @pytest.fixture
    def datastream_agent(self) -> DatastreamAgent:
        return DatastreamAgent(llm=MockLLMProvider())

    @pytest.fixture
    def screening_agent(self) -> ScreeningAgent:
        return ScreeningAgent(llm=MockLLMProvider())

    @pytest.mark.asyncio
    async def test_datastream_handles_category(
        self,
        category: str,
        loader: FixtureLoader,
        datastream_agent: DatastreamAgent,
    ):
        """Test DatastreamAgent can handle queries from this category."""
        cases = loader.load_category(category)[:100]  # Sample for speed

        if not cases:
            pytest.skip(f"No cases in category {category}")

        # Skip screening category for datastream
        if category == "screening":
            pytest.skip("Screening category not for DatastreamAgent")

        can_handle_count = 0
        for c in cases:
            if await datastream_agent.can_handle(c.nl_query) > 0:
                can_handle_count += 1

        rate = can_handle_count / len(cases)
        # At least some queries should be handleable
        assert rate >= 0.1, f"DatastreamAgent handles only {rate:.1%} of {category}"

    @pytest.mark.asyncio
    async def test_screening_handles_category(
        self,
        category: str,
        loader: FixtureLoader,
        screening_agent: ScreeningAgent,
    ):
        """Test ScreeningAgent can handle queries from this category."""
        cases = loader.load_category(category)[:100]

        if not cases:
            pytest.skip(f"No cases in category {category}")

        # Only test screening-relevant categories
        if category not in ("screening", "complex"):
            pytest.skip(f"Category {category} not for ScreeningAgent")

        can_handle_count = 0
        for c in cases:
            if await screening_agent.can_handle(c.nl_query) > 0:
                can_handle_count += 1

        rate = can_handle_count / len(cases)
        assert rate >= 0.2, f"ScreeningAgent handles only {rate:.1%} of {category}"


# =============================================================================
# Coverage Enforcement Tests
# =============================================================================


class TestCoverageEnforcement:
    """Tests that enforce minimum coverage thresholds."""

    @pytest.fixture
    def datastream_agent(self) -> DatastreamAgent:
        return DatastreamAgent(llm=MockLLMProvider())

    @pytest.fixture
    def screening_agent(self) -> ScreeningAgent:
        return ScreeningAgent(llm=MockLLMProvider())

    @requires_full_fixtures
    @pytest.mark.asyncio
    async def test_required_coverage_met(
        self,
        loader: FixtureLoader,
        datastream_agent: DatastreamAgent,
        screening_agent: ScreeningAgent,
    ):
        """Test that all required coverage thresholds are met."""
        agents = {
            DatastreamAgent: datastream_agent,
            ScreeningAgent: screening_agent,
        }

        failures = []

        for category, subcategory, min_rate, agent_class in CoverageRegistry.REQUIRED_COVERAGE:
            agent = agents[agent_class]

            # Load fixtures for this category/subcategory
            if subcategory:
                cases = loader.load_by_subcategory(subcategory)[:50]
            else:
                cases = loader.load_category(category)[:50]

            if not cases:
                continue

            # Calculate coverage
            can_handle = 0
            for c in cases:
                if await agent.can_handle(c.nl_query) > 0:
                    can_handle += 1
            rate = can_handle / len(cases)

            if rate < min_rate:
                failures.append(
                    f"{agent_class.__name__} on {category}/{subcategory}: "
                    f"{rate:.1%} < {min_rate:.1%} required"
                )

        assert not failures, "Coverage requirements not met:\n" + "\n".join(failures)

    @pytest.mark.asyncio
    async def test_new_subcategories_have_some_coverage(
        self,
        loader: FixtureLoader,
        datastream_agent: DatastreamAgent,
        screening_agent: ScreeningAgent,
    ):
        """
        Test that any new subcategories have at least minimal coverage.

        This catches cases where new test data is added but agents
        can't handle any of it.
        """
        subcats = discover_subcategories()

        # Track which subcategories have zero coverage
        zero_coverage = []

        for category, subcategory_set in subcats.items():
            for subcat in subcategory_set:
                cases = loader.load_by_subcategory(subcat)[:20]
                if not cases:
                    continue

                # Try both agents
                ds_handles = 0
                sc_handles = 0
                for c in cases:
                    if await datastream_agent.can_handle(c.nl_query) > 0:
                        ds_handles += 1
                    if await screening_agent.can_handle(c.nl_query) > 0:
                        sc_handles += 1

                if ds_handles == 0 and sc_handles == 0:
                    zero_coverage.append(f"{category}/{subcat}")

        # Known categories that are expected to have zero coverage for now
        # These are advanced features not yet implemented
        expected_zero_coverage_prefixes = [
            "complex/",  # Complex multi-step queries
            "errors/",  # Error handling scenarios
        ]

        unexpected_zero_coverage = [
            sub
            for sub in zero_coverage
            if not any(sub.startswith(prefix) for prefix in expected_zero_coverage_prefixes)
        ]

        # Fail only if there are too many UNEXPECTED zero coverage subcategories
        # (i.e., basic features that should work but don't)
        if len(unexpected_zero_coverage) > 5:
            pytest.fail(
                f"Unexpected zero coverage in {len(unexpected_zero_coverage)} subcategories:\n"
                + "\n".join(unexpected_zero_coverage[:10])
            )

        # Informational: print all zero coverage subcategories
        if zero_coverage:
            print(f"\nSubcategories with zero coverage ({len(zero_coverage)} total):")
            print(f"  Unexpected: {len(unexpected_zero_coverage)}")
            print(
                f"  Expected (complex/errors): {len(zero_coverage) - len(unexpected_zero_coverage)}"
            )
            if unexpected_zero_coverage:
                print("\n  Unexpected zero coverage:")
                for subcat in unexpected_zero_coverage[:5]:
                    print(f"    - {subcat}")


# =============================================================================
# Fixture Growth Detection
# =============================================================================


class TestFixtureGrowth:
    """Tests that detect when fixture data grows."""

    BASELINE_COUNTS = {
        "lookups": 3700,
        "temporal": 2700,
        "comparisons": 3600,
        "screening": 260,
        "complex": 2200,
    }

    @requires_full_fixtures
    def test_fixture_counts_not_decreased(self, loader: FixtureLoader):
        """Verify fixture counts haven't decreased (data regression)."""
        for category, baseline in self.BASELINE_COUNTS.items():
            actual = len(loader.load_category(category))
            assert actual >= baseline * 0.9, (
                f"Category {category} has {actual} fixtures, "
                f"expected at least {int(baseline * 0.9)} (baseline: {baseline})"
            )

    def test_report_fixture_growth(self, loader: FixtureLoader):
        """Report when fixtures have grown (informational)."""
        growth_report = []

        for category, baseline in self.BASELINE_COUNTS.items():
            actual = len(loader.load_category(category))
            if actual > baseline * 1.1:  # More than 10% growth
                growth_report.append(f"{category}: {actual} (was {baseline}, +{actual - baseline})")

        if growth_report:
            print("\nFixture growth detected:")
            for line in growth_report:
                print(f"  {line}")


# =============================================================================
# Tag Coverage Tests
# =============================================================================


class TestTagCoverage:
    """Tests based on fixture tags."""

    # Tags that should have agent coverage
    REQUIRED_TAG_COVERAGE = {
        "price": (DatastreamAgent, 0.3),
        "time_series": (DatastreamAgent, 0.4),
        "comparison": (DatastreamAgent, 0.3),
        "screening": (ScreeningAgent, 0.3),
        "top_n": (ScreeningAgent, 0.5),
    }

    @pytest.fixture
    def datastream_agent(self) -> DatastreamAgent:
        return DatastreamAgent(llm=MockLLMProvider())

    @pytest.fixture
    def screening_agent(self) -> ScreeningAgent:
        return ScreeningAgent(llm=MockLLMProvider())

    @requires_full_fixtures
    @pytest.mark.asyncio
    async def test_required_tags_covered(
        self,
        loader: FixtureLoader,
        datastream_agent: DatastreamAgent,
        screening_agent: ScreeningAgent,
    ):
        """Test that required tags have sufficient coverage."""
        agents = {
            DatastreamAgent: datastream_agent,
            ScreeningAgent: screening_agent,
        }

        failures = []

        for tag, (agent_class, min_rate) in self.REQUIRED_TAG_COVERAGE.items():
            cases = loader.load_by_tag(tag)[:50]
            if not cases:
                continue

            agent = agents[agent_class]
            can_handle = 0
            for c in cases:
                if await agent.can_handle(c.nl_query) > 0:
                    can_handle += 1
            rate = can_handle / len(cases)

            if rate < min_rate:
                failures.append(
                    f"Tag '{tag}' with {agent_class.__name__}: {rate:.1%} < {min_rate:.1%}"
                )

        assert not failures, "Tag coverage requirements not met:\n" + "\n".join(failures)

    def test_all_tags_discovered(self, loader: FixtureLoader):
        """Report all discovered tags."""
        tags = discover_tags()
        print(f"\nDiscovered {len(tags)} unique tags:")
        for tag in sorted(tags)[:20]:
            print(f"  - {tag}")
        if len(tags) > 20:
            print(f"  ... and {len(tags) - 20} more")


# =============================================================================
# Summary Report Test
# =============================================================================


class TestCoverageSummary:
    """Generate a coverage summary report."""

    @pytest.fixture
    def datastream_agent(self) -> DatastreamAgent:
        return DatastreamAgent(llm=MockLLMProvider())

    @pytest.fixture
    def screening_agent(self) -> ScreeningAgent:
        return ScreeningAgent(llm=MockLLMProvider())

    @pytest.mark.asyncio
    async def test_generate_coverage_report(
        self,
        loader: FixtureLoader,
        datastream_agent: DatastreamAgent,
        screening_agent: ScreeningAgent,
    ):
        """Generate and display a coverage summary."""
        report = {
            "total_fixtures": 0,
            "categories": {},
        }

        for category in discover_fixture_categories():
            cases = loader.load_category(category)
            sample = cases[:50]  # Sample for speed

            ds_handles = 0
            sc_handles = 0
            for c in sample:
                if await datastream_agent.can_handle(c.nl_query) > 0:
                    ds_handles += 1
                if await screening_agent.can_handle(c.nl_query) > 0:
                    sc_handles += 1

            report["total_fixtures"] += len(cases)
            report["categories"][category] = {
                "count": len(cases),
                "sample_size": len(sample),
                "datastream_coverage": ds_handles / len(sample) if sample else 0,
                "screening_coverage": sc_handles / len(sample) if sample else 0,
            }

        # Print report
        print("\n" + "=" * 60)
        print("FIXTURE COVERAGE SUMMARY")
        print("=" * 60)
        print(f"Total fixtures: {report['total_fixtures']:,}")
        print()
        print(f"{'Category':<15} {'Count':>8} {'Datastream':>12} {'Screening':>12}")
        print("-" * 50)

        for cat, data in report["categories"].items():
            print(
                f"{cat:<15} {data['count']:>8,} "
                f"{data['datastream_coverage']:>11.1%} "
                f"{data['screening_coverage']:>11.1%}"
            )

        print("=" * 60)
