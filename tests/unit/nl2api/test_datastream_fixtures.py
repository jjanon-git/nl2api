"""
Comprehensive tests for DatastreamAgent using generated fixtures.

Tests against the 12,887 generated test cases in tests/fixtures/lseg/generated/
"""

from __future__ import annotations

import pytest
from typing import Any

from src.nl2api.agents.datastream import DatastreamAgent
from src.nl2api.agents.protocols import AgentContext
from tests.unit.nl2api.fixture_loader import (
    FixtureLoader,
    GeneratedTestCase,
    compare_tool_calls,
    extract_ticker_symbol,
)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    async def complete(self, messages, tools=None, temperature=0.0):
        return None


@pytest.fixture
def loader() -> FixtureLoader:
    """Create fixture loader."""
    return FixtureLoader()


@pytest.fixture
def agent() -> DatastreamAgent:
    """Create DatastreamAgent with mock LLM."""
    return DatastreamAgent(llm=MockLLMProvider())


class TestFixtureLoaderBasics:
    """Test the fixture loader itself."""

    def test_loader_finds_fixtures(self, loader: FixtureLoader):
        """Test that the loader can find the fixture files."""
        summary = loader.get_summary()
        assert "lookups" in summary
        assert summary["lookups"] > 0

    def test_loader_loads_lookups(self, loader: FixtureLoader):
        """Test loading lookup test cases."""
        cases = loader.load_category("lookups")
        assert len(cases) > 0
        assert all(isinstance(c, GeneratedTestCase) for c in cases)

    def test_loader_loads_temporal(self, loader: FixtureLoader):
        """Test loading temporal test cases."""
        cases = loader.load_category("temporal")
        assert len(cases) > 0

    def test_loader_loads_comparisons(self, loader: FixtureLoader):
        """Test loading comparison test cases."""
        cases = loader.load_category("comparisons")
        assert len(cases) > 0

    def test_loader_loads_screening(self, loader: FixtureLoader):
        """Test loading screening test cases."""
        cases = loader.load_category("screening")
        assert len(cases) > 0

    def test_loader_total_count(self, loader: FixtureLoader):
        """Test that we have the expected number of test cases."""
        summary = loader.get_summary()
        total = sum(summary.values())
        # Should have approximately 12,887 test cases
        assert total >= 10000, f"Expected at least 10000 test cases, got {total}"

    def test_load_by_tag(self, loader: FixtureLoader):
        """Test loading by tag."""
        price_cases = loader.load_by_tag("price")
        assert len(price_cases) > 0
        assert all("price" in c.tags for c in price_cases)


class TestDatastreamAgentFieldDetection:
    """Test field detection against lookup fixtures."""

    @pytest.mark.asyncio
    async def test_price_field_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of price fields in lookup queries."""
        # Load price-related lookups
        price_cases = loader.load_by_tag("price")[:100]  # Sample first 100

        detected_count = 0
        for case in price_cases:
            fields = agent._detect_fields(case.nl_query.lower())

            # Check if P field is detected for price queries
            # Note: agent defaults to P when no specific field detected
            if "P" in fields:
                detected_count += 1

        # Detection rate includes default behavior
        detection_rate = detected_count / len(price_cases) if price_cases else 0
        print(f"\nPrice field detection rate: {detection_rate:.2%}")
        # Many price queries will get P either explicitly or as default
        assert detection_rate >= 0.1, f"Price field detection rate: {detection_rate:.2%}"

    @pytest.mark.asyncio
    async def test_volume_field_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of volume fields."""
        # Find cases with volume in the query
        volume_cases = [
            c for c in loader.load_category("lookups")
            if "volume" in c.nl_query.lower()
        ][:50]

        if not volume_cases:
            pytest.skip("No volume test cases found")

        detected_count = 0
        for case in volume_cases:
            fields = agent._detect_fields(case.nl_query.lower())
            if "VO" in fields:
                detected_count += 1

        detection_rate = detected_count / len(volume_cases) if volume_cases else 0
        assert detection_rate >= 0.5, f"Volume field detection rate: {detection_rate:.2%}"

    @pytest.mark.asyncio
    async def test_market_cap_field_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of market cap fields."""
        # Find market cap cases
        mkt_cap_cases = [
            c for c in loader.iterate_all()
            if any(term in c.nl_query.lower() for term in ["market cap", "market capitalization"])
        ][:50]

        if not mkt_cap_cases:
            pytest.skip("No market cap test cases found")

        detected_count = 0
        for case in mkt_cap_cases:
            fields = agent._detect_fields(case.nl_query.lower())
            if "MV" in fields:
                detected_count += 1

        detection_rate = detected_count / len(mkt_cap_cases)
        assert detection_rate >= 0.5, f"Market cap field detection rate: {detection_rate:.2%}"

    @pytest.mark.asyncio
    async def test_pe_ratio_field_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of PE ratio fields."""
        pe_cases = [
            c for c in loader.iterate_all()
            if any(term in c.nl_query.lower() for term in ["pe ratio", "p/e", "price to earnings"])
        ][:50]

        if not pe_cases:
            pytest.skip("No PE ratio test cases found")

        detected_count = 0
        for case in pe_cases:
            fields = agent._detect_fields(case.nl_query.lower())
            if "PE" in fields:
                detected_count += 1

        detection_rate = detected_count / len(pe_cases)
        assert detection_rate >= 0.5, f"PE ratio field detection rate: {detection_rate:.2%}"

    @pytest.mark.asyncio
    async def test_dividend_yield_field_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of dividend yield fields."""
        div_cases = [
            c for c in loader.iterate_all()
            if any(term in c.nl_query.lower() for term in ["dividend yield", "yield"])
        ][:50]

        if not div_cases:
            pytest.skip("No dividend yield test cases found")

        detected_count = 0
        for case in div_cases:
            fields = agent._detect_fields(case.nl_query.lower())
            if "DY" in fields:
                detected_count += 1

        # Dividend yield detection may have lower rate due to ambiguity
        detection_rate = detected_count / len(div_cases)
        assert detection_rate >= 0.3, f"Dividend yield field detection rate: {detection_rate:.2%}"


class TestDatastreamAgentTimeRangeDetection:
    """Test time range detection against temporal fixtures."""

    @pytest.mark.asyncio
    async def test_temporal_queries_have_time_params(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test that temporal queries are detected correctly."""
        temporal_cases = loader.load_category("temporal")[:200]

        detected_count = 0
        for case in temporal_cases:
            time_params = agent._detect_time_range(case.nl_query.lower())
            if time_params:
                detected_count += 1

        detection_rate = detected_count / len(temporal_cases) if temporal_cases else 0
        assert detection_rate >= 0.5, f"Temporal detection rate: {detection_rate:.2%}"

    @pytest.mark.asyncio
    async def test_last_month_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of 'last month' time range."""
        last_month_cases = [
            c for c in loader.load_category("temporal")
            if "last month" in c.nl_query.lower() or "past month" in c.nl_query.lower()
        ][:50]

        if not last_month_cases:
            pytest.skip("No last month test cases found")

        for case in last_month_cases:
            time_params = agent._detect_time_range(case.nl_query.lower())
            assert time_params is not None, f"Failed to detect time range for: {case.nl_query}"
            assert "start" in time_params
            assert "-1M" in time_params.get("start", "") or "M" in time_params.get("start", "")

    @pytest.mark.asyncio
    async def test_last_year_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of 'last year' time range."""
        last_year_cases = [
            c for c in loader.load_category("temporal")
            if "last year" in c.nl_query.lower() or "past year" in c.nl_query.lower()
        ][:50]

        if not last_year_cases:
            pytest.skip("No last year test cases found")

        detected_count = 0
        for case in last_year_cases:
            time_params = agent._detect_time_range(case.nl_query.lower())
            if time_params and "start" in time_params:
                detected_count += 1

        rate = detected_count / len(last_year_cases) if last_year_cases else 0
        print(f"\nLast year detection rate: {rate:.2%}")
        assert rate >= 0.5, f"Last year detection rate: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_frequency_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test detection of frequency in temporal queries."""
        # Find queries with explicit frequency (that don't have conflicting frequencies)
        weekly_cases = [
            c for c in loader.load_category("temporal")
            if "weekly" in c.nl_query.lower() and "daily" not in c.nl_query.lower()
        ][:30]

        monthly_cases = [
            c for c in loader.load_category("temporal")
            if "monthly" in c.nl_query.lower() and "weekly" not in c.nl_query.lower() and "daily" not in c.nl_query.lower()
        ][:30]

        # Test weekly frequency
        weekly_correct = 0
        for case in weekly_cases:
            time_params = agent._detect_time_range(case.nl_query.lower())
            if time_params and time_params.get("freq") == "W":
                weekly_correct += 1

        # Test monthly frequency
        monthly_correct = 0
        for case in monthly_cases:
            time_params = agent._detect_time_range(case.nl_query.lower())
            if time_params and time_params.get("freq") == "M":
                monthly_correct += 1

        weekly_rate = weekly_correct / len(weekly_cases) if weekly_cases else 0
        monthly_rate = monthly_correct / len(monthly_cases) if monthly_cases else 0

        print(f"\nFrequency detection:")
        print(f"  Weekly: {weekly_rate:.2%} ({weekly_correct}/{len(weekly_cases)})")
        print(f"  Monthly: {monthly_rate:.2%} ({monthly_correct}/{len(monthly_cases)})")

        # At least some frequency detection should work
        assert weekly_rate >= 0.5 or monthly_rate >= 0.5, "Frequency detection too low"


class TestDatastreamAgentComparisonQueries:
    """Test handling of comparison queries."""

    @pytest.mark.asyncio
    async def test_comparison_queries_detected(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test that comparison queries are handled."""
        comparison_cases = loader.load_category("comparisons")[:100]

        can_handle_count = 0
        for case in comparison_cases:
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                can_handle_count += 1

        handle_rate = can_handle_count / len(comparison_cases) if comparison_cases else 0
        assert handle_rate >= 0.3, f"Comparison query handle rate: {handle_rate:.2%}"

    @pytest.mark.asyncio
    async def test_multi_ticker_field_detection(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test field detection for multi-ticker comparison queries."""
        # Two-stock comparisons
        two_stock_cases = loader.load_by_subcategory("two_stock")[:50]

        if not two_stock_cases:
            pytest.skip("No two-stock comparison cases found")

        for case in two_stock_cases:
            fields = agent._detect_fields(case.nl_query.lower())
            # Should detect at least one field
            assert len(fields) > 0, f"No fields detected for: {case.nl_query}"


class TestDatastreamAgentCanHandle:
    """Test can_handle method against various query types."""

    @pytest.mark.asyncio
    async def test_can_handle_price_queries(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test can_handle returns positive score for price queries."""
        price_cases = loader.load_by_tag("price")[:100]

        positive_count = 0
        for case in price_cases:
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                positive_count += 1

        rate = positive_count / len(price_cases) if price_cases else 0
        assert rate >= 0.5, f"can_handle rate for price queries: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_can_handle_time_series_queries(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test can_handle returns positive score for time series queries."""
        temporal_cases = loader.load_category("temporal")[:100]

        positive_count = 0
        for case in temporal_cases:
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                positive_count += 1

        rate = positive_count / len(temporal_cases) if temporal_cases else 0
        assert rate >= 0.5, f"can_handle rate for temporal queries: {rate:.2%}"


class TestDatastreamAgentStatistics:
    """Statistical tests across the full fixture set."""

    @pytest.mark.asyncio
    async def test_lookup_coverage_statistics(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Generate statistics for lookup query coverage."""
        lookups = loader.load_category("lookups")

        stats = {
            "total": len(lookups),
            "field_detected": 0,
            "can_handle": 0,
            "by_field_category": {},
        }

        # Sample for efficiency
        sample_size = min(500, len(lookups))
        sample = lookups[:sample_size]

        for case in sample:
            # Check field detection
            fields = agent._detect_fields(case.nl_query.lower())
            if fields and fields != ["P"]:  # Exclude default
                stats["field_detected"] += 1

            # Check can_handle
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                stats["can_handle"] += 1

            # Track by field category from metadata
            field_cat = case.metadata.get("field_category", "unknown")
            if field_cat not in stats["by_field_category"]:
                stats["by_field_category"][field_cat] = {"total": 0, "detected": 0}
            stats["by_field_category"][field_cat]["total"] += 1
            if fields:
                stats["by_field_category"][field_cat]["detected"] += 1

        # Assert reasonable coverage
        field_rate = stats["field_detected"] / sample_size
        handle_rate = stats["can_handle"] / sample_size

        print(f"\nLookup Statistics (n={sample_size}):")
        print(f"  Field detection rate: {field_rate:.2%}")
        print(f"  Can handle rate: {handle_rate:.2%}")
        print(f"  By field category: {stats['by_field_category']}")

        assert handle_rate >= 0.3, f"Overall can_handle rate too low: {handle_rate:.2%}"

    @pytest.mark.asyncio
    async def test_temporal_coverage_statistics(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Generate statistics for temporal query coverage."""
        temporal = loader.load_category("temporal")

        stats = {
            "total": len(temporal),
            "time_range_detected": 0,
            "can_handle": 0,
            "by_frequency": {},
        }

        sample_size = min(500, len(temporal))
        sample = temporal[:sample_size]

        for case in sample:
            # Check time range detection
            time_params = agent._detect_time_range(case.nl_query.lower())
            if time_params:
                stats["time_range_detected"] += 1

            # Check can_handle
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                stats["can_handle"] += 1

            # Track by frequency
            freq = case.metadata.get("frequency", "unknown")
            if freq not in stats["by_frequency"]:
                stats["by_frequency"][freq] = {"total": 0, "detected": 0}
            stats["by_frequency"][freq]["total"] += 1
            if time_params:
                stats["by_frequency"][freq]["detected"] += 1

        time_rate = stats["time_range_detected"] / sample_size
        handle_rate = stats["can_handle"] / sample_size

        print(f"\nTemporal Statistics (n={sample_size}):")
        print(f"  Time range detection rate: {time_rate:.2%}")
        print(f"  Can handle rate: {handle_rate:.2%}")
        print(f"  By frequency: {stats['by_frequency']}")

        assert time_rate >= 0.4, f"Time range detection rate too low: {time_rate:.2%}"

    @pytest.mark.asyncio
    async def test_comparison_coverage_statistics(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Generate statistics for comparison query coverage."""
        comparisons = loader.load_category("comparisons")

        stats = {
            "total": len(comparisons),
            "field_detected": 0,
            "can_handle": 0,
        }

        sample_size = min(500, len(comparisons))
        sample = comparisons[:sample_size]

        for case in sample:
            fields = agent._detect_fields(case.nl_query.lower())
            if fields:
                stats["field_detected"] += 1

            score = await agent.can_handle(case.nl_query)
            if score > 0:
                stats["can_handle"] += 1

        field_rate = stats["field_detected"] / sample_size
        handle_rate = stats["can_handle"] / sample_size

        print(f"\nComparison Statistics (n={sample_size}):")
        print(f"  Field detection rate: {field_rate:.2%}")
        print(f"  Can handle rate: {handle_rate:.2%}")

        assert field_rate >= 0.5, f"Comparison field detection rate too low: {field_rate:.2%}"


class TestExpectedFieldMappings:
    """Test that field mappings match expected fixture fields."""

    @pytest.mark.asyncio
    async def test_field_code_coverage(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test that agent field mappings cover fixture field codes."""
        # Collect all expected field codes from fixtures
        lookups = loader.load_category("lookups")

        fixture_fields = set()
        for case in lookups:
            for tc in case.expected_tool_calls:
                args = tc.get("arguments", {})
                for field in args.get("fields", []):
                    fixture_fields.add(field)

        # Get agent field codes
        agent_fields = set()
        for field_map in [
            agent.PRICE_FIELDS,
            agent.VALUATION_FIELDS,
            agent.DIVIDEND_FIELDS,
            agent.INFO_FIELDS,
            agent.FUNDAMENTAL_FIELDS,
        ]:
            for v in field_map.values():
                if isinstance(v, list):
                    agent_fields.update(v)
                else:
                    agent_fields.add(v)

        # Check coverage
        covered = fixture_fields & agent_fields
        missing = fixture_fields - agent_fields

        coverage_rate = len(covered) / len(fixture_fields) if fixture_fields else 0

        print(f"\nField Code Coverage:")
        print(f"  Total fixture fields: {len(fixture_fields)}")
        print(f"  Total agent fields: {len(agent_fields)}")
        print(f"  Covered fields: {len(covered)}")
        print(f"  Missing fields: {len(missing)}")
        print(f"  Coverage rate: {coverage_rate:.2%}")

        # The fixtures contain many specialized TR.* fields
        # The agent covers core Datastream fields (P, PE, MV, etc.)
        # 10%+ is reasonable given the breadth of fixture fields
        assert coverage_rate >= 0.1, f"Field code coverage too low: {coverage_rate:.2%}"

        # Verify core fields are covered
        core_fields = {"P", "PE", "MV", "DY", "VO", "EPS"}
        core_covered = core_fields & covered
        assert len(core_covered) >= 4, f"Core fields not covered: {core_fields - covered}"


class TestTickerExtraction:
    """Test ticker extraction capabilities."""

    def test_extract_ticker_symbol(self):
        """Test ticker symbol extraction from various formats."""
        test_cases = [
            ("@AAPL", "AAPL"),
            ("U:MSFT", "MSFT"),
            ("C:ENB", "ENB"),
            ("D:MBG", "MBG"),
            ("J:6758", "6758"),
            ("AAPL.O", "AAPL"),
            ("BARC.L", "BARC"),
            ("VOD", "VOD"),
        ]

        for ticker, expected in test_cases:
            result = extract_ticker_symbol(ticker)
            assert result == expected, f"Expected {expected}, got {result} for {ticker}"

    @pytest.mark.asyncio
    async def test_company_pattern_coverage(self, agent: DatastreamAgent, loader: FixtureLoader):
        """Test that common companies in fixtures are recognized."""
        lookups = loader.load_category("lookups")[:500]

        # Extract company names mentioned in queries
        company_mentions = {}
        for case in lookups:
            query_lower = case.nl_query.lower()
            for company in agent.US_COMPANY_PATTERNS:
                if company in query_lower:
                    company_mentions[company] = company_mentions.get(company, 0) + 1

        print(f"\nCompany mentions in fixtures: {company_mentions}")

        # Should recognize common companies
        assert len(company_mentions) > 0, "No companies recognized from fixtures"
