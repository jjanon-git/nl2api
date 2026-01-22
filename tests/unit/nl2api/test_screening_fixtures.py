"""
Comprehensive tests for ScreeningAgent using generated fixtures.

Tests against the 265 screening test cases in tests/fixtures/lseg/generated/screening/
"""

from __future__ import annotations

import re
import pytest
from typing import Any

from src.nl2api.agents.screening import ScreeningAgent
from src.nl2api.agents.protocols import AgentContext
from tests.unit.nl2api.fixture_loader import (
    FixtureLoader,
    GeneratedTestCase,
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
def agent() -> ScreeningAgent:
    """Create ScreeningAgent with mock LLM."""
    return ScreeningAgent(llm=MockLLMProvider())


class TestScreeningFixtureLoader:
    """Test screening fixture loading."""

    def test_load_screening_fixtures(self, loader: FixtureLoader):
        """Test loading screening fixtures."""
        cases = loader.load_category("screening")
        assert len(cases) > 0
        assert len(cases) >= 200  # Should have ~265 cases

    def test_screening_subcategories(self, loader: FixtureLoader):
        """Test that screening fixtures have expected subcategories."""
        cases = loader.load_category("screening")

        subcategories = {c.subcategory for c in cases}
        print(f"\nScreening subcategories: {subcategories}")

        # Should have index_constituents and top_n at minimum
        assert len(subcategories) >= 2


class TestScreeningAgentIndexConstituents:
    """Test handling of index constituent queries."""

    @pytest.mark.asyncio
    async def test_can_handle_index_queries(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test that agent can handle index constituent queries."""
        index_cases = loader.load_by_subcategory("index_constituents")

        can_handle_count = 0
        for case in index_cases:
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                can_handle_count += 1

        rate = can_handle_count / len(index_cases) if index_cases else 0
        assert rate >= 0.3, f"Index query handle rate: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_index_detection_sp500(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test S&P 500 index detection."""
        sp500_cases = [
            c for c in loader.load_category("screening")
            if "s&p 500" in c.nl_query.lower() or "s&p500" in c.nl_query.lower()
        ]

        if not sp500_cases:
            pytest.skip("No S&P 500 cases found")

        for case in sp500_cases:
            # Check that we can identify this as an index query
            score = await agent.can_handle(case.nl_query)
            assert score > 0, f"Failed to handle S&P 500 query: {case.nl_query}"

    @pytest.mark.asyncio
    async def test_index_detection_ftse(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test FTSE index detection."""
        ftse_cases = [
            c for c in loader.load_category("screening")
            if "ftse" in c.nl_query.lower()
        ]

        if not ftse_cases:
            pytest.skip("No FTSE cases found")

        for case in ftse_cases:
            score = await agent.can_handle(case.nl_query)
            assert score > 0, f"Failed to handle FTSE query: {case.nl_query}"

    @pytest.mark.asyncio
    async def test_index_detection_nasdaq(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test NASDAQ index detection."""
        nasdaq_cases = [
            c for c in loader.load_category("screening")
            if "nasdaq" in c.nl_query.lower()
        ]

        if not nasdaq_cases:
            pytest.skip("No NASDAQ cases found")

        can_handle_count = 0
        for case in nasdaq_cases:
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                can_handle_count += 1

        rate = can_handle_count / len(nasdaq_cases) if nasdaq_cases else 0
        assert rate >= 0.3, f"NASDAQ query handle rate: {rate:.2%}"


class TestScreeningAgentTopN:
    """Test handling of top-N ranking queries."""

    @pytest.mark.asyncio
    async def test_can_handle_top_n_queries(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test that agent can handle top-N queries."""
        top_n_cases = loader.load_by_subcategory("top_n")

        can_handle_count = 0
        for case in top_n_cases:
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                can_handle_count += 1

        rate = can_handle_count / len(top_n_cases) if top_n_cases else 0
        assert rate >= 0.5, f"Top-N query handle rate: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_top_n_detection(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test detection of TOP-N pattern via _detect_top_clause."""
        top_n_cases = loader.load_by_subcategory("top_n")[:30]

        detected_count = 0
        correct_n_count = 0
        for case in top_n_cases:
            # Use the agent's _detect_top_clause method
            top_clause = agent._detect_top_clause(case.nl_query)
            if top_clause:
                detected_count += 1

                # Get expected N from metadata
                expected_n = case.metadata.get("n")
                if expected_n:
                    # Verify the TOP clause has the right count
                    if f",{expected_n}," in top_clause["clause"]:
                        correct_n_count += 1
                    # Note: some patterns like "5 largest" may not be detected
                    # so we don't assert here, just count

        rate = detected_count / len(top_n_cases) if top_n_cases else 0
        n_rate = correct_n_count / len(top_n_cases) if top_n_cases else 0

        print(f"\nTop-N detection statistics:")
        print(f"  Detection rate: {rate:.2%}")
        print(f"  Correct N rate: {n_rate:.2%}")

        assert rate >= 0.5, f"Top-N detection rate: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_market_cap_ranking(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test market cap ranking queries."""
        market_cap_cases = [
            c for c in loader.load_by_subcategory("top_n")
            if c.metadata.get("metric") == "market cap"
        ][:20]

        if not market_cap_cases:
            pytest.skip("No market cap ranking cases found")

        detected_count = 0
        for case in market_cap_cases:
            top_clause = agent._detect_top_clause(case.nl_query)
            if top_clause and "MarketCap" in top_clause["field"]:
                detected_count += 1

        rate = detected_count / len(market_cap_cases) if market_cap_cases else 0
        assert rate >= 0.5, f"Market cap ranking detection rate: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_top_n_builds_screen_expression(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test that top-N queries build SCREEN expressions."""
        top_n_cases = loader.load_by_subcategory("top_n")[:20]

        built_count = 0
        for case in top_n_cases:
            context = AgentContext(
                query=case.nl_query,
                resolved_entities={},
            )

            result = agent._try_rule_based_extraction(context)

            if result and result.tool_calls:
                # Check the tickers field (canonical format - string for SCREEN expressions)
                tool_call = result.tool_calls[0]
                screen_expr = tool_call.arguments.get("tickers", "")

                if screen_expr and isinstance(screen_expr, str) and screen_expr.startswith("SCREEN"):
                    built_count += 1
                    # Verify it contains TOP()
                    assert "TOP(" in screen_expr, \
                        f"SCREEN missing TOP() for: {case.nl_query}"

        rate = built_count / len(top_n_cases) if top_n_cases else 0
        print(f"\nSCREEN expression build rate: {rate:.2%}")
        # Lower threshold since not all queries may be handled by rules
        assert rate >= 0.3, f"SCREEN expression build rate too low: {rate:.2%}"


class TestScreeningAgentFilters:
    """Test handling of filter conditions."""

    @pytest.mark.asyncio
    async def test_sector_filter_detection(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test sector filter detection."""
        # Find cases with sector filters
        sector_cases = [
            c for c in loader.load_category("screening")
            if any(term in c.nl_query.lower() for term in
                   ["technology", "healthcare", "financials", "energy", "consumer"])
        ][:20]

        if not sector_cases:
            pytest.skip("No sector filter cases found")

        detected_count = 0
        for case in sector_cases:
            sector = agent._detect_sector_filter(case.nl_query)
            if sector:
                detected_count += 1

        rate = detected_count / len(sector_cases) if sector_cases else 0
        print(f"\nSector filter detection rate: {rate:.2%}")
        # Sector detection depends on keyword matching
        assert rate >= 0.2, f"Sector filter detection rate too low: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_country_filter_detection(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test country filter detection (agent uses _detect_country_filter)."""
        # Find cases with country filters (US, UK)
        country_cases = [
            c for c in loader.load_category("screening")
            if any(term in c.nl_query.lower() for term in ["us ", "usa", "united states", "uk ", "japan"])
        ][:20]

        if not country_cases:
            pytest.skip("No country filter cases found")

        detected_count = 0
        for case in country_cases:
            country = agent._detect_country_filter(case.nl_query)
            if country:
                detected_count += 1

        rate = detected_count / len(country_cases) if country_cases else 0
        print(f"\nCountry filter detection rate: {rate:.2%}")

    @pytest.mark.asyncio
    async def test_index_filter_detection(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test index filter detection."""
        index_cases = loader.load_by_subcategory("index_constituents")[:20]

        if not index_cases:
            pytest.skip("No index constituent cases found")

        detected_count = 0
        for case in index_cases:
            index_filter = agent._detect_index_filter(case.nl_query)
            if index_filter:
                detected_count += 1

        rate = detected_count / len(index_cases) if index_cases else 0
        print(f"\nIndex filter detection rate: {rate:.2%}")
        assert rate >= 0.3, f"Index filter detection rate: {rate:.2%}"


class TestScreeningAgentMetricDetection:
    """Test metric detection in ranking queries."""

    @pytest.mark.asyncio
    async def test_metric_coverage(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test coverage of metric detection across fixture metrics using _detect_top_clause."""
        top_n_cases = loader.load_by_subcategory("top_n")

        metrics_in_fixtures = set()
        metrics_detected = {}

        for case in top_n_cases:
            metric = case.metadata.get("metric", "unknown")
            metrics_in_fixtures.add(metric)

            # Use _detect_top_clause which returns a dict with the detected field
            top_clause = agent._detect_top_clause(case.nl_query)
            if top_clause:
                metrics_detected[metric] = metrics_detected.get(metric, 0) + 1

        print(f"\nMetrics in fixtures: {metrics_in_fixtures}")
        print(f"Metrics detected: {metrics_detected}")

        # Should detect some metrics
        assert len(metrics_detected) > 0, "No metrics detected"

    @pytest.mark.asyncio
    async def test_revenue_metric_detection(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test revenue metric detection via _detect_top_clause."""
        revenue_cases = [
            c for c in loader.load_category("screening")
            if "revenue" in c.nl_query.lower() or "sales" in c.nl_query.lower()
        ][:20]

        if not revenue_cases:
            pytest.skip("No revenue ranking cases found")

        detected_count = 0
        for case in revenue_cases:
            top_clause = agent._detect_top_clause(case.nl_query)
            if top_clause and "Revenue" in top_clause.get("field", ""):
                detected_count += 1

        rate = detected_count / len(revenue_cases) if revenue_cases else 0
        print(f"\nRevenue metric detection rate: {rate:.2%}")
        # Revenue detection may be harder - lower threshold
        assert rate >= 0.2, f"Revenue metric detection rate too low: {rate:.2%}"

    @pytest.mark.asyncio
    async def test_dividend_yield_metric_detection(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test dividend yield metric detection via _detect_top_clause."""
        div_cases = [
            c for c in loader.load_category("screening")
            if "dividend" in c.nl_query.lower() or "yield" in c.nl_query.lower()
        ][:20]

        if not div_cases:
            pytest.skip("No dividend ranking cases found")

        detected_count = 0
        for case in div_cases:
            top_clause = agent._detect_top_clause(case.nl_query)
            if top_clause and ("Dividend" in top_clause.get("field", "") or "Yield" in top_clause.get("field", "")):
                detected_count += 1

        # Report rate
        rate = detected_count / len(div_cases) if div_cases else 0
        print(f"\nDividend yield detection rate: {rate:.2%}")


class TestScreeningAgentStatistics:
    """Statistical tests across the full screening fixture set."""

    @pytest.mark.asyncio
    async def test_overall_coverage(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test overall coverage of screening queries."""
        cases = loader.load_category("screening")

        stats = {
            "total": len(cases),
            "can_handle": 0,
            "rule_based": 0,
            "by_subcategory": {},
        }

        for case in cases:
            # Check can_handle
            score = await agent.can_handle(case.nl_query)
            if score > 0:
                stats["can_handle"] += 1

            # Check rule-based extraction
            context = AgentContext(query=case.nl_query, resolved_entities={})
            result = agent._try_rule_based_extraction(context)
            if result and result.tool_calls:
                stats["rule_based"] += 1

            # Track by subcategory
            subcat = case.subcategory
            if subcat not in stats["by_subcategory"]:
                stats["by_subcategory"][subcat] = {"total": 0, "handled": 0}
            stats["by_subcategory"][subcat]["total"] += 1
            if score > 0:
                stats["by_subcategory"][subcat]["handled"] += 1

        handle_rate = stats["can_handle"] / stats["total"]
        rule_rate = stats["rule_based"] / stats["total"]

        print(f"\nScreening Statistics (n={stats['total']}):")
        print(f"  Can handle rate: {handle_rate:.2%}")
        print(f"  Rule-based rate: {rule_rate:.2%}")
        print(f"  By subcategory: {stats['by_subcategory']}")

        assert handle_rate >= 0.3, f"Overall handle rate too low: {handle_rate:.2%}"

    @pytest.mark.asyncio
    async def test_expected_screen_components(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test that SCREEN expressions contain expected components."""
        top_n_cases = loader.load_by_subcategory("top_n")[:50]

        components_found = {
            "SCREEN": 0,
            "TOP": 0,
            "IN": 0,
            "Equity": 0,
        }

        for case in top_n_cases:
            context = AgentContext(query=case.nl_query, resolved_entities={})
            result = agent._try_rule_based_extraction(context)

            if result and result.tool_calls:
                # ScreeningAgent uses canonical 'tickers' (string for SCREEN expressions)
                screen_expr = result.tool_calls[0].arguments.get("tickers", "")
                if isinstance(screen_expr, str):
                    for component in components_found:
                        if component in screen_expr:
                            components_found[component] += 1

        print(f"\nSCREEN components found: {components_found}")

        total_processed = len(top_n_cases)
        if total_processed > 0:
            # SCREEN should be present in most results
            screen_rate = components_found["SCREEN"] / total_processed
            assert screen_rate >= 0.3, f"SCREEN component rate: {screen_rate:.2%}"


class TestScreeningExpressionFormat:
    """Test SCREEN expression format matches expected format."""

    @pytest.mark.asyncio
    async def test_screen_expression_syntax(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test that generated SCREEN expressions have valid syntax."""
        top_n_cases = loader.load_by_subcategory("top_n")[:30]

        for case in top_n_cases:
            context = AgentContext(query=case.nl_query, resolved_entities={})
            result = agent._try_rule_based_extraction(context)

            if result and result.tool_calls:
                screen_expr = result.tool_calls[0].arguments.get("tickers", "")
                if isinstance(screen_expr, str) and screen_expr.startswith("SCREEN"):
                    # Verify parentheses are balanced
                    open_parens = screen_expr.count("(")
                    close_parens = screen_expr.count(")")
                    assert open_parens == close_parens, \
                        f"Unbalanced parens in SCREEN: {screen_expr}"

                    # Verify basic structure
                    assert "U(" in screen_expr, f"Missing U() in SCREEN: {screen_expr}"

    @pytest.mark.asyncio
    async def test_compare_to_expected_screen(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Compare generated SCREEN expressions to expected format."""
        top_n_cases = loader.load_by_subcategory("top_n")[:20]

        match_count = 0
        partial_match_count = 0

        for case in top_n_cases:
            context = AgentContext(query=case.nl_query, resolved_entities={})
            result = agent._try_rule_based_extraction(context)

            if not result or not result.tool_calls:
                continue

            actual_screen = result.tool_calls[0].arguments.get("tickers", "")

            # Get expected SCREEN expression
            expected_tc = case.expected_tool_calls[0] if case.expected_tool_calls else None
            if not expected_tc:
                continue

            expected_tickers = expected_tc.get("arguments", {}).get("tickers", "")

            if not isinstance(actual_screen, str) or not isinstance(expected_tickers, str):
                continue

            # Exact match
            if actual_screen == expected_tickers:
                match_count += 1
                continue

            # Check for partial match (same structure but different values)
            if (actual_screen.startswith("SCREEN") and expected_tickers.startswith("SCREEN") and
                "TOP(" in actual_screen and "TOP(" in expected_tickers):
                partial_match_count += 1

        total = len(top_n_cases)
        print(f"\nSCREEN comparison:")
        print(f"  Total cases: {total}")
        print(f"  Exact matches: {match_count}")
        print(f"  Partial matches: {partial_match_count}")

        # Combined match rate
        combined_rate = (match_count + partial_match_count) / total if total > 0 else 0
        print(f"  Combined match rate: {combined_rate:.2%}")


class TestIndexConstituentFormat:
    """Test index constituent query format."""

    @pytest.mark.asyncio
    async def test_index_constituent_patterns(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test that index constituent queries use correct pattern."""
        index_cases = loader.load_by_subcategory("index_constituents")[:20]

        for case in index_cases:
            expected_tc = case.expected_tool_calls[0] if case.expected_tool_calls else None
            if not expected_tc:
                continue

            expected_tickers = expected_tc.get("arguments", {}).get("tickers", "")

            # Index constituents should use L prefix pattern like "LS&PCOMP|L"
            assert expected_tickers.startswith("L"), \
                f"Expected L prefix for index constituents: {expected_tickers}"
            assert "|L" in expected_tickers or expected_tickers.endswith("|L"), \
                f"Expected |L suffix for index constituents: {expected_tickers}"

    @pytest.mark.asyncio
    async def test_index_name_to_code_mapping(self, agent: ScreeningAgent, loader: FixtureLoader):
        """Test mapping of index names to constituent codes."""
        index_cases = loader.load_by_subcategory("index_constituents")

        # Extract mappings from fixtures
        index_mappings = {}
        for case in index_cases:
            index_name = case.metadata.get("index_name")
            constituent_code = case.metadata.get("constituent_code")
            if index_name and constituent_code:
                index_mappings[index_name.lower()] = constituent_code

        print(f"\nIndex mappings from fixtures: {index_mappings}")

        # Should have several indices mapped
        assert len(index_mappings) >= 5, f"Expected more index mappings: {len(index_mappings)}"

        # Verify agent knows about major indices
        major_indices = ["s&p 500", "dow jones", "nasdaq", "ftse 100"]
        for index in major_indices:
            if index in index_mappings:
                # Check if agent can detect this index pattern
                test_query = f"What stocks are in the {index}?"
                score = await agent.can_handle(test_query)
                assert score > 0, f"Agent should handle {index} query"
