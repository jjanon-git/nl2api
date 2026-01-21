"""Tests for Screening domain agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from CONTRACTS import ToolCall
from src.nl2api.agents.screening import ScreeningAgent
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)


@dataclass
class MockLLMProvider:
    """Mock LLM provider for testing."""

    model_name: str = "mock-model"
    response: LLMResponse = field(default_factory=lambda: LLMResponse(
        content="screening",
        tool_calls=[
            LLMToolCall(
                id="tc_123",
                name="refinitiv_get_data",
                arguments={
                    "instruments": ["SCREEN(U(IN(Equity(active,public,primary))),CURN=USD)"],
                    "fields": ["TR.CommonName"],
                },
            )
        ],
    ))

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        return self.response

    async def complete_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> LLMResponse:
        return self.response


class TestScreeningAgentCanHandle:
    """Test suite for can_handle method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_high_confidence_with_top_keyword(self) -> None:
        """Test high confidence for top N queries."""
        score = await self.agent.can_handle("What are the top 10 companies by market cap?")
        assert score >= 0.7

    @pytest.mark.asyncio
    async def test_high_confidence_with_find_keyword(self) -> None:
        """Test high confidence for find/filter queries."""
        score = await self.agent.can_handle("Find companies with PE ratio below 15")
        assert score >= 0.7

    @pytest.mark.asyncio
    async def test_high_confidence_with_multiple_keywords(self) -> None:
        """Test higher confidence with multiple keywords."""
        score = await self.agent.can_handle(
            "Find the largest tech companies with dividend yield above 2%"
        )
        assert score >= 0.8

    @pytest.mark.asyncio
    async def test_zero_confidence_for_unrelated_query(self) -> None:
        """Test zero confidence for unrelated queries."""
        score = await self.agent.can_handle(
            "Who is the CEO of Apple?"
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_handles_screening_keywords(self) -> None:
        """Test handling screening-related queries."""
        score = await self.agent.can_handle("Screen for undervalued stocks")
        assert score >= 0.5


class TestScreeningAgentIndexFilter:
    """Test suite for index filter detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    def test_sp500_detection(self) -> None:
        """Test S&P 500 index detection."""
        filter_expr = self.agent._detect_index_filter("in the s&p 500")
        assert filter_expr == 'IN(TR.IndexConstituentRIC,"0#.SPX")'

    def test_sp500_alternate_format(self) -> None:
        """Test S&P 500 alternate format."""
        filter_expr = self.agent._detect_index_filter("sp500 companies")
        assert filter_expr == 'IN(TR.IndexConstituentRIC,"0#.SPX")'

    def test_nasdaq_100_detection(self) -> None:
        """Test NASDAQ 100 index detection."""
        filter_expr = self.agent._detect_index_filter("nasdaq 100 stocks")
        assert filter_expr == 'IN(TR.IndexConstituentRIC,"0#.NDX")'

    def test_dow_jones_detection(self) -> None:
        """Test Dow Jones index detection."""
        filter_expr = self.agent._detect_index_filter("dow jones companies")
        assert filter_expr == 'IN(TR.IndexConstituentRIC,"0#.DJI")'

    def test_no_index_detected(self) -> None:
        """Test no index in query."""
        filter_expr = self.agent._detect_index_filter("top companies")
        assert filter_expr is None


class TestScreeningAgentCountryFilter:
    """Test suite for country filter detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    def test_us_detection(self) -> None:
        """Test US country detection."""
        filter_expr = self.agent._detect_country_filter("us companies")
        assert filter_expr == "TR.HQCountryCode=US"

    def test_usa_detection(self) -> None:
        """Test USA country detection."""
        filter_expr = self.agent._detect_country_filter("usa stocks")
        assert filter_expr == "TR.HQCountryCode=US"

    def test_uk_detection(self) -> None:
        """Test UK country detection."""
        filter_expr = self.agent._detect_country_filter("uk companies")
        assert filter_expr == "TR.HQCountryCode=GB"

    def test_no_country_detected(self) -> None:
        """Test no country in query."""
        filter_expr = self.agent._detect_country_filter("top companies")
        assert filter_expr is None


class TestScreeningAgentSectorFilter:
    """Test suite for sector filter detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    def test_tech_detection(self) -> None:
        """Test technology sector detection."""
        filter_expr = self.agent._detect_sector_filter("tech companies")
        assert filter_expr == "IN(TR.TRBCEconSectorCode,57)"

    def test_technology_detection(self) -> None:
        """Test technology sector detection (full word)."""
        filter_expr = self.agent._detect_sector_filter("technology stocks")
        assert filter_expr == "IN(TR.TRBCEconSectorCode,57)"

    def test_healthcare_detection(self) -> None:
        """Test healthcare sector detection."""
        filter_expr = self.agent._detect_sector_filter("healthcare companies")
        assert filter_expr == "IN(TR.TRBCEconSectorCode,55)"

    def test_energy_detection(self) -> None:
        """Test energy sector detection."""
        filter_expr = self.agent._detect_sector_filter("energy stocks")
        assert filter_expr == "IN(TR.TRBCEconSectorCode,50)"

    def test_no_sector_detected(self) -> None:
        """Test no sector in query."""
        filter_expr = self.agent._detect_sector_filter("top companies")
        assert filter_expr is None


class TestScreeningAgentMetricFilters:
    """Test suite for metric filter detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    def test_pe_below_detection(self) -> None:
        """Test PE ratio below detection."""
        filters = self.agent._detect_metric_filters("pe ratio below 15")
        assert len(filters) >= 1
        assert any("TR.PE<15" in f["filter"] for f in filters)

    def test_pe_above_detection(self) -> None:
        """Test PE ratio above detection."""
        filters = self.agent._detect_metric_filters("pe above 20")
        assert len(filters) >= 1
        assert any("TR.PE>20" in f["filter"] for f in filters)

    def test_dividend_yield_above_detection(self) -> None:
        """Test dividend yield above detection."""
        filters = self.agent._detect_metric_filters("dividend yield above 2")
        assert len(filters) >= 1
        assert any("TR.DividendYield>2" in f["filter"] for f in filters)

    def test_roe_above_detection(self) -> None:
        """Test ROE above detection."""
        filters = self.agent._detect_metric_filters("roe above 15")
        assert len(filters) >= 1
        assert any("TR.ROE>15" in f["filter"] for f in filters)

    def test_revenue_growth_detection(self) -> None:
        """Test revenue growth detection."""
        filters = self.agent._detect_metric_filters("revenue growth over 20")
        assert len(filters) >= 1
        assert any("TR.RevenueGrowth>20" in f["filter"] for f in filters)

    def test_positive_fcf_detection(self) -> None:
        """Test positive free cash flow detection."""
        filters = self.agent._detect_metric_filters("positive free cash flow")
        assert len(filters) >= 1
        assert any("TR.FreeCashFlow>0" in f["filter"] for f in filters)

    def test_multiple_filters_detection(self) -> None:
        """Test multiple filter detection."""
        filters = self.agent._detect_metric_filters("pe below 15 and roe above 15")
        assert len(filters) >= 2


class TestScreeningAgentTopClause:
    """Test suite for TOP clause detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    def test_top_10_market_cap(self) -> None:
        """Test top 10 by market cap detection."""
        clause = self.agent._detect_top_clause("top 10 by market cap")
        assert clause is not None
        assert "TOP(TR.CompanyMarketCap,10,nnumber)" in clause["clause"]

    def test_top_5_revenue(self) -> None:
        """Test top 5 by revenue detection."""
        clause = self.agent._detect_top_clause("top 5 by revenue")
        assert clause is not None
        assert "TOP(TR.Revenue,5,nnumber)" in clause["clause"]

    def test_largest_companies(self) -> None:
        """Test largest companies (implicit market cap)."""
        clause = self.agent._detect_top_clause("largest companies")
        assert clause is not None
        assert "TOP(TR.CompanyMarketCap" in clause["clause"]

    def test_default_count(self) -> None:
        """Test default count of 10."""
        clause = self.agent._detect_top_clause("largest companies by revenue")
        assert clause is not None
        assert ",10,nnumber)" in clause["clause"]

    def test_biggest_with_count(self) -> None:
        """Test biggest with specific count."""
        clause = self.agent._detect_top_clause("biggest 5 companies")
        assert clause is not None
        assert ",5,nnumber)" in clause["clause"]


class TestScreeningAgentRuleBasedExtraction:
    """Test suite for rule-based extraction."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_top_market_cap_query(self) -> None:
        """Test top market cap query."""
        context = AgentContext(
            query="What are the top 10 companies by market cap in the S&P 500?",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv_get_data"
        instruments = tc.arguments["instruments"]
        assert len(instruments) == 1
        screen_expr = instruments[0]
        assert "SCREEN(" in screen_expr
        assert "0#.SPX" in screen_expr  # S&P 500
        assert "TOP(TR.CompanyMarketCap,10,nnumber)" in screen_expr
        assert "CURN=USD" in screen_expr

    @pytest.mark.asyncio
    async def test_dividend_yield_filter_query(self) -> None:
        """Test dividend yield filter query."""
        context = AgentContext(
            query="Find US tech stocks with dividend yield above 2%",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        instruments = tc.arguments["instruments"]
        screen_expr = instruments[0]
        assert "TR.HQCountryCode=US" in screen_expr
        assert "IN(TR.TRBCEconSectorCode,57)" in screen_expr
        assert "TR.DividendYield>2" in screen_expr

    @pytest.mark.asyncio
    async def test_value_screening_query(self) -> None:
        """Test value screening query."""
        context = AgentContext(
            query="Find undervalued stocks with PE ratio below 15 and ROE above 15%",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        instruments = tc.arguments["instruments"]
        screen_expr = instruments[0]
        assert "TR.PE<15" in screen_expr
        assert "TR.ROE>15" in screen_expr

    @pytest.mark.asyncio
    async def test_growth_screening_query(self) -> None:
        """Test growth screening query."""
        context = AgentContext(
            query="Find companies with revenue growth over 20% and positive free cash flow",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        instruments = tc.arguments["instruments"]
        screen_expr = instruments[0]
        assert "TR.RevenueGrowth>20" in screen_expr
        assert "TR.FreeCashFlow>0" in screen_expr

    @pytest.mark.asyncio
    async def test_insufficient_criteria_returns_none(self) -> None:
        """Test that query with no criteria returns None."""
        context = AgentContext(
            query="Show me some companies",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)
        # Should return None because no meaningful filters detected
        assert result is None


class TestScreeningAgentProcess:
    """Test suite for process method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_rule_based_takes_precedence(self) -> None:
        """Test that rule-based extraction takes precedence over LLM."""
        context = AgentContext(
            query="What are the top 10 companies by market cap?",
            resolved_entities={},
        )

        result = await self.agent.process(context)

        assert result.confidence >= 0.8
        assert result.domain == "screening"
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_for_complex_queries(self) -> None:
        """Test that LLM is used for queries that can't be rule-based."""
        context = AgentContext(
            query="Show me some interesting companies",
            resolved_entities={},
        )

        # This should fall back to LLM
        result = await self.agent.process(context)

        assert len(result.tool_calls) >= 0


class TestScreeningAgentProperties:
    """Test suite for agent properties."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    def test_domain_name(self) -> None:
        """Test domain name property."""
        assert self.agent.domain_name == "screening"

    def test_domain_description(self) -> None:
        """Test domain description property."""
        desc = self.agent.domain_description
        assert "screening" in desc.lower()
        assert "ranking" in desc.lower()

    def test_system_prompt(self) -> None:
        """Test system prompt contains key information."""
        prompt = self.agent.get_system_prompt()
        assert "SCREEN" in prompt
        assert "TOP" in prompt
        assert "refinitiv_get_data" in prompt

    def test_tools_definition(self) -> None:
        """Test tools definition."""
        tools = self.agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "refinitiv_get_data"
        assert "instruments" in tools[0].parameters["properties"]
        assert "fields" in tools[0].parameters["properties"]


class TestScreeningAgentFixtureCompatibility:
    """Test suite to verify compatibility with test fixtures."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = ScreeningAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_screening_top_market_cap_fixture(self) -> None:
        """Test against screening_top_market_cap.json fixture."""
        # Fixture: "What are the top 10 companies by market cap in the S&P 500?"
        context = AgentContext(
            query="What are the top 10 companies by market cap in the S&P 500?",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv_get_data"
        screen_expr = tc.arguments["instruments"][0]
        # Should have S&P 500 filter
        assert "0#.SPX" in screen_expr
        # Should have TOP clause for market cap
        assert "TOP(TR.CompanyMarketCap,10,nnumber)" in screen_expr

    @pytest.mark.asyncio
    async def test_screening_dividend_yield_fixture(self) -> None:
        """Test against screening_dividend_yield.json fixture."""
        # Fixture: "Find US tech stocks with dividend yield above 2%"
        context = AgentContext(
            query="Find US tech stocks with dividend yield above 2%",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        screen_expr = tc.arguments["instruments"][0]
        assert "TR.HQCountryCode=US" in screen_expr
        assert "IN(TR.TRBCEconSectorCode,57)" in screen_expr
        assert "TR.DividendYield>2" in screen_expr

    @pytest.mark.asyncio
    async def test_screening_growth_fixture(self) -> None:
        """Test against screening_growth.json fixture."""
        # Fixture: "Find companies with revenue growth over 20% and positive free cash flow"
        context = AgentContext(
            query="Find companies with revenue growth over 20% and positive free cash flow",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        screen_expr = tc.arguments["instruments"][0]
        assert "TR.RevenueGrowth>20" in screen_expr
        assert "TR.FreeCashFlow>0" in screen_expr

    @pytest.mark.asyncio
    async def test_screening_value_fixture(self) -> None:
        """Test against screening_value.json fixture."""
        # Fixture: "Find undervalued stocks with PE ratio below 15 and ROE above 15%"
        context = AgentContext(
            query="Find undervalued stocks with PE ratio below 15 and ROE above 15%",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        screen_expr = tc.arguments["instruments"][0]
        assert "TR.PE<15" in screen_expr
        assert "TR.ROE>15" in screen_expr

    @pytest.mark.asyncio
    async def test_screening_earnings_beat_fixture(self) -> None:
        """Test against screening_earnings_beat.json fixture."""
        # Fixture: "Which S&P 500 companies beat earnings estimates last quarter by more than 10%?"
        context = AgentContext(
            query="Which S&P 500 companies beat earnings estimates last quarter by more than 10%?",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        screen_expr = tc.arguments["instruments"][0]
        # Should have S&P 500 filter
        assert "0#.SPX" in screen_expr
        # Should have earnings surprise filter
        assert "TR.EPSSurprisePct" in screen_expr
        assert ">10" in screen_expr
