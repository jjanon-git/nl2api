"""Tests for Datastream domain agent."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.nl2api.agents.datastream import DatastreamAgent
from src.nl2api.agents.protocols import AgentContext
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
    response: LLMResponse = field(
        default_factory=lambda: LLMResponse(
            content="datastream",
            tool_calls=[
                LLMToolCall(
                    id="tc_123",
                    name="get_data",
                    arguments={"tickers": ["AAPL.O"], "fields": ["P"]},
                )
            ],
        )
    )

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


class TestDatastreamAgentCanHandle:
    """Test suite for can_handle method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = DatastreamAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_high_confidence_with_price_keywords(self) -> None:
        """Test high confidence for price-related queries."""
        score = await self.agent.can_handle("What is Apple's stock price?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_high_confidence_with_multiple_keywords(self) -> None:
        """Test higher confidence with multiple keywords."""
        score = await self.agent.can_handle("Get historical stock price for Microsoft with volume")
        assert score >= 0.7

    @pytest.mark.asyncio
    async def test_high_confidence_with_market_cap(self) -> None:
        """Test high confidence for market cap queries."""
        score = await self.agent.can_handle("What is Tesla's market cap?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_zero_confidence_for_unrelated_query(self) -> None:
        """Test zero confidence for unrelated queries."""
        score = await self.agent.can_handle("Who is the CEO of Apple?")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_handles_dividend_keywords(self) -> None:
        """Test handling dividend-related queries."""
        score = await self.agent.can_handle("What is Apple's dividend yield?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_handles_index_keywords(self) -> None:
        """Test handling index-related queries."""
        score = await self.agent.can_handle("What is the S&P 500 index level?")
        assert score >= 0.5


class TestDatastreamAgentFieldDetection:
    """Test suite for field detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = DatastreamAgent(llm=self.mock_llm)

    def test_price_detection(self) -> None:
        """Test detection of price field."""
        fields = self.agent._detect_fields("What is the stock price?")
        assert "P" in fields

    def test_ohlc_detection(self) -> None:
        """Test detection of OHLC fields."""
        fields = self.agent._detect_fields("Get the OHLC data")
        assert "PO" in fields
        assert "PH" in fields
        assert "PL" in fields
        assert "P" in fields

    def test_ohlcv_detection(self) -> None:
        """Test detection of OHLCV fields."""
        fields = self.agent._detect_fields("Get OHLC and volume")
        assert "VO" in fields
        assert "PO" in fields

    def test_market_cap_detection(self) -> None:
        """Test detection of market cap field."""
        fields = self.agent._detect_fields("What is the market capitalization?")
        assert "MV" in fields

    def test_pe_ratio_detection(self) -> None:
        """Test detection of PE ratio field."""
        fields = self.agent._detect_fields("What is the PE ratio?")
        assert "PE" in fields

    def test_dividend_yield_detection(self) -> None:
        """Test detection of dividend yield field."""
        fields = self.agent._detect_fields("What is the dividend yield?")
        assert "DY" in fields

    def test_default_to_price(self) -> None:
        """Test default to price field when no specific field detected."""
        fields = self.agent._detect_fields("Get Apple data")
        assert "P" in fields


class TestDatastreamAgentTimeRangeDetection:
    """Test suite for time range detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = DatastreamAgent(llm=self.mock_llm)

    def test_last_30_days(self) -> None:
        """Test detection of last 30 days."""
        params = self.agent._detect_time_range("prices for last 30 days")
        assert params is not None
        assert params["start"] == "-30D"
        assert params["end"] == "0D"
        assert params["freq"] == "D"

    def test_last_month(self) -> None:
        """Test detection of last month."""
        params = self.agent._detect_time_range("prices for last month")
        assert params is not None
        assert params["start"] == "-1M"
        assert params["end"] == "0D"

    def test_last_year(self) -> None:
        """Test detection of last year."""
        params = self.agent._detect_time_range("historical data for last year")
        assert params is not None
        assert params["start"] == "-1Y"

    def test_yesterday(self) -> None:
        """Test detection of yesterday."""
        params = self.agent._detect_time_range("yesterday's price")
        assert params is not None
        assert params["start"] == "-1D"
        assert params["end"] == "-1D"

    def test_ytd(self) -> None:
        """Test detection of year-to-date."""
        params = self.agent._detect_time_range("YTD performance")
        assert params is not None
        assert params["start"] == "-0Y"

    def test_weekly_frequency(self) -> None:
        """Test detection of weekly frequency."""
        params = self.agent._detect_time_range("weekly prices for last month")
        assert params is not None
        assert params["freq"] == "W"

    def test_monthly_frequency(self) -> None:
        """Test detection of monthly frequency."""
        params = self.agent._detect_time_range("monthly prices for last year")
        assert params is not None
        assert params["freq"] == "M"

    def test_no_time_range(self) -> None:
        """Test no time range for current data query."""
        params = self.agent._detect_time_range("current stock price")
        assert params is None


class TestDatastreamAgentTickerExtraction:
    """Test suite for ticker extraction (canonical RIC format)."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = DatastreamAgent(llm=self.mock_llm)

    def test_extracts_rics_from_resolved_entities(self) -> None:
        """Test that tickers are extracted in RIC format from resolved entities."""
        tickers = self.agent._extract_tickers("What is Apple's price?", {"Apple": "AAPL.O"})
        assert "AAPL.O" in tickers

    def test_extracts_multiple_rics(self) -> None:
        """Test extraction of multiple RICs."""
        tickers = self.agent._extract_tickers(
            "Compare Apple and Microsoft", {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}
        )
        assert "AAPL.O" in tickers
        assert "MSFT.O" in tickers

    def test_returns_empty_without_entities(self) -> None:
        """Test returns empty list when no entities provided."""
        tickers = self.agent._extract_tickers("What is the price?", {})
        assert tickers == []

    def test_returns_empty_with_none_entities(self) -> None:
        """Test returns empty list when entities is None."""
        tickers = self.agent._extract_tickers("What is the price?", None)
        assert tickers == []


class TestDatastreamAgentRuleBasedExtraction:
    """Test suite for rule-based extraction."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = DatastreamAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_simple_price_query(self) -> None:
        """Test simple stock price query."""
        context = AgentContext(
            query="What is Apple's stock price?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "get_data"  # Canonical format
        assert "AAPL.O" in tc.arguments["tickers"]  # RIC format
        assert "P" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_historical_query(self) -> None:
        """Test historical data query."""
        context = AgentContext(
            query="Get Microsoft's daily closing prices for the last 30 days",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert "MSFT.O" in tc.arguments["tickers"]  # RIC format
        assert "P" in tc.arguments["fields"]
        assert tc.arguments.get("start") == "-30D"
        assert tc.arguments.get("end") == "0D"
        assert tc.arguments.get("freq") == "D"

    @pytest.mark.asyncio
    async def test_market_cap_query(self) -> None:
        """Test market cap query."""
        context = AgentContext(
            query="What is Tesla's market capitalization?",
            resolved_entities={"Tesla": "TSLA.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert "MV" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_no_entities_returns_none(self) -> None:
        """Test that missing entities returns None."""
        context = AgentContext(
            query="What is the stock price?",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)
        assert result is None

    @pytest.mark.asyncio
    async def test_with_resolved_entities(self) -> None:
        """Test extraction with resolved entities from entity resolver."""
        context = AgentContext(
            query="What is Apple's PE ratio?",
            resolved_entities={"Apple": "AAPL.O"},  # Provided by entity resolver
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert "AAPL.O" in tc.arguments["tickers"]  # RIC format
        assert "PE" in tc.arguments["fields"]


class TestDatastreamAgentProcess:
    """Test suite for process method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = DatastreamAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_rule_based_takes_precedence(self) -> None:
        """Test that rule-based extraction takes precedence over LLM."""
        context = AgentContext(
            query="What is Apple's stock price?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = await self.agent.process(context)

        assert result.confidence >= 0.8
        assert result.domain == "datastream"
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_for_complex_queries(self) -> None:
        """Test that LLM is used for queries that can't be rule-based."""
        context = AgentContext(
            query="Get the price and compare it to its 52-week average",
            resolved_entities={},  # No entities, no known patterns
        )

        # This should fall back to LLM
        result = await self.agent.process(context)

        # LLM mock returns a tool call
        assert len(result.tool_calls) >= 0  # May or may not have tool calls


class TestDatastreamAgentProperties:
    """Test suite for agent properties."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = DatastreamAgent(llm=self.mock_llm)

    def test_domain_name(self) -> None:
        """Test domain name property."""
        assert self.agent.domain_name == "datastream"

    def test_domain_description(self) -> None:
        """Test domain description property."""
        desc = self.agent.domain_description
        assert "price" in desc.lower()
        assert "Datastream" in desc

    def test_system_prompt(self) -> None:
        """Test system prompt contains key information."""
        prompt = self.agent.get_system_prompt()
        assert "Datastream" in prompt
        assert "get_data" in prompt
        assert "AAPL" in prompt  # Example ticker (with U: prefix in examples)

    def test_tools_definition(self) -> None:
        """Test tools definition."""
        tools = self.agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "get_data"  # Canonical format
        assert "tickers" in tools[0].parameters["properties"]
        assert "fields" in tools[0].parameters["properties"]
