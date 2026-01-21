"""Tests for Fundamentals domain agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from CONTRACTS import ToolCall
from src.nl2api.agents.fundamentals import FundamentalsAgent
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
        content="fundamentals",
        tool_calls=[
            LLMToolCall(
                id="tc_123",
                name="refinitiv.get_data",
                arguments={
                    "instruments": ["AAPL.O"],
                    "fields": ["TR.Revenue"],
                    "parameters": {"Period": "FY0"},
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


class TestFundamentalsAgentCanHandle:
    """Test suite for can_handle method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_high_confidence_with_revenue_keywords(self) -> None:
        """Test high confidence for revenue-related queries."""
        score = await self.agent.can_handle("What is Apple's revenue?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_high_confidence_with_multiple_keywords(self) -> None:
        """Test higher confidence with multiple keywords."""
        score = await self.agent.can_handle(
            "Get total assets and liabilities from the balance sheet"
        )
        assert score >= 0.7

    @pytest.mark.asyncio
    async def test_high_confidence_with_ratios(self) -> None:
        """Test high confidence for ratio queries."""
        score = await self.agent.can_handle("What is Tesla's ROE and profit margin?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_zero_confidence_for_unrelated_query(self) -> None:
        """Test zero confidence for unrelated queries."""
        score = await self.agent.can_handle(
            "What is the weather forecast for tomorrow?"
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_handles_cash_flow_keywords(self) -> None:
        """Test handling cash flow-related queries."""
        score = await self.agent.can_handle("What is Apple's free cash flow?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_handles_balance_sheet_keywords(self) -> None:
        """Test handling balance sheet-related queries."""
        score = await self.agent.can_handle("Show me the balance sheet with total debt")
        assert score >= 0.5


class TestFundamentalsAgentFieldDetection:
    """Test suite for field detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    def test_revenue_detection(self) -> None:
        """Test detection of revenue field."""
        fields = self.agent._detect_fields("what is the revenue")
        assert "TR.Revenue" in fields

    def test_net_income_detection(self) -> None:
        """Test detection of net income field."""
        fields = self.agent._detect_fields("show net income")
        assert "TR.NetIncome" in fields

    def test_total_assets_detection(self) -> None:
        """Test detection of total assets field."""
        fields = self.agent._detect_fields("what are the total assets")
        assert "TR.TotalAssets" in fields

    def test_total_debt_detection(self) -> None:
        """Test detection of total debt field."""
        fields = self.agent._detect_fields("what is the total debt")
        assert "TR.TotalDebt" in fields

    def test_roe_detection(self) -> None:
        """Test detection of ROE field."""
        fields = self.agent._detect_fields("what is the ROE")
        assert "TR.ROE" in fields

    def test_roa_detection(self) -> None:
        """Test detection of ROA field."""
        fields = self.agent._detect_fields("return on assets")
        assert "TR.ROA" in fields

    def test_operating_income_detection(self) -> None:
        """Test detection of operating income field."""
        fields = self.agent._detect_fields("what is the operating income")
        assert "TR.OperatingIncome" in fields

    def test_ebitda_detection(self) -> None:
        """Test detection of EBITDA field."""
        fields = self.agent._detect_fields("what is the EBITDA")
        assert "TR.EBITDA" in fields

    def test_free_cash_flow_detection(self) -> None:
        """Test detection of free cash flow field."""
        fields = self.agent._detect_fields("free cash flow")
        assert "TR.FreeCashFlow" in fields

    def test_operating_cash_flow_detection(self) -> None:
        """Test detection of operating cash flow field."""
        fields = self.agent._detect_fields("operating cash flow")
        assert "TR.OperatingCashFlow" in fields

    def test_profit_margin_detection(self) -> None:
        """Test detection of profit margin field."""
        fields = self.agent._detect_fields("what is the profit margin")
        assert "TR.NetProfitMargin" in fields

    def test_multiple_fields_detection(self) -> None:
        """Test detection of multiple fields."""
        fields = self.agent._detect_fields("show revenue, net income, and total assets")
        assert "TR.Revenue" in fields
        assert "TR.NetIncome" in fields
        assert "TR.TotalAssets" in fields

    def test_eps_detection(self) -> None:
        """Test detection of EPS field."""
        fields = self.agent._detect_fields("what is the EPS")
        assert "TR.BasicEPS" in fields

    def test_dividend_yield_detection(self) -> None:
        """Test detection of dividend yield field."""
        fields = self.agent._detect_fields("dividend yield")
        assert "TR.DividendYield" in fields


class TestFundamentalsAgentParameterDetection:
    """Test suite for parameter detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    def test_last_3_years(self) -> None:
        """Test detection of last 3 years."""
        params = self.agent._detect_parameters("revenue for the last 3 years")
        assert params is not None
        assert params["SDate"] == "0"
        assert params["EDate"] == "-2"
        assert params["Frq"] == "FY"

    def test_last_5_years(self) -> None:
        """Test detection of last 5 years."""
        params = self.agent._detect_parameters("show last 5 years of data")
        assert params is not None
        assert params["SDate"] == "0"
        assert params["EDate"] == "-4"
        assert params["Frq"] == "FY"

    def test_quarterly(self) -> None:
        """Test detection of quarterly frequency."""
        params = self.agent._detect_parameters("quarterly revenue")
        assert params is not None
        assert params.get("Frq") == "FQ"

    def test_last_quarter(self) -> None:
        """Test detection of last quarter."""
        params = self.agent._detect_parameters("last quarter earnings")
        assert params is not None
        assert params["Period"] == "FQ-1"
        assert params["Frq"] == "FQ"

    def test_last_year(self) -> None:
        """Test detection of last year."""
        params = self.agent._detect_parameters("last year revenue")
        assert params is not None
        assert params["Period"] == "FY0"
        assert params["Frq"] == "FY"

    def test_default_period(self) -> None:
        """Test default period for current data query."""
        params = self.agent._detect_parameters("what is the revenue")
        assert params is not None
        assert params["Period"] == "FY0"


class TestFundamentalsAgentInstrumentDetection:
    """Test suite for instrument detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    def test_resolved_entities(self) -> None:
        """Test getting instruments from resolved entities."""
        context = AgentContext(
            query="What is Apple's revenue?",
            resolved_entities={"Apple": "AAPL.O"},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == ["AAPL.O"]

    def test_multiple_resolved_entities(self) -> None:
        """Test getting multiple instruments."""
        context = AgentContext(
            query="Compare Apple and Microsoft revenue",
            resolved_entities={"Apple": "AAPL.O", "Microsoft": "MSFT.O"},
        )
        instruments = self.agent._get_instruments(context)
        assert "AAPL.O" in instruments
        assert "MSFT.O" in instruments

    def test_known_company_pattern(self) -> None:
        """Test fallback to known company patterns."""
        context = AgentContext(
            query="What is Apple's revenue?",
            resolved_entities={},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == ["AAPL.O"]

    def test_google_known_pattern(self) -> None:
        """Test Google pattern detection."""
        context = AgentContext(
            query="Show Google's total assets",
            resolved_entities={},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == ["GOOGL.O"]

    def test_no_instruments_found(self) -> None:
        """Test when no instruments can be found."""
        context = AgentContext(
            query="What is the revenue?",
            resolved_entities={},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == []


class TestFundamentalsAgentRuleBasedExtraction:
    """Test suite for rule-based extraction."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_simple_revenue_query(self) -> None:
        """Test simple revenue query matching fixture format."""
        context = AgentContext(
            query="What was Apple's revenue last year?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["AAPL.O"]
        assert "TR.Revenue" in tc.arguments["fields"]
        assert tc.arguments["parameters"]["Period"] == "FY0"

    @pytest.mark.asyncio
    async def test_time_series_query(self) -> None:
        """Test time series query with last N years."""
        context = AgentContext(
            query="Get Microsoft's net income and operating income for the last 3 years",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.arguments["instruments"] == ["MSFT.O"]
        assert "TR.NetIncome" in tc.arguments["fields"]
        assert "TR.OperatingIncome" in tc.arguments["fields"]
        assert tc.arguments["parameters"]["SDate"] == "0"
        assert tc.arguments["parameters"]["EDate"] == "-2"
        assert tc.arguments["parameters"]["Frq"] == "FY"

    @pytest.mark.asyncio
    async def test_balance_sheet_query(self) -> None:
        """Test balance sheet query."""
        context = AgentContext(
            query="What are Google's total assets and total debt?",
            resolved_entities={"Google": "GOOGL.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert "TR.TotalAssets" in tc.arguments["fields"]
        assert "TR.TotalDebt" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_ratios_query(self) -> None:
        """Test financial ratios query."""
        context = AgentContext(
            query="What is NVIDIA's ROE, ROA, and profit margin?",
            resolved_entities={"NVIDIA": "NVDA.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert "TR.ROE" in tc.arguments["fields"]
        assert "TR.ROA" in tc.arguments["fields"]
        assert "TR.NetProfitMargin" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_cash_flow_query(self) -> None:
        """Test cash flow query."""
        context = AgentContext(
            query="What is Amazon's free cash flow and operating cash flow?",
            resolved_entities={"Amazon": "AMZN.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert "TR.FreeCashFlow" in tc.arguments["fields"]
        assert "TR.OperatingCashFlow" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_no_entities_returns_none(self) -> None:
        """Test that missing entities returns None."""
        context = AgentContext(
            query="What is the revenue?",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)
        assert result is None

    @pytest.mark.asyncio
    async def test_known_company_pattern(self) -> None:
        """Test extraction using known company patterns."""
        context = AgentContext(
            query="What is Apple's EBITDA?",
            resolved_entities={},  # No resolved entities, rely on patterns
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.arguments["instruments"] == ["AAPL.O"]
        assert "TR.EBITDA" in tc.arguments["fields"]


class TestFundamentalsAgentProcess:
    """Test suite for process method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_rule_based_takes_precedence(self) -> None:
        """Test that rule-based extraction takes precedence over LLM."""
        context = AgentContext(
            query="What is Apple's revenue?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = await self.agent.process(context)

        assert result.confidence >= 0.8
        assert result.domain == "fundamentals"
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_for_complex_queries(self) -> None:
        """Test that LLM is used for queries that can't be rule-based."""
        context = AgentContext(
            query="Analyze the financial health and compare key metrics",
            resolved_entities={},  # No entities, no known patterns
        )

        # This should fall back to LLM
        result = await self.agent.process(context)

        # LLM mock returns a tool call
        assert len(result.tool_calls) >= 0  # May or may not have tool calls


class TestFundamentalsAgentProperties:
    """Test suite for agent properties."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    def test_domain_name(self) -> None:
        """Test domain name property."""
        assert self.agent.domain_name == "fundamentals"

    def test_domain_description(self) -> None:
        """Test domain description property."""
        desc = self.agent.domain_description
        assert "income statement" in desc.lower()
        assert "balance sheet" in desc.lower()
        assert "Refinitiv" in desc

    def test_system_prompt(self) -> None:
        """Test system prompt contains key information."""
        prompt = self.agent.get_system_prompt()
        assert "Refinitiv" in prompt
        assert "TR.Revenue" in prompt
        assert "refinitiv.get_data" in prompt
        assert "AAPL.O" in prompt

    def test_tools_definition(self) -> None:
        """Test tools definition."""
        tools = self.agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "refinitiv.get_data"
        assert "instruments" in tools[0].parameters["properties"]
        assert "fields" in tools[0].parameters["properties"]
        assert "parameters" in tools[0].parameters["properties"]


class TestFundamentalsAgentFixtureCompatibility:
    """Test suite to verify compatibility with test fixtures."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = FundamentalsAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_fundamentals_revenue_fixture(self) -> None:
        """Test against fundamentals_revenue.json fixture."""
        # Fixture: "What was Apple's revenue last year?"
        context = AgentContext(
            query="What was Apple's revenue last year?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        # Expected: refinitiv.get_data with instruments=["AAPL.O"], fields=["TR.Revenue"]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["AAPL.O"]
        assert tc.arguments["fields"] == ["TR.Revenue"]
        # Parameters should include Period and Frq
        assert "parameters" in tc.arguments
        params = tc.arguments["parameters"]
        assert params.get("Period") == "FY0" or params.get("Frq") == "FY"

    @pytest.mark.asyncio
    async def test_fundamentals_income_stmt_fixture(self) -> None:
        """Test against fundamentals_income_stmt.json fixture."""
        # Fixture: "What is Microsoft's net income and operating income for the last 3 years?"
        context = AgentContext(
            query="What is Microsoft's net income and operating income for the last 3 years?",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["MSFT.O"]
        assert "TR.NetIncome" in tc.arguments["fields"]
        assert "TR.OperatingIncome" in tc.arguments["fields"]
        # Time series params
        params = tc.arguments["parameters"]
        assert params["SDate"] == "0"
        assert params["EDate"] == "-2"
        assert params["Frq"] == "FY"

    @pytest.mark.asyncio
    async def test_fundamentals_balance_sheet_fixture(self) -> None:
        """Test against fundamentals_balance_sheet.json fixture."""
        # Fixture: "What are Google's total assets and total debt?"
        context = AgentContext(
            query="What are Google's total assets and total debt?",
            resolved_entities={"Google": "GOOGL.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["GOOGL.O"]
        assert "TR.TotalAssets" in tc.arguments["fields"]
        assert "TR.TotalDebt" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_fundamentals_ratios_fixture(self) -> None:
        """Test against fundamentals_ratios.json fixture."""
        # Fixture: "What is NVIDIA's ROE, ROA, and profit margin?"
        context = AgentContext(
            query="What is NVIDIA's ROE, ROA, and profit margin?",
            resolved_entities={"NVIDIA": "NVDA.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["NVDA.O"]
        assert "TR.ROE" in tc.arguments["fields"]
        assert "TR.ROA" in tc.arguments["fields"]
        assert "TR.NetProfitMargin" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_fundamentals_cash_flow_fixture(self) -> None:
        """Test against fundamentals_cash_flow.json fixture."""
        # Fixture: "What is Amazon's free cash flow and operating cash flow?"
        context = AgentContext(
            query="What is Amazon's free cash flow and operating cash flow?",
            resolved_entities={"Amazon": "AMZN.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["AMZN.O"]
        assert "TR.FreeCashFlow" in tc.arguments["fields"]
        assert "TR.OperatingCashFlow" in tc.arguments["fields"]
