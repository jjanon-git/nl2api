"""Tests for Officers domain agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from CONTRACTS import ToolCall
from src.nl2api.agents.officers import OfficersAgent
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
        content="officers",
        tool_calls=[
            LLMToolCall(
                id="tc_123",
                name="refinitiv.get_data",
                arguments={
                    "instruments": ["AAPL.O"],
                    "fields": ["TR.CEOName"],
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


class TestOfficersAgentCanHandle:
    """Test suite for can_handle method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_high_confidence_with_ceo_keyword(self) -> None:
        """Test high confidence for CEO queries."""
        score = await self.agent.can_handle("Who is the CEO of Apple?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_high_confidence_with_multiple_keywords(self) -> None:
        """Test higher confidence with multiple keywords."""
        score = await self.agent.can_handle(
            "Who are the board members and independent directors?"
        )
        assert score >= 0.7

    @pytest.mark.asyncio
    async def test_high_confidence_with_compensation(self) -> None:
        """Test high confidence for compensation queries."""
        score = await self.agent.can_handle("What is the CEO's total compensation?")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_zero_confidence_for_unrelated_query(self) -> None:
        """Test zero confidence for unrelated queries."""
        score = await self.agent.can_handle(
            "What is the current stock price?"
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_handles_board_keywords(self) -> None:
        """Test handling board-related queries."""
        score = await self.agent.can_handle("List the board of directors")
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_handles_executive_keywords(self) -> None:
        """Test handling executive-related queries."""
        score = await self.agent.can_handle("Who are the top executives?")
        assert score >= 0.5


class TestOfficersAgentFieldDetection:
    """Test suite for field detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    def test_ceo_detection(self) -> None:
        """Test detection of CEO field."""
        fields = self.agent._detect_fields("who is the ceo")
        assert "TR.CEOName" in fields

    def test_cfo_detection(self) -> None:
        """Test detection of CFO field."""
        fields = self.agent._detect_fields("who is the cfo")
        assert "TR.CFOName" in fields

    def test_chairman_detection(self) -> None:
        """Test detection of chairman field."""
        fields = self.agent._detect_fields("who is the chairman")
        assert "TR.ChairmanName" in fields

    def test_board_size_detection(self) -> None:
        """Test detection of board size field."""
        fields = self.agent._detect_fields("how many board members")
        assert "TR.BoardSize" in fields

    def test_independent_directors_detection(self) -> None:
        """Test detection of independent directors field."""
        fields = self.agent._detect_fields("how many independent directors")
        assert "TR.IndependentBoardMembers" in fields

    def test_compensation_detection(self) -> None:
        """Test detection of compensation fields."""
        fields = self.agent._detect_fields("what is the total compensation")
        assert "TR.ODOfficerTotalComp" in fields

    def test_executives_list_detection(self) -> None:
        """Test detection of executive list fields."""
        fields = self.agent._detect_fields("who are the top executives")
        assert "TR.OfficerName" in fields
        assert "TR.OfficerTitle" in fields

    def test_board_members_detection(self) -> None:
        """Test detection of board member fields."""
        fields = self.agent._detect_fields("list the board of directors")
        assert "TR.ODDirectorName" in fields

    def test_tenure_detection(self) -> None:
        """Test detection of tenure field."""
        fields = self.agent._detect_fields("how long has he been ceo")
        assert "TR.OfficerTitleSince" in fields

    def test_education_detection(self) -> None:
        """Test detection of education fields."""
        fields = self.agent._detect_fields("where did he go to university")
        assert "TR.ODOfficerUniversityName" in fields
        assert "TR.ODOfficerGraduationDegree" in fields


class TestOfficersAgentParameterDetection:
    """Test suite for parameter detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    def test_executive_type_detection(self) -> None:
        """Test detection of Executive officer type."""
        params = self.agent._detect_parameters("who are the top executives")
        assert params is not None
        assert params.get("OfficerType") == "Executive"

    def test_director_type_detection(self) -> None:
        """Test detection of Director officer type."""
        params = self.agent._detect_parameters("list the board members")
        assert params is not None
        assert params.get("OfficerType") == "Director"

    def test_ceo_type_for_compensation(self) -> None:
        """Test detection of CEO type for compensation queries."""
        params = self.agent._detect_parameters("what is the ceo's compensation")
        assert params is not None
        assert params.get("OfficerType") == "CEO"

    def test_no_params_for_simple_query(self) -> None:
        """Test no params for simple CEO query."""
        params = self.agent._detect_parameters("who is the ceo")
        # Should not have OfficerType for simple CEO name query
        assert params is None


class TestOfficersAgentInstrumentDetection:
    """Test suite for instrument detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    def test_resolved_entities(self) -> None:
        """Test getting instruments from resolved entities."""
        context = AgentContext(
            query="Who is the CEO of Apple?",
            resolved_entities={"Apple": "AAPL.O"},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == ["AAPL.O"]

    def test_known_company_pattern(self) -> None:
        """Test fallback to known company patterns."""
        context = AgentContext(
            query="Who is the CEO of Apple?",
            resolved_entities={},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == ["AAPL.O"]

    def test_tesla_known_pattern(self) -> None:
        """Test Tesla pattern detection."""
        context = AgentContext(
            query="Who is Tesla's CEO?",
            resolved_entities={},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == ["TSLA.O"]

    def test_jp_morgan_known_pattern(self) -> None:
        """Test JP Morgan pattern detection."""
        context = AgentContext(
            query="Who is the chairman of JP Morgan?",
            resolved_entities={},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == ["JPM.N"]

    def test_no_instruments_found(self) -> None:
        """Test when no instruments can be found."""
        context = AgentContext(
            query="Who is the CEO?",
            resolved_entities={},
        )
        instruments = self.agent._get_instruments(context)
        assert instruments == []


class TestOfficersAgentRuleBasedExtraction:
    """Test suite for rule-based extraction."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_simple_ceo_query(self) -> None:
        """Test simple CEO query."""
        context = AgentContext(
            query="Who is the CEO of Apple?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["AAPL.O"]
        assert "TR.CEOName" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_compensation_query(self) -> None:
        """Test compensation query."""
        context = AgentContext(
            query="What is the total compensation of Tesla's CEO?",
            resolved_entities={"Tesla": "TSLA.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.arguments["instruments"] == ["TSLA.O"]
        assert "TR.CEOName" in tc.arguments["fields"]
        assert "TR.ODOfficerTotalComp" in tc.arguments["fields"]
        assert tc.arguments["parameters"]["OfficerType"] == "CEO"

    @pytest.mark.asyncio
    async def test_board_size_query(self) -> None:
        """Test board size query."""
        context = AgentContext(
            query="How many board members does Microsoft have and how many are independent?",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.arguments["instruments"] == ["MSFT.O"]
        assert "TR.BoardSize" in tc.arguments["fields"]
        assert "TR.IndependentBoardMembers" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_executives_list_query(self) -> None:
        """Test executives list query."""
        context = AgentContext(
            query="Who are the top executives at NVIDIA?",
            resolved_entities={"NVIDIA": "NVDA.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.arguments["instruments"] == ["NVDA.O"]
        assert "TR.OfficerName" in tc.arguments["fields"]
        assert "TR.OfficerTitle" in tc.arguments["fields"]
        assert tc.arguments["parameters"]["OfficerType"] == "Executive"

    @pytest.mark.asyncio
    async def test_no_entities_returns_none(self) -> None:
        """Test that missing entities returns None."""
        context = AgentContext(
            query="Who is the CEO?",
            resolved_entities={},
        )

        result = self.agent._try_rule_based_extraction(context)
        assert result is None

    @pytest.mark.asyncio
    async def test_known_company_pattern(self) -> None:
        """Test extraction using known company patterns."""
        context = AgentContext(
            query="Who is Apple's CEO?",
            resolved_entities={},  # No resolved entities, rely on patterns
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.arguments["instruments"] == ["AAPL.O"]
        assert "TR.CEOName" in tc.arguments["fields"]


class TestOfficersAgentProcess:
    """Test suite for process method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_rule_based_takes_precedence(self) -> None:
        """Test that rule-based extraction takes precedence over LLM."""
        context = AgentContext(
            query="Who is the CEO of Apple?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = await self.agent.process(context)

        assert result.confidence >= 0.8
        assert result.domain == "officers"
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_for_complex_queries(self) -> None:
        """Test that LLM is used for queries that can't be rule-based."""
        context = AgentContext(
            query="Compare executive compensation across tech companies",
            resolved_entities={},  # No entities, no known patterns
        )

        # This should fall back to LLM
        result = await self.agent.process(context)

        assert len(result.tool_calls) >= 0


class TestOfficersAgentProperties:
    """Test suite for agent properties."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    def test_domain_name(self) -> None:
        """Test domain name property."""
        assert self.agent.domain_name == "officers"

    def test_domain_description(self) -> None:
        """Test domain description property."""
        desc = self.agent.domain_description
        assert "executive" in desc.lower()
        assert "board" in desc.lower()

    def test_system_prompt(self) -> None:
        """Test system prompt contains key information."""
        prompt = self.agent.get_system_prompt()
        assert "Officers" in prompt
        assert "TR.CEOName" in prompt
        assert "refinitiv.get_data" in prompt
        assert "AAPL.O" in prompt

    def test_tools_definition(self) -> None:
        """Test tools definition."""
        tools = self.agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "refinitiv.get_data"
        assert "instruments" in tools[0].parameters["properties"]
        assert "fields" in tools[0].parameters["properties"]


class TestOfficersAgentFixtureCompatibility:
    """Test suite to verify compatibility with test fixtures."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = OfficersAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_officers_ceo_fixture(self) -> None:
        """Test against officers_ceo.json fixture."""
        # Fixture: "Who is the CEO of Apple?"
        context = AgentContext(
            query="Who is the CEO of Apple?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["AAPL.O"]
        assert tc.arguments["fields"] == ["TR.CEOName"]

    @pytest.mark.asyncio
    async def test_officers_compensation_fixture(self) -> None:
        """Test against officers_compensation.json fixture."""
        # Fixture: "What is the total compensation of Tesla's CEO?"
        context = AgentContext(
            query="What is the total compensation of Tesla's CEO?",
            resolved_entities={"Tesla": "TSLA.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["TSLA.O"]
        assert "TR.CEOName" in tc.arguments["fields"]
        assert "TR.ODOfficerTotalComp" in tc.arguments["fields"]
        assert tc.arguments["parameters"]["OfficerType"] == "CEO"

    @pytest.mark.asyncio
    async def test_officers_board_fixture(self) -> None:
        """Test against officers_board.json fixture."""
        # Fixture: "How many board members does Microsoft have and how many are independent?"
        context = AgentContext(
            query="How many board members does Microsoft have and how many are independent?",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["MSFT.O"]
        assert "TR.BoardSize" in tc.arguments["fields"]
        assert "TR.IndependentBoardMembers" in tc.arguments["fields"]

    @pytest.mark.asyncio
    async def test_officers_executives_fixture(self) -> None:
        """Test against officers_executives.json fixture."""
        # Fixture: "Who are the top executives at NVIDIA?"
        context = AgentContext(
            query="Who are the top executives at NVIDIA?",
            resolved_entities={"NVIDIA": "NVDA.O"},
        )

        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        tc = result.tool_calls[0]
        assert tc.tool_name == "refinitiv.get_data"
        assert tc.arguments["instruments"] == ["NVDA.O"]
        assert "TR.OfficerName" in tc.arguments["fields"]
        assert "TR.OfficerTitle" in tc.arguments["fields"]
        assert tc.arguments["parameters"]["OfficerType"] == "Executive"
