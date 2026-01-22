"""
Unit Tests for NL2API MCP Tools

Tests the NL2API orchestrator and domain agent tools:
- Tool definitions
- Tool handlers
- Placeholder response generation
- Clarification support
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_servers.entity_resolution.nl2api_tools import (
    NL2API_TOOL_DEFINITIONS,
    NL2APIToolHandlers,
)


# =============================================================================
# Tool Definition Tests
# =============================================================================


class TestNL2APIToolDefinitions:
    """Tests for NL2API MCP tool definitions."""

    def test_tool_definitions_exist(self) -> None:
        """Test that all NL2API tool definitions are present."""
        tool_names = {t["name"] for t in NL2API_TOOL_DEFINITIONS}

        assert "nl2api_query" in tool_names
        assert "query_datastream" in tool_names
        assert "query_estimates" in tool_names
        assert "query_fundamentals" in tool_names
        assert "query_officers" in tool_names
        assert "query_screening" in tool_names
        assert len(tool_names) == 6

    def test_nl2api_query_definition(self) -> None:
        """Test nl2api_query tool definition."""
        tool = next(t for t in NL2API_TOOL_DEFINITIONS if t["name"] == "nl2api_query")

        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"
        assert "query" in tool["inputSchema"]["properties"]
        assert "query" in tool["inputSchema"]["required"]

    def test_query_datastream_definition(self) -> None:
        """Test query_datastream tool definition."""
        tool = next(t for t in NL2API_TOOL_DEFINITIONS if t["name"] == "query_datastream")

        assert "description" in tool
        assert "inputSchema" in tool
        assert "query" in tool["inputSchema"]["properties"]
        assert "rics" in tool["inputSchema"]["properties"]
        assert tool["inputSchema"]["properties"]["rics"]["type"] == "array"
        assert "query" in tool["inputSchema"]["required"]

    def test_query_estimates_definition(self) -> None:
        """Test query_estimates tool definition."""
        tool = next(t for t in NL2API_TOOL_DEFINITIONS if t["name"] == "query_estimates")

        assert "description" in tool
        assert "query" in tool["inputSchema"]["properties"]
        assert "rics" in tool["inputSchema"]["properties"]

    def test_query_fundamentals_definition(self) -> None:
        """Test query_fundamentals tool definition."""
        tool = next(t for t in NL2API_TOOL_DEFINITIONS if t["name"] == "query_fundamentals")

        assert "description" in tool
        assert "query" in tool["inputSchema"]["properties"]

    def test_query_officers_definition(self) -> None:
        """Test query_officers tool definition."""
        tool = next(t for t in NL2API_TOOL_DEFINITIONS if t["name"] == "query_officers")

        assert "description" in tool
        assert "query" in tool["inputSchema"]["properties"]

    def test_query_screening_definition(self) -> None:
        """Test query_screening tool definition."""
        tool = next(t for t in NL2API_TOOL_DEFINITIONS if t["name"] == "query_screening")

        assert "description" in tool
        assert "query" in tool["inputSchema"]["properties"]
        # Screening doesn't need rics parameter
        assert "rics" not in tool["inputSchema"]["properties"]


# =============================================================================
# Handler Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Create a mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.process = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_agents() -> dict[str, MagicMock]:
    """Create mock domain agents."""
    agents = {}
    for domain in ["datastream", "estimates", "fundamentals", "officers", "screening"]:
        agent = MagicMock()
        agent.process = AsyncMock()
        agents[domain] = agent
    return agents


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def mock_resolver() -> MagicMock:
    """Create a mock entity resolver."""
    resolver = MagicMock()
    resolver.resolve = AsyncMock()
    return resolver


@pytest.fixture
def handlers(
    mock_orchestrator: MagicMock,
    mock_agents: dict[str, MagicMock],
    mock_llm: MagicMock,
    mock_resolver: MagicMock,
) -> NL2APIToolHandlers:
    """Create tool handlers with mock dependencies."""
    return NL2APIToolHandlers(
        orchestrator=mock_orchestrator,
        agents=mock_agents,
        llm=mock_llm,
        resolver=mock_resolver,
    )


# =============================================================================
# nl2api_query Handler Tests
# =============================================================================


class TestNL2APIQueryHandler:
    """Tests for nl2api_query tool handler."""

    @pytest.mark.asyncio
    async def test_nl2api_query_success(
        self, handlers: NL2APIToolHandlers, mock_orchestrator: MagicMock, mock_llm: MagicMock
    ) -> None:
        """Test successful nl2api_query processing."""
        # Create mock ToolCall
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "refinitiv_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O"], "fields": ["TR.PERatio"]}

        # Create mock OrchestratorResult
        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.resolved_entities = {"Apple": "AAPL.O"}
        mock_result.domain = "fundamentals"
        mock_result.confidence = 0.92
        mock_result.reasoning = "Query asks for P/E ratio"
        mock_result.tool_calls = (mock_tool_call,)

        mock_orchestrator.process.return_value = mock_result

        # Mock LLM response for placeholder generation
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"AAPL.O": {"TR.PERatio": 28.5}}, "nl_response": "Apple has a P/E ratio of 28.5."}'
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "nl2api_query",
            {"query": "What is Apple's P/E ratio?"},
        )

        # Now returns just the answer string
        assert isinstance(result, str)
        assert "28.5" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_nl2api_query_missing_query(self, handlers: NL2APIToolHandlers) -> None:
        """Test nl2api_query with missing query parameter."""
        result = await handlers.handle_tool_call(
            "nl2api_query",
            {},
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Query is required"

    @pytest.mark.asyncio
    async def test_nl2api_query_empty_query(self, handlers: NL2APIToolHandlers) -> None:
        """Test nl2api_query with empty query string."""
        result = await handlers.handle_tool_call(
            "nl2api_query",
            {"query": ""},
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_nl2api_query_needs_clarification(
        self, handlers: NL2APIToolHandlers, mock_orchestrator: MagicMock
    ) -> None:
        """Test nl2api_query when clarification is needed."""
        # Create mock clarification question
        mock_question = MagicMock()
        mock_question.question = "Which Apple did you mean? The company or the fruit?"

        # Create mock OrchestratorResult with clarification
        mock_result = MagicMock()
        mock_result.needs_clarification = True
        mock_result.clarification_questions = [mock_question]

        mock_orchestrator.process.return_value = mock_result

        result = await handlers.handle_tool_call(
            "nl2api_query",
            {"query": "What is Apple's price?"},
        )

        assert result["success"] is True
        assert result["needs_clarification"] is True
        assert len(result["clarification_questions"]) == 1
        assert "processing_time_ms" in result

    @pytest.mark.asyncio
    async def test_nl2api_query_orchestrator_error(
        self, handlers: NL2APIToolHandlers, mock_orchestrator: MagicMock
    ) -> None:
        """Test nl2api_query when orchestrator raises an exception."""
        mock_orchestrator.process.side_effect = Exception("Internal error")

        result = await handlers.handle_tool_call(
            "nl2api_query",
            {"query": "What is Apple's P/E ratio?"},
        )

        assert result["success"] is False
        assert "error" in result
        assert "Internal error" in result["error"]
        assert "processing_time_ms" in result

    @pytest.mark.asyncio
    async def test_nl2api_query_no_tool_calls(
        self, handlers: NL2APIToolHandlers, mock_orchestrator: MagicMock
    ) -> None:
        """Test nl2api_query when no tool calls are generated."""
        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.resolved_entities = {}
        mock_result.domain = None
        mock_result.confidence = 0.5
        mock_result.reasoning = "Unable to determine domain"
        mock_result.tool_calls = ()

        mock_orchestrator.process.return_value = mock_result

        result = await handlers.handle_tool_call(
            "nl2api_query",
            {"query": "Hello, how are you?"},
        )

        assert isinstance(result, str)
        assert "No data available" in result


# =============================================================================
# Domain Agent Handler Tests
# =============================================================================


class TestDomainAgentHandlers:
    """Tests for domain agent tool handlers."""

    @pytest.mark.asyncio
    async def test_query_datastream_success(
        self,
        handlers: NL2APIToolHandlers,
        mock_agents: dict[str, MagicMock],
        mock_llm: MagicMock,
        mock_resolver: MagicMock,
    ) -> None:
        """Test successful query_datastream processing."""
        # Mock resolver.resolve() to return extracted + resolved entities
        mock_resolver.resolve.return_value = {"Apple": "AAPL.O"}

        # Create mock ToolCall
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "datastream_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O"], "fields": ["P"]}

        # Create mock AgentResult
        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.tool_calls = (mock_tool_call,)
        mock_result.confidence = 0.89
        mock_result.reasoning = "Query requests price"

        mock_agents["datastream"].process.return_value = mock_result

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"AAPL.O": {"P": 178.50}}, "nl_response": "Apple price is $178.50."}'
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "query_datastream",
            {"query": "Get Apple's price"},
        )

        assert isinstance(result, str)
        assert "178.50" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_query_datastream_with_preresolved_rics(
        self,
        handlers: NL2APIToolHandlers,
        mock_agents: dict[str, MagicMock],
        mock_llm: MagicMock,
    ) -> None:
        """Test query_datastream with pre-resolved RICs."""
        # Create mock ToolCall
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "datastream_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O", "MSFT.O"], "fields": ["P"]}

        # Create mock AgentResult
        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.tool_calls = (mock_tool_call,)
        mock_result.confidence = 0.95
        mock_result.reasoning = "Query requests prices"

        mock_agents["datastream"].process.return_value = mock_result

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"AAPL.O": {"P": 178.50}, "MSFT.O": {"P": 380.00}}, "nl_response": "Prices retrieved."}'
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "query_datastream",
            {"query": "Get prices", "rics": ["AAPL.O", "MSFT.O"]},
        )

        assert isinstance(result, str)
        assert "Prices" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_query_estimates_success(
        self,
        handlers: NL2APIToolHandlers,
        mock_agents: dict[str, MagicMock],
        mock_llm: MagicMock,
    ) -> None:
        """Test successful query_estimates processing."""
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "estimates_get_data"
        mock_tool_call.arguments = {"RICs": ["TSLA.O"], "fields": ["TR.EPSMean"]}

        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.tool_calls = (mock_tool_call,)
        mock_result.confidence = 0.88
        mock_result.reasoning = "Query requests EPS estimates"

        mock_agents["estimates"].process.return_value = mock_result

        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"TSLA.O": {"TR.EPSMean": 4.25}}, "nl_response": "Tesla FY1 EPS is $4.25."}'
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "query_estimates",
            {"query": "Tesla FY1 EPS forecast", "rics": ["TSLA.O"]},
        )

        assert isinstance(result, str)
        assert "4.25" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_query_fundamentals_success(
        self,
        handlers: NL2APIToolHandlers,
        mock_agents: dict[str, MagicMock],
        mock_llm: MagicMock,
    ) -> None:
        """Test successful query_fundamentals processing."""
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "fundamentals_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O"], "fields": ["TR.PERatio"]}

        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.tool_calls = (mock_tool_call,)
        mock_result.confidence = 0.92
        mock_result.reasoning = "Query requests P/E ratio"

        mock_agents["fundamentals"].process.return_value = mock_result

        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"AAPL.O": {"TR.PERatio": 28.5}}, "nl_response": "Apple P/E is 28.5x."}'
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "query_fundamentals",
            {"query": "Apple P/E ratio", "rics": ["AAPL.O"]},
        )

        assert isinstance(result, str)
        assert "28.5" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_query_officers_success(
        self,
        handlers: NL2APIToolHandlers,
        mock_agents: dict[str, MagicMock],
        mock_llm: MagicMock,
    ) -> None:
        """Test successful query_officers processing."""
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "officers_get_data"
        mock_tool_call.arguments = {"RICs": ["AMZN.O"]}

        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.tool_calls = (mock_tool_call,)
        mock_result.confidence = 0.90
        mock_result.reasoning = "Query requests CEO info"

        mock_agents["officers"].process.return_value = mock_result

        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"AMZN.O": {"CEO": "Andy Jassy"}}, "nl_response": "Amazon CEO is Andy Jassy."}'
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "query_officers",
            {"query": "Who is Amazon's CEO?", "rics": ["AMZN.O"]},
        )

        assert isinstance(result, str)
        assert "Andy Jassy" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_query_screening_success(
        self,
        handlers: NL2APIToolHandlers,
        mock_agents: dict[str, MagicMock],
        mock_llm: MagicMock,
        mock_resolver: MagicMock,
    ) -> None:
        """Test successful query_screening processing."""
        # Screening doesn't typically need entity resolution, but the handler
        # still calls resolve() when no RICs are provided
        mock_resolver.resolve.return_value = {}  # No entities for screening queries

        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "screen"
        mock_tool_call.arguments = {"expression": "SCREEN(U(IN#(.SPX)), MV, TOP#(10))"}

        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.tool_calls = (mock_tool_call,)
        mock_result.confidence = 0.85
        mock_result.reasoning = "Query requests top 10 by market cap"

        mock_agents["screening"].process.return_value = mock_result

        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"results": ["AAPL.O", "MSFT.O", "GOOGL.O"]}, "nl_response": "Top 3 are Apple, Microsoft, Google."}'
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "query_screening",
            {"query": "Top 10 S&P 500 by market cap"},
        )

        assert isinstance(result, str)
        assert "Apple" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_agent_query_missing_query(self, handlers: NL2APIToolHandlers) -> None:
        """Test domain agent query with missing query parameter."""
        result = await handlers.handle_tool_call(
            "query_datastream",
            {},
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Query is required"

    @pytest.mark.asyncio
    async def test_agent_query_needs_clarification(
        self, handlers: NL2APIToolHandlers, mock_agents: dict[str, MagicMock]
    ) -> None:
        """Test domain agent query when clarification is needed."""
        mock_result = MagicMock()
        mock_result.needs_clarification = True
        mock_result.clarification_questions = ["Which time period?"]

        mock_agents["datastream"].process.return_value = mock_result

        result = await handlers.handle_tool_call(
            "query_datastream",
            {"query": "Get historical prices", "rics": ["AAPL.O"]},
        )

        assert result["success"] is True
        assert result["needs_clarification"] is True
        assert len(result["clarification_questions"]) == 1


# =============================================================================
# Unknown Tool Tests
# =============================================================================


class TestUnknownTool:
    """Tests for handling unknown tools."""

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self, handlers: NL2APIToolHandlers) -> None:
        """Test handling unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await handlers.handle_tool_call("unknown_tool", {})


# =============================================================================
# Placeholder Response Generation Tests
# =============================================================================


class TestPlaceholderResponseGeneration:
    """Tests for placeholder response generation with Haiku."""

    @pytest.mark.asyncio
    async def test_placeholder_response_success(
        self, handlers: NL2APIToolHandlers, mock_llm: MagicMock
    ) -> None:
        """Test successful placeholder response generation."""
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "refinitiv_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O"], "fields": ["TR.PERatio"]}

        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"execution_data": {"AAPL.O": {"TR.PERatio": 28.5}}, "nl_response": "Apple P/E is 28.5x."}'
        mock_llm.complete.return_value = mock_llm_response

        data, nl_response = await handlers._generate_placeholder_response(
            query="What is Apple's P/E ratio?",
            tool_calls=(mock_tool_call,),
            domain="fundamentals",
        )

        assert "AAPL.O" in data
        assert data["AAPL.O"]["TR.PERatio"] == 28.5
        assert "28.5" in nl_response

    @pytest.mark.asyncio
    async def test_placeholder_response_markdown_handling(
        self, handlers: NL2APIToolHandlers, mock_llm: MagicMock
    ) -> None:
        """Test placeholder response strips markdown code blocks."""
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "refinitiv_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O"], "fields": ["P"]}

        mock_llm_response = MagicMock()
        mock_llm_response.content = '```json\n{"execution_data": {"AAPL.O": {"P": 178.50}}, "nl_response": "Apple is at $178.50."}\n```'
        mock_llm.complete.return_value = mock_llm_response

        data, nl_response = await handlers._generate_placeholder_response(
            query="What is Apple's price?",
            tool_calls=(mock_tool_call,),
            domain="datastream",
        )

        assert "AAPL.O" in data
        assert "$178.50" in nl_response

    @pytest.mark.asyncio
    async def test_placeholder_response_llm_error(
        self, handlers: NL2APIToolHandlers, mock_llm: MagicMock
    ) -> None:
        """Test placeholder response handles LLM errors gracefully."""
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "refinitiv_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O"], "fields": ["P"]}

        mock_llm.complete.side_effect = Exception("API error")

        data, nl_response = await handlers._generate_placeholder_response(
            query="What is Apple's price?",
            tool_calls=(mock_tool_call,),
            domain="datastream",
        )

        assert "error" in data
        assert "Unable to generate" in nl_response

    @pytest.mark.asyncio
    async def test_placeholder_response_invalid_json(
        self, handlers: NL2APIToolHandlers, mock_llm: MagicMock
    ) -> None:
        """Test placeholder response handles invalid JSON gracefully."""
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "refinitiv_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O"], "fields": ["P"]}

        mock_llm_response = MagicMock()
        mock_llm_response.content = "This is not valid JSON"
        mock_llm.complete.return_value = mock_llm_response

        data, nl_response = await handlers._generate_placeholder_response(
            query="What is Apple's price?",
            tool_calls=(mock_tool_call,),
            domain="datastream",
        )

        assert "error" in data
        assert "Unable to generate" in nl_response


# =============================================================================
# Integration-Like Tests (Mocked Dependencies)
# =============================================================================


class TestHandlerIntegration:
    """Integration-like tests with full handler flow."""

    @pytest.mark.asyncio
    async def test_full_nl2api_query_flow(
        self, handlers: NL2APIToolHandlers, mock_orchestrator: MagicMock, mock_llm: MagicMock
    ) -> None:
        """Test complete nl2api_query flow from input to response."""
        # Set up complete mock chain
        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "refinitiv_get_data"
        mock_tool_call.arguments = {"RICs": ["AAPL.O", "MSFT.O"], "fields": ["TR.PERatio"]}

        mock_result = MagicMock()
        mock_result.needs_clarification = False
        mock_result.resolved_entities = {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}
        mock_result.domain = "fundamentals"
        mock_result.confidence = 0.95
        mock_result.reasoning = "Comparing P/E ratios of two companies"
        mock_result.tool_calls = (mock_tool_call,)

        mock_orchestrator.process.return_value = mock_result

        mock_llm_response = MagicMock()
        mock_llm_response.content = """{"execution_data": {"AAPL.O": {"TR.PERatio": 28.5}, "MSFT.O": {"TR.PERatio": 35.2}}, "nl_response": "Apple has a P/E of 28.5x while Microsoft has 35.2x."}"""
        mock_llm.complete.return_value = mock_llm_response

        result = await handlers.handle_tool_call(
            "nl2api_query",
            {"query": "Compare Apple and Microsoft P/E ratios"},
        )

        # Verify response is just the answer string
        assert isinstance(result, str)
        assert "Apple" in result
        assert "28.5" in result or "35.2" in result  # From the mock LLM response

    @pytest.mark.asyncio
    async def test_all_domain_agents_accessible(
        self,
        handlers: NL2APIToolHandlers,
        mock_agents: dict[str, MagicMock],
        mock_llm: MagicMock,
        mock_resolver: MagicMock,
    ) -> None:
        """Test that all domain agents can be accessed via their tools."""
        domains = ["datastream", "estimates", "fundamentals", "officers", "screening"]

        # Mock resolver for when no RICs are provided (screening case)
        mock_resolver.resolve.return_value = {}

        for domain in domains:
            mock_tool_call = MagicMock()
            mock_tool_call.tool_name = f"{domain}_get_data"
            mock_tool_call.arguments = {"RICs": ["TEST.O"]}

            mock_result = MagicMock()
            mock_result.needs_clarification = False
            mock_result.tool_calls = (mock_tool_call,)
            mock_result.confidence = 0.90
            mock_result.reasoning = f"Test {domain}"

            mock_agents[domain].process.return_value = mock_result

            mock_llm_response = MagicMock()
            mock_llm_response.content = '{"execution_data": {"TEST.O": {}}, "nl_response": "Test response."}'
            mock_llm.complete.return_value = mock_llm_response

            result = await handlers.handle_tool_call(
                f"query_{domain}",
                {"query": f"Test {domain} query", "rics": ["TEST.O"]}
                if domain != "screening"
                else {"query": f"Test {domain} query"},
            )

            assert isinstance(result, str), f"Failed for domain: {domain}"
            assert "Test response" in result  # From the mock LLM response
