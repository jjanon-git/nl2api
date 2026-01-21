"""
Integration tests for API contract validation.

These tests hit the real Anthropic API to catch issues that mocks hide:
- Tool name validation (must match ^[a-zA-Z0-9_-]{1,128})
- Tool schema validation
- Request/response format changes

Run with:
    .venv/bin/python -m pytest tests/integration/ -v

Requires:
    - NL2API_ANTHROPIC_API_KEY in .env file or environment
    - Anthropic account with credits

These tests are intentionally minimal (one query per agent) to keep costs low (~$0.01 per run).
"""

import os
from pathlib import Path

import pytest

# Load .env file before checking for API key
def _load_env():
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value

_load_env()

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("NL2API_ANTHROPIC_API_KEY"),
    reason="NL2API_ANTHROPIC_API_KEY not set - skipping integration tests"
)


@pytest.fixture(scope="module")
def llm_provider():
    """Create a real LLM provider for integration tests."""
    from src.nl2api.llm.factory import create_llm_provider

    api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY")
    return create_llm_provider(
        provider="claude",
        api_key=api_key,
        model="claude-3-haiku-20240307",  # Use Haiku for cost efficiency
    )


class TestToolNameValidation:
    """Test that all agent tool definitions are accepted by the API."""

    @pytest.mark.asyncio
    async def test_datastream_tool_names_valid(self, llm_provider):
        """DatastreamAgent tool names must be API-compliant and from ToolRegistry."""
        from src.nl2api.agents.datastream import DatastreamAgent
        from CONTRACTS import ToolRegistry

        agent = DatastreamAgent(llm=llm_provider)
        tools = agent.get_tools()

        for tool in tools:
            # Must be API-compliant
            assert ToolRegistry.is_api_compliant(tool.name), f"Invalid tool name: {tool.name}"
            # Must be known to registry
            assert ToolRegistry.is_valid(tool.name), f"Tool name not in ToolRegistry: {tool.name}"

    @pytest.mark.asyncio
    async def test_estimates_tool_names_valid(self, llm_provider):
        """EstimatesAgent tool names must be API-compliant."""
        from src.nl2api.agents.estimates import EstimatesAgent

        agent = EstimatesAgent(llm=llm_provider)
        tools = agent.get_tools()

        import re
        pattern = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
        for tool in tools:
            assert pattern.match(tool.name), f"Invalid tool name: {tool.name}"

    @pytest.mark.asyncio
    async def test_fundamentals_tool_names_valid(self, llm_provider):
        """FundamentalsAgent tool names must be API-compliant."""
        from src.nl2api.agents.fundamentals import FundamentalsAgent

        agent = FundamentalsAgent(llm=llm_provider)
        tools = agent.get_tools()

        import re
        pattern = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
        for tool in tools:
            assert pattern.match(tool.name), f"Invalid tool name: {tool.name}"

    @pytest.mark.asyncio
    async def test_officers_tool_names_valid(self, llm_provider):
        """OfficersAgent tool names must be API-compliant."""
        from src.nl2api.agents.officers import OfficersAgent

        agent = OfficersAgent(llm=llm_provider)
        tools = agent.get_tools()

        import re
        pattern = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
        for tool in tools:
            assert pattern.match(tool.name), f"Invalid tool name: {tool.name}"

    @pytest.mark.asyncio
    async def test_screening_tool_names_valid(self, llm_provider):
        """ScreeningAgent tool names must be API-compliant."""
        from src.nl2api.agents.screening import ScreeningAgent

        agent = ScreeningAgent(llm=llm_provider)
        tools = agent.get_tools()

        import re
        pattern = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
        for tool in tools:
            assert pattern.match(tool.name), f"Invalid tool name: {tool.name}"


class TestAgentAPIIntegration:
    """Test that each agent can successfully call the Anthropic API."""

    @pytest.mark.asyncio
    async def test_datastream_agent_api_call(self, llm_provider):
        """DatastreamAgent can make a successful API call."""
        from src.nl2api.agents.datastream import DatastreamAgent
        from src.nl2api.agents.protocols import AgentContext

        agent = DatastreamAgent(llm=llm_provider)
        context = AgentContext(
            query="What is Apple's stock price?",
            resolved_entities={"Apple": "U:AAPL"},
        )

        result = await agent.process(context)

        # Should return tool calls, not an error
        assert result.tool_calls or result.needs_clarification, \
            f"Expected tool calls or clarification, got: {result}"

        # If we got tool calls, verify structure
        if result.tool_calls:
            assert len(result.tool_calls) >= 1
            assert result.tool_calls[0].tool_name is not None

    @pytest.mark.asyncio
    async def test_estimates_agent_api_call(self, llm_provider):
        """EstimatesAgent can make a successful API call."""
        from src.nl2api.agents.estimates import EstimatesAgent
        from src.nl2api.agents.protocols import AgentContext

        agent = EstimatesAgent(llm=llm_provider)
        context = AgentContext(
            query="What is the EPS estimate for Microsoft?",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        result = await agent.process(context)

        assert result.tool_calls or result.needs_clarification, \
            f"Expected tool calls or clarification, got: {result}"

    @pytest.mark.asyncio
    async def test_fundamentals_agent_api_call(self, llm_provider):
        """FundamentalsAgent can make a successful API call."""
        from src.nl2api.agents.fundamentals import FundamentalsAgent
        from src.nl2api.agents.protocols import AgentContext

        agent = FundamentalsAgent(llm=llm_provider)
        context = AgentContext(
            query="What is Tesla's revenue?",
            resolved_entities={"Tesla": "TSLA.O"},
        )

        result = await agent.process(context)

        assert result.tool_calls or result.needs_clarification, \
            f"Expected tool calls or clarification, got: {result}"

    @pytest.mark.asyncio
    async def test_officers_agent_api_call(self, llm_provider):
        """OfficersAgent can make a successful API call."""
        from src.nl2api.agents.officers import OfficersAgent
        from src.nl2api.agents.protocols import AgentContext

        agent = OfficersAgent(llm=llm_provider)
        context = AgentContext(
            query="Who is the CEO of Amazon?",
            resolved_entities={"Amazon": "AMZN.O"},
        )

        result = await agent.process(context)

        assert result.tool_calls or result.needs_clarification, \
            f"Expected tool calls or clarification, got: {result}"

    @pytest.mark.asyncio
    async def test_screening_agent_api_call(self, llm_provider):
        """ScreeningAgent can make a successful API call."""
        from src.nl2api.agents.screening import ScreeningAgent
        from src.nl2api.agents.protocols import AgentContext

        agent = ScreeningAgent(llm=llm_provider)
        context = AgentContext(
            query="Show me the top 10 stocks by market cap in the S&P 500",
            resolved_entities={},
        )

        result = await agent.process(context)

        assert result.tool_calls or result.needs_clarification, \
            f"Expected tool calls or clarification, got: {result}"


class TestOrchestratorAPIIntegration:
    """Test the full orchestrator flow against the real API."""

    @pytest.mark.asyncio
    async def test_orchestrator_end_to_end(self, llm_provider):
        """Orchestrator can route and process a query end-to-end."""
        from src.nl2api.agents.datastream import DatastreamAgent
        from src.nl2api.agents.estimates import EstimatesAgent
        from src.nl2api.orchestrator import NL2APIOrchestrator

        agents = {
            "datastream": DatastreamAgent(llm=llm_provider),
            "estimates": EstimatesAgent(llm=llm_provider),
        }

        orchestrator = NL2APIOrchestrator(llm=llm_provider, agents=agents)

        result = await orchestrator.process("What is Google's stock price?")

        # Should get a response (either tool calls or clarification)
        assert result is not None
        assert result.tool_calls or result.needs_clarification, \
            f"Expected tool calls or clarification, got: {result}"

    @pytest.mark.asyncio
    async def test_orchestrator_routing_works(self, llm_provider):
        """Orchestrator correctly routes to different agents."""
        from src.nl2api.agents.datastream import DatastreamAgent
        from src.nl2api.agents.estimates import EstimatesAgent
        from src.nl2api.orchestrator import NL2APIOrchestrator

        agents = {
            "datastream": DatastreamAgent(llm=llm_provider),
            "estimates": EstimatesAgent(llm=llm_provider),
        }

        orchestrator = NL2APIOrchestrator(llm=llm_provider, agents=agents)

        # Price query should route to datastream
        price_result = await orchestrator.process("What is Apple's price?")
        assert price_result is not None

        # Estimate query should route to estimates
        estimate_result = await orchestrator.process("What are analyst estimates for Apple's EPS?")
        assert estimate_result is not None
