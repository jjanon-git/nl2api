"""
Tests for Tool Providers

Tests the AgentToolProvider, MCPToolProvider, and dual-mode factory functions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from src.nl2api.llm.protocols import LLMToolDefinition
from src.nl2api.mcp.client import MCPClient, MCPClientConfig
from src.nl2api.mcp.protocols import MCPServer, MCPToolDefinition as MCPToolDef, MCPToolParameter
from src.nl2api.routing.providers import (
    AgentToolProvider,
    AgentToolExecutor,
    MCPToolProvider,
    MCPToolExecutor,
    create_providers_from_agents,
    create_dual_mode_providers,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockDomainAgent:
    """Mock domain agent for testing."""

    def __init__(
        self,
        name: str = "test_agent",
        description: str = "Test agent description",
        capabilities: tuple[str, ...] = ("cap1", "cap2"),
        example_queries: tuple[str, ...] = ("query1", "query2"),
        tools: list[LLMToolDefinition] | None = None,
    ):
        self._domain_name = name
        self._domain_description = description
        self._capabilities = capabilities
        self._example_queries = example_queries
        self._tools = tools or [
            LLMToolDefinition(
                name=f"{name}.get_data",
                description=f"Get data from {name}",
                parameters={"type": "object", "properties": {}},
            )
        ]

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def domain_description(self) -> str:
        return self._domain_description

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._capabilities

    @property
    def example_queries(self) -> tuple[str, ...]:
        return self._example_queries

    def get_tools(self) -> list[LLMToolDefinition]:
        return self._tools


class MockAgentWithoutGetTools:
    """Mock agent without get_tools method."""

    @property
    def domain_name(self) -> str:
        return "no_tools_agent"

    @property
    def domain_description(self) -> str:
        return "Agent without get_tools"


class MockAgentWithoutCapabilities:
    """Mock agent without capabilities property."""

    @property
    def domain_name(self) -> str:
        return "no_caps_agent"

    @property
    def domain_description(self) -> str:
        return "Agent without capabilities"

    def get_tools(self) -> list[LLMToolDefinition]:
        return []


# =============================================================================
# AgentToolProvider Tests
# =============================================================================


class TestAgentToolProvider:
    """Tests for AgentToolProvider."""

    def test_provider_name_from_agent(self):
        """Test that provider_name comes from agent."""
        agent = MockDomainAgent(name="datastream")
        provider = AgentToolProvider(agent=agent)

        assert provider.provider_name == "datastream"

    def test_provider_description_from_agent(self):
        """Test that provider_description comes from agent."""
        agent = MockDomainAgent(description="Stock price data")
        provider = AgentToolProvider(agent=agent)

        assert provider.provider_description == "Stock price data"

    def test_capabilities_from_agent(self):
        """Test that capabilities come from agent."""
        agent = MockDomainAgent(capabilities=("prices", "volumes"))
        provider = AgentToolProvider(agent=agent)

        assert provider.capabilities == ("prices", "volumes")

    def test_capabilities_override(self):
        """Test capabilities override."""
        agent = MockDomainAgent(capabilities=("original",))
        provider = AgentToolProvider(
            agent=agent,
            capabilities=("overridden", "caps"),
        )

        assert provider.capabilities == ("overridden", "caps")

    def test_capabilities_fallback_for_agent_without_property(self):
        """Test capabilities fallback when agent lacks property."""
        agent = MockAgentWithoutCapabilities()
        provider = AgentToolProvider(agent=agent)

        assert provider.capabilities == ()

    def test_example_queries_from_agent(self):
        """Test that example_queries come from agent."""
        agent = MockDomainAgent(example_queries=("What is X?", "Show Y"))
        provider = AgentToolProvider(agent=agent)

        assert provider.example_queries == ("What is X?", "Show Y")

    def test_example_queries_override(self):
        """Test example_queries override."""
        agent = MockDomainAgent(example_queries=("original",))
        provider = AgentToolProvider(
            agent=agent,
            example_queries=("new query 1", "new query 2"),
        )

        assert provider.example_queries == ("new query 1", "new query 2")

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools from agent."""
        tools = [
            LLMToolDefinition(name="tool1", description="desc1", parameters={}),
            LLMToolDefinition(name="tool2", description="desc2", parameters={}),
        ]
        agent = MockDomainAgent(tools=tools)
        provider = AgentToolProvider(agent=agent)

        result = await provider.list_tools()

        assert len(result) == 2
        assert result[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_list_tools_returns_empty_for_agent_without_method(self):
        """Test that list_tools returns empty for agents without get_tools."""
        agent = MockAgentWithoutGetTools()
        provider = AgentToolProvider(agent=agent)

        result = await provider.list_tools()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_tool_description(self):
        """Test getting tool description."""
        tools = [
            LLMToolDefinition(name="my_tool", description="My description", parameters={}),
        ]
        agent = MockDomainAgent(tools=tools)
        provider = AgentToolProvider(agent=agent)

        desc = await provider.get_tool_description("my_tool")

        assert desc == "My description"

    @pytest.mark.asyncio
    async def test_get_tool_description_not_found(self):
        """Test getting description for nonexistent tool."""
        agent = MockDomainAgent()
        provider = AgentToolProvider(agent=agent)

        desc = await provider.get_tool_description("nonexistent")

        assert desc is None

    def test_agent_property(self):
        """Test accessing the wrapped agent."""
        agent = MockDomainAgent()
        provider = AgentToolProvider(agent=agent)

        assert provider.agent is agent


# =============================================================================
# AgentToolExecutor Tests
# =============================================================================


class TestAgentToolExecutor:
    """Tests for AgentToolExecutor."""

    def test_builds_tool_mapping(self):
        """Test that tool mapping is built on initialization."""
        agent1 = MockDomainAgent(
            name="agent1",
            tools=[LLMToolDefinition(name="agent1.tool", description="", parameters={})],
        )
        agent2 = MockDomainAgent(
            name="agent2",
            tools=[LLMToolDefinition(name="agent2.tool", description="", parameters={})],
        )

        providers = [
            AgentToolProvider(agent=agent1),
            AgentToolProvider(agent=agent2),
        ]
        executor = AgentToolExecutor(providers=providers)

        assert "agent1.tool" in executor._tool_to_provider
        assert "agent2.tool" in executor._tool_to_provider

    @pytest.mark.asyncio
    async def test_execute_returns_pending_status(self):
        """Test that execute returns pending execution status."""
        agent = MockDomainAgent(
            name="test",
            tools=[LLMToolDefinition(name="test.get", description="", parameters={})],
        )
        provider = AgentToolProvider(agent=agent)
        executor = AgentToolExecutor(providers=[provider])

        result = await executor.execute("test.get", {"arg": "value"})

        assert result["tool_name"] == "test.get"
        assert result["arguments"] == {"arg": "value"}
        assert result["provider"] == "test"
        assert result["status"] == "pending_execution"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self):
        """Test that executing unknown tool raises ValueError."""
        executor = AgentToolExecutor(providers=[])

        with pytest.raises(ValueError, match="Unknown tool"):
            await executor.execute("nonexistent.tool", {})


# =============================================================================
# MCPToolProvider Tests
# =============================================================================


class TestMCPToolProvider:
    """Tests for MCPToolProvider."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = MagicMock(spec=MCPClient)
        client.is_connected.return_value = True
        client.list_tools = AsyncMock(return_value=[])
        return client

    def test_provider_name_from_override(self):
        """Test provider_name from explicit override."""
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            provider_name="custom_name",
        )

        assert provider.provider_name == "custom_name"

    def test_provider_name_extracted_from_uri(self):
        """Test provider_name extracted from URI."""
        provider = MCPToolProvider(server_uri="mcp://datastream.lseg.com")

        assert provider.provider_name == "datastream"

    def test_provider_description_from_override(self):
        """Test provider_description from explicit override."""
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            provider_description="Custom description",
        )

        assert provider.provider_description == "Custom description"

    def test_provider_description_default(self):
        """Test default provider_description."""
        provider = MCPToolProvider(server_uri="mcp://test.com")

        assert "mcp://test.com" in provider.provider_description

    def test_capabilities(self):
        """Test capabilities property."""
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            capabilities=("prices", "volumes"),
        )

        assert provider.capabilities == ("prices", "volumes")

    def test_example_queries(self):
        """Test example_queries property."""
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            example_queries=("query 1", "query 2"),
        )

        assert provider.example_queries == ("query 1", "query 2")

    def test_server_uri(self):
        """Test server_uri property."""
        provider = MCPToolProvider(server_uri="mcp://test.com")

        assert provider.server_uri == "mcp://test.com"

    def test_is_connected_without_client(self):
        """Test is_connected returns False without client."""
        provider = MCPToolProvider(server_uri="mcp://test.com")

        assert provider.is_connected is False

    def test_is_connected_with_client(self, mock_mcp_client):
        """Test is_connected with connected client."""
        mock_mcp_client.is_connected.return_value = True
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        assert provider.is_connected is True
        mock_mcp_client.is_connected.assert_called_with("mcp://test.com")

    @pytest.mark.asyncio
    async def test_list_tools_without_client(self):
        """Test list_tools returns empty without client."""
        provider = MCPToolProvider(server_uri="mcp://test.com")

        result = await provider.list_tools()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_tools_when_not_connected(self, mock_mcp_client):
        """Test list_tools returns empty when not connected."""
        mock_mcp_client.is_connected.return_value = False
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        result = await provider.list_tools()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_tools_fetches_from_server(self, mock_mcp_client):
        """Test list_tools fetches from MCP server."""
        mcp_tools = [
            MCPToolDef(
                name="get_price",
                description="Get price",
                parameters=(
                    MCPToolParameter(name="ric", description="Code", type="string"),
                ),
            ),
        ]
        mock_mcp_client.list_tools = AsyncMock(return_value=mcp_tools)

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        result = await provider.list_tools()

        assert len(result) == 1
        assert result[0].name == "get_price"
        mock_mcp_client.list_tools.assert_called_once_with("mcp://test.com")

    @pytest.mark.asyncio
    async def test_list_tools_caches_result(self, mock_mcp_client):
        """Test that tools are cached after first fetch."""
        mock_mcp_client.list_tools = AsyncMock(return_value=[])

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        # First call
        await provider.list_tools()
        # Second call
        await provider.list_tools()

        # Should only call server once
        assert mock_mcp_client.list_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_list_tools_handles_error(self, mock_mcp_client):
        """Test list_tools handles server errors gracefully."""
        mock_mcp_client.list_tools = AsyncMock(side_effect=Exception("Server error"))

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        result = await provider.list_tools()

        assert result == []

    def test_invalidate_cache(self, mock_mcp_client):
        """Test cache invalidation."""
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )
        provider._cached_tools = [LLMToolDefinition(name="cached", description="", parameters={})]

        provider.invalidate_cache()

        assert provider._cached_tools is None

    @pytest.mark.asyncio
    async def test_refresh_tools(self, mock_mcp_client):
        """Test refresh_tools invalidates cache and fetches again."""
        mock_mcp_client.list_tools = AsyncMock(return_value=[])
        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        # Populate cache
        await provider.list_tools()
        assert mock_mcp_client.list_tools.call_count == 1

        # Refresh
        await provider.refresh_tools()

        assert mock_mcp_client.list_tools.call_count == 2

    @pytest.mark.asyncio
    async def test_get_tool_description(self, mock_mcp_client):
        """Test getting tool description."""
        mcp_tools = [
            MCPToolDef(name="my_tool", description="My tool description"),
        ]
        mock_mcp_client.list_tools = AsyncMock(return_value=mcp_tools)

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        desc = await provider.get_tool_description("my_tool")

        assert desc == "My tool description"

    @pytest.mark.asyncio
    async def test_get_tool_description_not_found(self, mock_mcp_client):
        """Test get_tool_description for nonexistent tool."""
        mock_mcp_client.list_tools = AsyncMock(return_value=[])

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )

        desc = await provider.get_tool_description("nonexistent")

        assert desc is None


# =============================================================================
# MCPToolExecutor Tests
# =============================================================================


class TestMCPToolExecutor:
    """Tests for MCPToolExecutor."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = MagicMock(spec=MCPClient)
        client.is_connected.return_value = True
        client.list_tools = AsyncMock(return_value=[])
        return client

    @pytest.mark.asyncio
    async def test_execute_without_client_raises(self):
        """Test execute raises without MCP client."""
        executor = MCPToolExecutor(mcp_client=None)

        with pytest.raises(RuntimeError, match="MCP client not configured"):
            await executor.execute("tool", {})

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self, mock_mcp_client):
        """Test execute raises for unknown tool."""
        executor = MCPToolExecutor(mcp_client=mock_mcp_client, providers=[])

        with pytest.raises(ValueError, match="Unknown tool"):
            await executor.execute("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_build_tool_mapping(self, mock_mcp_client):
        """Test building tool mapping from providers."""
        mcp_tools = [MCPToolDef(name="test_tool", description="Test")]
        mock_mcp_client.list_tools = AsyncMock(return_value=mcp_tools)

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )
        executor = MCPToolExecutor(
            mcp_client=mock_mcp_client,
            providers=[provider],
        )

        await executor.build_tool_mapping()

        assert "test_tool" in executor._tool_to_server
        assert executor._tool_to_server["test_tool"] == "mcp://test.com"

    def test_add_provider(self, mock_mcp_client):
        """Test adding provider to executor."""
        executor = MCPToolExecutor(mcp_client=mock_mcp_client)

        provider = MCPToolProvider(
            server_uri="mcp://new.com",
            mcp_client=mock_mcp_client,
        )
        executor.add_provider(provider)

        assert provider in executor._providers

    @pytest.mark.asyncio
    async def test_execute_calls_mcp_client(self, mock_mcp_client):
        """Test execute calls MCP client."""
        from src.nl2api.mcp.protocols import MCPToolResult

        mock_mcp_client.call_tool = AsyncMock(
            return_value=MCPToolResult(
                tool_name="test_tool",
                content={"result": "success"},
                execution_time_ms=50,
            )
        )

        mcp_tools = [MCPToolDef(name="test_tool", description="Test")]
        mock_mcp_client.list_tools = AsyncMock(return_value=mcp_tools)

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )
        executor = MCPToolExecutor(
            mcp_client=mock_mcp_client,
            providers=[provider],
        )
        await executor.build_tool_mapping()

        result = await executor.execute("test_tool", {"arg": "value"})

        assert result["tool_name"] == "test_tool"
        assert result["content"] == {"result": "success"}
        assert result["is_error"] is False
        mock_mcp_client.call_tool.assert_called_once_with(
            server_uri="mcp://test.com",
            tool_name="test_tool",
            arguments={"arg": "value"},
        )

    @pytest.mark.asyncio
    async def test_execute_handles_error_result(self, mock_mcp_client):
        """Test execute handles error result from MCP."""
        from src.nl2api.mcp.protocols import MCPToolResult

        mock_mcp_client.call_tool = AsyncMock(
            return_value=MCPToolResult(
                tool_name="test_tool",
                content=None,
                is_error=True,
                error_message="Tool failed",
            )
        )

        mcp_tools = [MCPToolDef(name="test_tool", description="Test")]
        mock_mcp_client.list_tools = AsyncMock(return_value=mcp_tools)

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )
        executor = MCPToolExecutor(
            mcp_client=mock_mcp_client,
            providers=[provider],
        )
        await executor.build_tool_mapping()

        result = await executor.execute("test_tool", {})

        assert result["is_error"] is True
        assert result["error"] == "Tool failed"

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self, mock_mcp_client):
        """Test execute handles exceptions from MCP client."""
        mock_mcp_client.call_tool = AsyncMock(side_effect=Exception("Network error"))

        mcp_tools = [MCPToolDef(name="test_tool", description="Test")]
        mock_mcp_client.list_tools = AsyncMock(return_value=mcp_tools)

        provider = MCPToolProvider(
            server_uri="mcp://test.com",
            mcp_client=mock_mcp_client,
        )
        executor = MCPToolExecutor(
            mcp_client=mock_mcp_client,
            providers=[provider],
        )
        await executor.build_tool_mapping()

        result = await executor.execute("test_tool", {})

        assert result["is_error"] is True
        assert "Network error" in result["error"]


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateProvidersFromAgents:
    """Tests for create_providers_from_agents factory."""

    def test_creates_providers_for_each_agent(self):
        """Test that providers are created for each agent."""
        agents = {
            "datastream": MockDomainAgent(name="datastream"),
            "estimates": MockDomainAgent(name="estimates"),
        }

        providers = create_providers_from_agents(agents)

        assert len(providers) == 2
        names = {p.provider_name for p in providers}
        assert names == {"datastream", "estimates"}

    def test_applies_capabilities_map(self):
        """Test that capabilities map is applied."""
        agents = {"test": MockDomainAgent(name="test", capabilities=("original",))}
        caps_map = {"test": ("custom1", "custom2")}

        providers = create_providers_from_agents(agents, capabilities_map=caps_map)

        assert providers[0].capabilities == ("custom1", "custom2")

    def test_applies_examples_map(self):
        """Test that examples map is applied."""
        agents = {"test": MockDomainAgent(name="test", example_queries=("original",))}
        examples_map = {"test": ("new query",)}

        providers = create_providers_from_agents(agents, examples_map=examples_map)

        assert providers[0].example_queries == ("new query",)


class TestCreateDualModeProviders:
    """Tests for create_dual_mode_providers factory."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = MagicMock(spec=MCPClient)
        client.is_connected.return_value = False
        client.connect = AsyncMock(return_value=True)
        return client

    @pytest.mark.asyncio
    async def test_local_mode_only_agents(self):
        """Test local mode only uses agents."""
        agents = {
            "datastream": MockDomainAgent(name="datastream"),
        }

        providers = await create_dual_mode_providers(
            agents=agents,
            mode="local",
        )

        assert len(providers) == 1
        assert isinstance(providers[0], AgentToolProvider)

    @pytest.mark.asyncio
    async def test_mcp_mode_only_mcp_servers(self, mock_mcp_client):
        """Test mcp mode only uses MCP servers."""
        servers = [
            MCPServer(uri="mcp://test.com", name="test"),
        ]
        agents = {
            "datastream": MockDomainAgent(name="datastream"),
        }

        providers = await create_dual_mode_providers(
            agents=agents,
            mcp_client=mock_mcp_client,
            mcp_servers=servers,
            mode="mcp",
        )

        assert len(providers) == 1
        assert isinstance(providers[0], MCPToolProvider)

    @pytest.mark.asyncio
    async def test_hybrid_mode_uses_both(self, mock_mcp_client):
        """Test hybrid mode uses both agents and MCP servers."""
        servers = [
            MCPServer(uri="mcp://test.com", name="test"),
        ]
        agents = {
            "datastream": MockDomainAgent(name="datastream"),
        }

        providers = await create_dual_mode_providers(
            agents=agents,
            mcp_client=mock_mcp_client,
            mcp_servers=servers,
            mode="hybrid",
        )

        assert len(providers) == 2
        types = {type(p).__name__ for p in providers}
        assert types == {"AgentToolProvider", "MCPToolProvider"}

    @pytest.mark.asyncio
    async def test_connects_to_mcp_servers(self, mock_mcp_client):
        """Test that MCP servers are connected."""
        servers = [
            MCPServer(uri="mcp://test.com", name="test"),
        ]

        await create_dual_mode_providers(
            mcp_client=mock_mcp_client,
            mcp_servers=servers,
            mode="mcp",
        )

        mock_mcp_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_connection_failure(self, mock_mcp_client):
        """Test that connection failures are handled gracefully."""
        mock_mcp_client.connect = AsyncMock(side_effect=Exception("Connection failed"))
        servers = [
            MCPServer(uri="mcp://test.com", name="test"),
        ]

        providers = await create_dual_mode_providers(
            mcp_client=mock_mcp_client,
            mcp_servers=servers,
            mode="mcp",
        )

        # Should not raise, but provider list may be empty
        assert providers == []

    @pytest.mark.asyncio
    async def test_empty_mode_returns_empty(self):
        """Test that empty configuration returns empty list."""
        providers = await create_dual_mode_providers(mode="local")

        assert providers == []

    @pytest.mark.asyncio
    async def test_skips_already_connected_servers(self, mock_mcp_client):
        """Test that already connected servers are not reconnected."""
        mock_mcp_client.is_connected.return_value = True
        servers = [
            MCPServer(uri="mcp://test.com", name="test"),
        ]

        await create_dual_mode_providers(
            mcp_client=mock_mcp_client,
            mcp_servers=servers,
            mode="mcp",
        )

        mock_mcp_client.connect.assert_not_called()
