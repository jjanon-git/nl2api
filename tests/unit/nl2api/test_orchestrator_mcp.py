"""Tests for NL2API orchestrator MCP routing integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from CONTRACTS import ToolCall
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import (
    LLMResponse,
    LLMToolDefinition,
)
from src.nl2api.mcp.protocols import MCPServer, MCPToolDefinition, MCPToolParameter
from src.nl2api.orchestrator import NL2APIOrchestrator


@dataclass
class MockLLMProvider:
    """Mock LLM provider."""

    model_name: str = "mock"
    response_content: str = "estimates"

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return LLMResponse(content=self.response_content)

    async def complete_with_retry(
        self, messages, tools=None, temperature=0.0, max_tokens=4096, max_retries=3
    ):
        return LLMResponse(content=self.response_content)


@dataclass
class MockAgent:
    """Mock domain agent."""

    domain_name: str = "estimates"
    domain_description: str = "Test agent for estimates"
    can_handle_score: float = 0.9
    result: AgentResult = field(default_factory=lambda: AgentResult(
        tool_calls=(ToolCall(tool_name="get_data", arguments={"RICs": ["AAPL.O"]}),),
        confidence=0.9,
        domain="estimates",
    ))

    async def can_handle(self, query: str) -> float:
        return self.can_handle_score

    async def process(self, context: AgentContext) -> AgentResult:
        return self.result

    def get_tools(self) -> list[LLMToolDefinition]:
        return [
            LLMToolDefinition(
                name="get_data",
                description="Get financial data",
                parameters={"type": "object", "properties": {"RICs": {"type": "array"}}},
            )
        ]


@dataclass
class MockMCPClient:
    """Mock MCP client for testing."""

    connected_servers: dict[str, MCPServer] = field(default_factory=dict)
    tools_by_server: dict[str, list[MCPToolDefinition]] = field(default_factory=dict)
    connect_should_fail: bool = False

    async def connect(self, server: MCPServer) -> bool:
        if self.connect_should_fail:
            raise ConnectionError(f"Failed to connect to {server.uri}")
        self.connected_servers[server.uri] = server
        return True

    async def disconnect(self, server_uri: str) -> None:
        self.connected_servers.pop(server_uri, None)

    def is_connected(self, server_uri: str) -> bool:
        return server_uri in self.connected_servers

    async def list_tools(self, server_uri: str) -> list[MCPToolDefinition]:
        return self.tools_by_server.get(server_uri, [])

    async def call_tool(self, server_uri: str, tool_name: str, arguments: dict) -> Any:
        return {"status": "ok", "tool": tool_name}


class TestOrchestratorMCPMode:
    """Test suite for orchestrator MCP routing modes."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.llm = MockLLMProvider()
        self.agent = MockAgent()

    @pytest.mark.asyncio
    async def test_router_uses_local_providers_when_mode_local(self) -> None:
        """Test router uses AgentToolProvider when mcp_mode='local'."""
        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            mcp_mode="local",
        )

        # Router should be initialized synchronously for local mode
        assert orchestrator._router_initialized is True
        assert orchestrator._router is not None

        # Process a query to verify it works
        result = await orchestrator.process("What is Apple's EPS?")
        assert result.domain == "estimates"
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_router_lazy_init_for_mcp_mode(self) -> None:
        """Test router is lazily initialized for mcp mode."""
        mcp_client = MockMCPClient()
        mcp_server = MCPServer(uri="mcp://test.local", name="test")

        # Add tools to mock MCP client
        mcp_client.tools_by_server["mcp://test.local"] = [
            MCPToolDefinition(
                name="mcp_get_data",
                description="Get data via MCP",
                parameters=(
                    MCPToolParameter(name="rics", description="RIC codes", type="array"),
                ),
            )
        ]

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            mcp_mode="mcp",
            mcp_client=mcp_client,
            mcp_servers=[mcp_server],
        )

        # Router should NOT be initialized yet (lazy init)
        assert orchestrator._router_initialized is False
        assert orchestrator._router is None

        # Process query triggers lazy init
        await orchestrator.process("What is Apple's EPS?")

        # Router should now be initialized
        assert orchestrator._router_initialized is True
        assert orchestrator._router is not None

    @pytest.mark.asyncio
    async def test_router_uses_mcp_providers_when_mode_mcp(self) -> None:
        """Test router uses MCPToolProvider when mcp_mode='mcp'."""
        mcp_client = MockMCPClient()
        mcp_server = MCPServer(uri="mcp://estimates.local", name="estimates")

        # Add tools to mock MCP client
        mcp_client.tools_by_server["mcp://estimates.local"] = [
            MCPToolDefinition(
                name="get_estimates",
                description="Get earnings estimates",
                parameters=(
                    MCPToolParameter(name="rics", description="RIC codes", type="array"),
                ),
            )
        ]

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={},  # No local agents
            mcp_mode="mcp",
            mcp_client=mcp_client,
            mcp_servers=[mcp_server],
        )

        # Trigger lazy init
        await orchestrator.process("What is Apple's EPS?")

        # MCP client should have been connected
        assert mcp_client.is_connected("mcp://estimates.local")

        # Router should have MCP provider
        assert orchestrator._router is not None
        # The router's providers should include MCP providers
        providers = orchestrator._router._providers
        assert len(providers) > 0

    @pytest.mark.asyncio
    async def test_hybrid_mode_includes_both_providers(self) -> None:
        """Test hybrid mode includes both local agents and MCP providers."""
        mcp_client = MockMCPClient()
        mcp_server = MCPServer(uri="mcp://datastream.local", name="datastream")

        mcp_client.tools_by_server["mcp://datastream.local"] = [
            MCPToolDefinition(
                name="get_timeseries",
                description="Get time series data",
                parameters=(),
            )
        ]

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            mcp_mode="hybrid",
            mcp_client=mcp_client,
            mcp_servers=[mcp_server],
        )

        # Trigger lazy init
        await orchestrator.process("What is Apple's EPS?")

        # Router should be initialized
        assert orchestrator._router is not None

        # Should have both local and MCP providers
        providers = orchestrator._router._providers
        # At least 2 providers: 1 local agent + 1 MCP server
        assert len(providers) >= 2

    @pytest.mark.asyncio
    async def test_fallback_to_local_on_mcp_failure(self) -> None:
        """Test fallback to local when MCP unavailable and fallback enabled."""
        mcp_client = MockMCPClient()
        mcp_client.connect_should_fail = True  # Simulate connection failure
        mcp_server = MCPServer(uri="mcp://failing.local", name="failing")

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            mcp_mode="mcp",
            mcp_client=mcp_client,
            mcp_servers=[mcp_server],
            mcp_fallback_enabled=True,
        )

        # Should not raise, should fall back to local agents
        result = await orchestrator.process("What is Apple's EPS?")

        # Should still get a result (from fallback)
        assert result is not None
        # Router should have fallen back to local providers
        assert orchestrator._router is not None

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self) -> None:
        """Test no fallback when mcp_fallback_enabled=False."""
        mcp_client = MockMCPClient()
        mcp_client.connect_should_fail = True
        mcp_server = MCPServer(uri="mcp://failing.local", name="failing")

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            mcp_mode="mcp",
            mcp_client=mcp_client,
            mcp_servers=[mcp_server],
            mcp_fallback_enabled=False,
        )

        # Process should still work but with no providers warning
        result = await orchestrator.process("What is Apple's EPS?")

        # Should get some result (even if it's an error response)
        assert result is not None

    @pytest.mark.asyncio
    async def test_mcp_mode_without_client_uses_local(self) -> None:
        """Test mcp_mode without client gracefully falls back to local."""
        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            mcp_mode="mcp",
            mcp_client=None,  # No client provided
            mcp_servers=[],
            mcp_fallback_enabled=True,
        )

        result = await orchestrator.process("What is Apple's EPS?")

        # Should work with local fallback
        assert result is not None
        assert result.domain == "estimates"

    @pytest.mark.asyncio
    async def test_orchestrator_stores_mcp_config(self) -> None:
        """Test that orchestrator stores MCP configuration correctly."""
        mcp_client = MockMCPClient()
        mcp_servers = [
            MCPServer(uri="mcp://server1.local", name="server1"),
            MCPServer(uri="mcp://server2.local", name="server2"),
        ]

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            mcp_mode="hybrid",
            mcp_client=mcp_client,
            mcp_servers=mcp_servers,
        )

        assert orchestrator._mcp_client is mcp_client
        assert len(orchestrator._mcp_servers) == 2
        assert orchestrator._mcp_mode == "hybrid"

    @pytest.mark.asyncio
    async def test_custom_router_bypasses_mcp_init(self) -> None:
        """Test that custom router is used as-is, bypassing MCP init."""
        from src.nl2api.routing.llm_router import LLMToolRouter
        from src.nl2api.routing.providers import AgentToolProvider

        custom_router = LLMToolRouter(
            llm=self.llm,
            tool_providers=[AgentToolProvider(self.agent)],
        )

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            router=custom_router,
            mcp_mode="mcp",  # This should be ignored since router is provided
        )

        # Router should be the custom one
        assert orchestrator._router is custom_router
        assert orchestrator._router_initialized is True

        # Process should use the custom router
        result = await orchestrator.process("What is Apple's EPS?")
        assert result is not None


class TestOrchestratorMCPProviderIntegration:
    """Test suite for MCP provider integration details."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.llm = MockLLMProvider()

    @pytest.mark.asyncio
    async def test_multiple_mcp_servers(self) -> None:
        """Test orchestrator with multiple MCP servers."""
        mcp_client = MockMCPClient()

        # Set up multiple servers with different tools
        servers = [
            MCPServer(uri="mcp://estimates.local", name="estimates"),
            MCPServer(uri="mcp://fundamentals.local", name="fundamentals"),
        ]

        mcp_client.tools_by_server["mcp://estimates.local"] = [
            MCPToolDefinition(name="get_eps", description="Get EPS", parameters=()),
        ]
        mcp_client.tools_by_server["mcp://fundamentals.local"] = [
            MCPToolDefinition(name="get_revenue", description="Get revenue", parameters=()),
        ]

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={},
            mcp_mode="mcp",
            mcp_client=mcp_client,
            mcp_servers=servers,
        )

        # Trigger init
        await orchestrator.process("Get financial data")

        # Both servers should be connected
        assert mcp_client.is_connected("mcp://estimates.local")
        assert mcp_client.is_connected("mcp://fundamentals.local")

    @pytest.mark.asyncio
    async def test_partial_mcp_connection_failure(self) -> None:
        """Test handling of partial MCP connection failures in hybrid mode."""
        # Create a client that fails on specific servers
        class PartialFailMCPClient(MockMCPClient):
            async def connect(self, server: MCPServer) -> bool:
                if "failing" in server.uri:
                    raise ConnectionError(f"Failed: {server.uri}")
                self.connected_servers[server.uri] = server
                return True

        mcp_client = PartialFailMCPClient()
        servers = [
            MCPServer(uri="mcp://working.local", name="working"),
            MCPServer(uri="mcp://failing.local", name="failing"),
        ]

        mcp_client.tools_by_server["mcp://working.local"] = [
            MCPToolDefinition(name="working_tool", description="Works", parameters=()),
        ]

        agent = MockAgent()
        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": agent},
            mcp_mode="hybrid",
            mcp_client=mcp_client,
            mcp_servers=servers,
        )

        # Should succeed with working server + local agent
        result = await orchestrator.process("Get data")
        assert result is not None

        # Working server should be connected
        assert mcp_client.is_connected("mcp://working.local")
        # Failing server should not be connected
        assert not mcp_client.is_connected("mcp://failing.local")
