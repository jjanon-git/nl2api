"""
Tool Providers

Adapters that implement the ToolProvider protocol for different sources:
- AgentToolProvider: Wraps DomainAgent as a ToolProvider
- MCPToolProvider: Discovers tools from MCP servers

These providers enable a unified interface for routing, whether tools
come from local agents or remote MCP servers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.nl2api.llm.protocols import LLMToolDefinition

if TYPE_CHECKING:
    from src.nl2api.agents.protocols import DomainAgent
    from src.nl2api.mcp.client import MCPClient

logger = logging.getLogger(__name__)


class AgentToolProvider:
    """
    Wraps a DomainAgent as a ToolProvider for use with routers.

    This adapter enables domain agents to be used interchangeably with
    MCP servers in the routing system.
    """

    def __init__(
        self,
        agent: DomainAgent,
        capabilities: tuple[str, ...] | None = None,
        example_queries: tuple[str, ...] | None = None,
    ):
        """
        Initialize the agent tool provider.

        Args:
            agent: The domain agent to wrap
            capabilities: Optional override for agent capabilities
            example_queries: Optional override for example queries
        """
        self._agent = agent
        self._capabilities_override = capabilities
        self._example_queries_override = example_queries

    @property
    def provider_name(self) -> str:
        """Return the agent's domain name."""
        return self._agent.domain_name

    @property
    def provider_description(self) -> str:
        """Return the agent's domain description."""
        return self._agent.domain_description

    @property
    def capabilities(self) -> tuple[str, ...]:
        """
        Return the data types this provider handles.

        Uses override if provided, otherwise checks for agent capabilities
        property, falling back to empty tuple.
        """
        if self._capabilities_override:
            return self._capabilities_override

        # Try to get from agent if it has the property
        return getattr(self._agent, "capabilities", ())

    @property
    def example_queries(self) -> tuple[str, ...]:
        """
        Return example queries this provider handles well.

        Uses override if provided, otherwise checks for agent example_queries
        property, falling back to empty tuple.
        """
        if self._example_queries_override:
            return self._example_queries_override

        # Try to get from agent if it has the property
        return getattr(self._agent, "example_queries", ())

    async def list_tools(self) -> list[LLMToolDefinition]:
        """
        List tools available from this agent.

        Returns the tools defined by the agent's get_tools() method.
        """
        # Check if agent has get_tools method (BaseDomainAgent does)
        if hasattr(self._agent, "get_tools"):
            return self._agent.get_tools()

        # Fallback: return empty list
        logger.warning(f"Agent {self.provider_name} does not implement get_tools()")
        return []

    async def get_tool_description(self, tool_name: str) -> str | None:
        """
        Get detailed description for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool description or None if not found
        """
        tools = await self.list_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool.description
        return None

    @property
    def agent(self) -> DomainAgent:
        """Return the wrapped agent for direct access."""
        return self._agent


class AgentToolExecutor:
    """
    Executes tools via domain agents.

    Maps tool names to agents and invokes the appropriate agent
    to process the request.
    """

    def __init__(
        self,
        providers: list[AgentToolProvider],
    ):
        """
        Initialize the executor.

        Args:
            providers: List of agent tool providers
        """
        self._providers = {p.provider_name: p for p in providers}
        self._tool_to_provider: dict[str, AgentToolProvider] = {}

        # Build tool to provider mapping
        self._build_tool_mapping()

    def _build_tool_mapping(self) -> None:
        """Build mapping from tool names to providers."""
        for provider in self._providers.values():
            # Note: This is sync initialization, tools are cached
            # In a real implementation, you might want async initialization
            if hasattr(provider.agent, "get_tools"):
                for tool in provider.agent.get_tools():
                    self._tool_to_provider[tool.name] = provider

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool via the appropriate agent.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Execution result as dictionary

        Raises:
            ValueError: If tool is not found
        """
        provider = self._tool_to_provider.get(tool_name)
        if not provider:
            raise ValueError(f"Unknown tool: {tool_name}")

        # For now, we return the tool call info
        # Actual execution would require integration with API clients
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "provider": provider.provider_name,
            "status": "pending_execution",
        }


def create_providers_from_agents(
    agents: dict[str, Any],
    capabilities_map: dict[str, tuple[str, ...]] | None = None,
    examples_map: dict[str, tuple[str, ...]] | None = None,
) -> list[AgentToolProvider]:
    """
    Create tool providers from a dictionary of domain agents.

    Args:
        agents: Dictionary mapping domain names to agents
        capabilities_map: Optional mapping of domain to capabilities
        examples_map: Optional mapping of domain to example queries

    Returns:
        List of AgentToolProvider instances
    """
    capabilities_map = capabilities_map or {}
    examples_map = examples_map or {}

    providers = []
    for domain_name, agent in agents.items():
        provider = AgentToolProvider(
            agent=agent,
            capabilities=capabilities_map.get(domain_name),
            example_queries=examples_map.get(domain_name),
        )
        providers.append(provider)

    return providers


# =============================================================================
# MCP Tool Provider - Remote Tool Discovery via MCP Servers
# =============================================================================


class MCPToolProvider:
    """
    Discovers and provides tools from MCP (Model Context Protocol) servers.

    This provider connects to MCP servers (e.g., LSEG Datastream MCP server)
    and exposes their tools through the ToolProvider protocol. This enables
    the routing system to use tools from both local agents and remote MCP
    servers interchangeably.

    Usage:
        from src.nl2api.mcp.client import MCPClient, MCPClientConfig
        from src.nl2api.mcp.protocols import MCPServer

        client = MCPClient(MCPClientConfig())
        server = MCPServer(uri="mcp://datastream.lseg.com", name="datastream")
        await client.connect(server)

        provider = MCPToolProvider(
            server_uri="mcp://datastream.lseg.com",
            mcp_client=client,
            provider_name="datastream",
        )
        tools = await provider.list_tools()
    """

    def __init__(
        self,
        server_uri: str,
        mcp_client: MCPClient | None = None,
        provider_name: str | None = None,
        provider_description: str | None = None,
        capabilities: tuple[str, ...] | None = None,
        example_queries: tuple[str, ...] | None = None,
    ):
        """
        Initialize MCP tool provider.

        Args:
            server_uri: URI of the MCP server
            mcp_client: MCP client instance (must be connected to the server)
            provider_name: Optional name override (defaults to extracted from URI)
            provider_description: Optional description override
            capabilities: Optional capabilities to advertise
            example_queries: Optional example queries
        """
        self._server_uri = server_uri
        self._mcp_client = mcp_client
        self._provider_name = provider_name
        self._provider_description = provider_description
        self._capabilities = capabilities or ()
        self._example_queries = example_queries or ()
        self._cached_tools: list[LLMToolDefinition] | None = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        if self._provider_name:
            return self._provider_name
        # Extract from URI: mcp://datastream.lseg.com -> datastream
        uri_parts = self._server_uri.replace("mcp://", "").split(".")
        return uri_parts[0] if uri_parts else "unknown"

    @property
    def provider_description(self) -> str:
        """Return the provider description."""
        if self._provider_description:
            return self._provider_description
        return f"MCP server at {self._server_uri}"

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Return capabilities for this provider."""
        return self._capabilities

    @property
    def example_queries(self) -> tuple[str, ...]:
        """Return example queries for this provider."""
        return self._example_queries

    @property
    def server_uri(self) -> str:
        """Return the MCP server URI."""
        return self._server_uri

    @property
    def is_connected(self) -> bool:
        """Check if the MCP client is connected to the server."""
        if self._mcp_client is None:
            return False
        return self._mcp_client.is_connected(self._server_uri)

    async def list_tools(self) -> list[LLMToolDefinition]:
        """
        List tools from the MCP server.

        Discovers tools via the MCP tools/list protocol and converts
        them to LLMToolDefinition format.

        Returns:
            List of tool definitions discovered from MCP
        """
        if self._mcp_client is None:
            logger.warning(f"MCP client not configured for {self._server_uri}")
            return []

        if not self.is_connected:
            logger.warning(f"MCP client not connected to {self._server_uri}")
            return []

        # Return cached tools if available
        if self._cached_tools is not None:
            return self._cached_tools

        try:
            # Fetch tools from MCP server
            mcp_tools = await self._mcp_client.list_tools(self._server_uri)

            # Convert MCP tool definitions to LLM tool definitions
            self._cached_tools = [tool.to_llm_tool_definition() for tool in mcp_tools]

            logger.info(
                f"Discovered {len(self._cached_tools)} tools from MCP server {self._server_uri}"
            )
            return self._cached_tools

        except Exception as e:
            logger.error(f"Failed to list tools from MCP server {self._server_uri}: {e}")
            return []

    async def get_tool_description(self, tool_name: str) -> str | None:
        """
        Get description for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool description or None if not found
        """
        tools = await self.list_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool.description
        return None

    def invalidate_cache(self) -> None:
        """Invalidate the cached tools, forcing a refresh on next list_tools()."""
        self._cached_tools = None

    async def refresh_tools(self) -> list[LLMToolDefinition]:
        """Force refresh of tools from the MCP server."""
        self.invalidate_cache()
        return await self.list_tools()


class MCPToolExecutor:
    """
    Executes tools via MCP tools/call protocol.

    This executor routes tool execution to the appropriate MCP server
    based on the tool name and server mapping.
    """

    def __init__(
        self,
        mcp_client: MCPClient | None = None,
        providers: list[MCPToolProvider] | None = None,
    ):
        """
        Initialize MCP tool executor.

        Args:
            mcp_client: MCP client instance
            providers: List of MCP tool providers for tool-to-server mapping
        """
        self._mcp_client = mcp_client
        self._providers = providers or []
        self._tool_to_server: dict[str, str] = {}

    async def build_tool_mapping(self) -> None:
        """Build mapping from tool names to server URIs."""
        self._tool_to_server.clear()

        for provider in self._providers:
            tools = await provider.list_tools()
            for tool in tools:
                self._tool_to_server[tool.name] = provider.server_uri

        logger.info(f"Built tool mapping for {len(self._tool_to_server)} tools")

    def add_provider(self, provider: MCPToolProvider) -> None:
        """Add a provider to the executor."""
        self._providers.append(provider)

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool via MCP.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Execution result as dictionary

        Raises:
            RuntimeError: If MCP client not configured
            ValueError: If tool not found
        """
        if self._mcp_client is None:
            raise RuntimeError("MCP client not configured")

        # Find server for this tool
        server_uri = self._tool_to_server.get(tool_name)
        if not server_uri:
            # Try to find in providers
            for provider in self._providers:
                tools = await provider.list_tools()
                if any(t.name == tool_name for t in tools):
                    server_uri = provider.server_uri
                    self._tool_to_server[tool_name] = server_uri
                    break

        if not server_uri:
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            # Execute via MCP client
            result = await self._mcp_client.call_tool(
                server_uri=server_uri,
                tool_name=tool_name,
                arguments=arguments,
            )

            if result.is_error:
                return {
                    "tool_name": tool_name,
                    "error": result.error_message,
                    "is_error": True,
                }

            return {
                "tool_name": tool_name,
                "content": result.content,
                "execution_time_ms": result.execution_time_ms,
                "is_error": False,
            }

        except Exception as e:
            logger.error(f"MCP tool execution failed: {tool_name}: {e}")
            return {
                "tool_name": tool_name,
                "error": str(e),
                "is_error": True,
            }
