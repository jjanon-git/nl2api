"""
MCP Client Implementation

Provides a client for interacting with MCP (Model Context Protocol) servers.
Supports tool discovery, tool execution, and resource retrieval.

This implementation is designed to work with future LSEG MCP servers while
maintaining compatibility with the existing agent-based architecture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.nl2api.mcp.protocols import (
    MCPCapabilities,
    MCPResource,
    MCPServer,
    MCPToolDefinition,
    MCPToolResult,
)

logger = logging.getLogger(__name__)


@dataclass
class MCPClientConfig:
    """Configuration for MCP client."""

    default_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    connection_pool_size: int = 10
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


class MCPConnectionError(Exception):
    """Raised when connection to MCP server fails."""

    pass


class MCPToolError(Exception):
    """Raised when tool execution fails."""

    pass


class MCPClient:
    """
    Client for MCP (Model Context Protocol) servers.

    Manages connections to multiple MCP servers and provides a unified
    interface for tool discovery, execution, and resource retrieval.

    Example:
        client = MCPClient(config)
        await client.connect(MCPServer(uri="mcp://datastream.lseg.com", name="datastream"))
        tools = await client.list_tools("mcp://datastream.lseg.com")
        result = await client.call_tool("mcp://datastream.lseg.com", "get_price", {"ric": "AAPL.O"})
    """

    def __init__(self, config: MCPClientConfig | None = None):
        """
        Initialize MCP client.

        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self._config = config or MCPClientConfig()
        self._servers: dict[str, MCPServer] = {}
        self._tool_cache: dict[str, list[MCPToolDefinition]] = {}
        self._resource_cache: dict[str, list[MCPResource]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._lock = asyncio.Lock()

    @property
    def connected_servers(self) -> list[str]:
        """Return list of connected server URIs."""
        return list(self._servers.keys())

    async def connect(self, server: MCPServer) -> bool:
        """
        Connect to an MCP server.

        Performs capability negotiation and caches server metadata.

        Args:
            server: MCP server configuration

        Returns:
            True if connection successful

        Raises:
            MCPConnectionError: If connection fails
        """
        async with self._lock:
            if server.uri in self._servers:
                logger.debug(f"Already connected to {server.uri}")
                return True

            try:
                # In a real implementation, this would perform MCP handshake
                # For now, we simulate successful connection
                capabilities = await self._negotiate_capabilities(server)

                # Store server with discovered capabilities
                connected_server = MCPServer(
                    uri=server.uri,
                    name=server.name,
                    description=server.description,
                    capabilities=capabilities,
                    api_key=server.api_key,
                    timeout_seconds=server.timeout_seconds,
                )
                self._servers[server.uri] = connected_server

                logger.info(
                    f"Connected to MCP server: {server.name} ({server.uri}) "
                    f"[tools={capabilities.tools}, resources={capabilities.resources}]"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server.uri}: {e}")
                raise MCPConnectionError(f"Connection failed: {e}") from e

    async def _negotiate_capabilities(self, server: MCPServer) -> MCPCapabilities:
        """
        Negotiate capabilities with MCP server.

        In a real implementation, this would send initialize request
        and parse the server's capability response.
        """
        # Placeholder: In production, this would make actual MCP protocol calls
        # For now, assume servers support tools and resources
        return MCPCapabilities(
            tools=True,
            resources=True,
            prompts=False,
            sampling=False,
            logging=False,
        )

    async def disconnect(self, server_uri: str) -> None:
        """
        Disconnect from an MCP server.

        Clears cached data for the server.

        Args:
            server_uri: URI of server to disconnect from
        """
        async with self._lock:
            if server_uri in self._servers:
                del self._servers[server_uri]
                self._tool_cache.pop(server_uri, None)
                self._resource_cache.pop(server_uri, None)
                self._cache_timestamps.pop(server_uri, None)
                logger.info(f"Disconnected from MCP server: {server_uri}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        server_uris = list(self._servers.keys())
        for uri in server_uris:
            await self.disconnect(uri)

    def is_connected(self, server_uri: str) -> bool:
        """Check if connected to a specific server."""
        return server_uri in self._servers

    def get_server(self, server_uri: str) -> MCPServer | None:
        """Get server configuration by URI."""
        return self._servers.get(server_uri)

    async def list_tools(self, server_uri: str) -> list[MCPToolDefinition]:
        """
        List available tools from an MCP server.

        Results are cached according to config.cache_ttl_seconds.

        Args:
            server_uri: URI of the MCP server

        Returns:
            List of tool definitions

        Raises:
            MCPConnectionError: If not connected to server
        """
        if server_uri not in self._servers:
            raise MCPConnectionError(f"Not connected to server: {server_uri}")

        # Check cache
        if self._config.enable_caching:
            cache_key = f"tools:{server_uri}"
            if self._is_cache_valid(cache_key):
                return self._tool_cache.get(server_uri, [])

        # Fetch tools from server
        tools = await self._fetch_tools(server_uri)

        # Update cache
        if self._config.enable_caching:
            self._tool_cache[server_uri] = tools
            self._cache_timestamps[f"tools:{server_uri}"] = time.time()

        return tools

    async def _fetch_tools(self, server_uri: str) -> list[MCPToolDefinition]:
        """
        Fetch tools from MCP server.

        In a real implementation, this would call the tools/list endpoint.
        """
        # Placeholder: In production, this would make actual MCP protocol calls
        # Return empty list for now - actual implementation would parse MCP response
        logger.debug(f"Fetching tools from {server_uri}")
        return []

    async def call_tool(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """
        Execute a tool on an MCP server.

        Args:
            server_uri: URI of the MCP server
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            MCPConnectionError: If not connected to server
            MCPToolError: If tool execution fails
        """
        if server_uri not in self._servers:
            raise MCPConnectionError(f"Not connected to server: {server_uri}")

        server = self._servers[server_uri]
        start_time = time.time()

        try:
            result = await self._execute_tool(server, tool_name, arguments)
            execution_time_ms = int((time.time() - start_time) * 1000)

            return MCPToolResult(
                tool_name=tool_name,
                content=result,
                is_error=False,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Tool execution failed: {tool_name} on {server_uri}: {e}")

            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
            )

    async def _execute_tool(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute tool via MCP protocol.

        In a real implementation, this would call the tools/call endpoint.
        """
        # Placeholder: In production, this would make actual MCP protocol calls
        logger.debug(f"Executing tool {tool_name} on {server.uri} with args: {arguments}")
        raise MCPToolError(
            f"MCP tool execution not yet implemented. "
            f"Tool: {tool_name}, Server: {server.uri}"
        )

    async def list_resources(self, server_uri: str) -> list[MCPResource]:
        """
        List available resources from an MCP server.

        Results are cached according to config.cache_ttl_seconds.

        Args:
            server_uri: URI of the MCP server

        Returns:
            List of available resources

        Raises:
            MCPConnectionError: If not connected to server
        """
        if server_uri not in self._servers:
            raise MCPConnectionError(f"Not connected to server: {server_uri}")

        # Check cache
        if self._config.enable_caching:
            cache_key = f"resources:{server_uri}"
            if self._is_cache_valid(cache_key):
                return self._resource_cache.get(server_uri, [])

        # Fetch resources from server
        resources = await self._fetch_resources(server_uri)

        # Update cache
        if self._config.enable_caching:
            self._resource_cache[server_uri] = resources
            self._cache_timestamps[f"resources:{server_uri}"] = time.time()

        return resources

    async def _fetch_resources(self, server_uri: str) -> list[MCPResource]:
        """
        Fetch resources from MCP server.

        In a real implementation, this would call the resources/list endpoint.
        """
        # Placeholder: In production, this would make actual MCP protocol calls
        logger.debug(f"Fetching resources from {server_uri}")
        return []

    async def read_resource(self, server_uri: str, resource_uri: str) -> MCPResource:
        """
        Read a specific resource from an MCP server.

        Args:
            server_uri: URI of the MCP server
            resource_uri: URI of the resource to read

        Returns:
            Resource with content populated

        Raises:
            MCPConnectionError: If not connected to server
        """
        if server_uri not in self._servers:
            raise MCPConnectionError(f"Not connected to server: {server_uri}")

        return await self._read_resource(server_uri, resource_uri)

    async def _read_resource(
        self,
        server_uri: str,
        resource_uri: str,
    ) -> MCPResource:
        """
        Read resource via MCP protocol.

        In a real implementation, this would call the resources/read endpoint.
        """
        # Placeholder: In production, this would make actual MCP protocol calls
        logger.debug(f"Reading resource {resource_uri} from {server_uri}")

        # Return empty resource - actual implementation would fetch content
        return MCPResource(
            uri=resource_uri,
            name=resource_uri.split("/")[-1],
            description="Resource not yet fetched (MCP not implemented)",
            content=None,
        )

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False

        age = time.time() - self._cache_timestamps[cache_key]
        return age < self._config.cache_ttl_seconds

    def invalidate_cache(self, server_uri: str | None = None) -> None:
        """
        Invalidate cached data.

        Args:
            server_uri: If provided, only invalidate cache for this server.
                       If None, invalidate all cached data.
        """
        if server_uri:
            keys_to_remove = [
                k for k in self._cache_timestamps if server_uri in k
            ]
            for key in keys_to_remove:
                del self._cache_timestamps[key]
            self._tool_cache.pop(server_uri, None)
            self._resource_cache.pop(server_uri, None)
        else:
            self._cache_timestamps.clear()
            self._tool_cache.clear()
            self._resource_cache.clear()

    async def health_check(self, server_uri: str) -> bool:
        """
        Check if MCP server is healthy and responsive.

        Args:
            server_uri: URI of the MCP server

        Returns:
            True if server is healthy
        """
        if server_uri not in self._servers:
            return False

        try:
            # In a real implementation, this would ping the server
            # For now, just check if we have the server registered
            return True
        except Exception:
            return False
