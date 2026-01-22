"""
MCP (Model Context Protocol) Module

Provides support for accessing tools and context from remote MCP servers
(when LSEG APIs are exposed as MCP servers).

This module is designed for future-proofing - the MCP client and protocols
are ready for when MCP servers become available, while maintaining full
backward compatibility with the existing agent-based architecture.
"""

from src.nl2api.mcp.cache import InMemoryMCPCache, MCPCache
from src.nl2api.mcp.client import MCPClient, MCPClientConfig
from src.nl2api.mcp.context import (
    ContextProvider,
    MCPContextRetriever,
)
from src.nl2api.mcp.protocols import (
    MCPCapabilities,
    MCPResource,
    MCPServer,
    MCPToolDefinition,
    MCPToolResult,
)

__all__ = [
    # Protocols and types
    "MCPResource",
    "MCPToolDefinition",
    "MCPToolResult",
    "MCPServer",
    "MCPCapabilities",
    # Client
    "MCPClient",
    "MCPClientConfig",
    # Cache
    "MCPCache",
    "InMemoryMCPCache",
    # Context retrieval
    "ContextProvider",
    "MCPContextRetriever",
]
