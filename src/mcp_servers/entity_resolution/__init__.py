"""
NL2API MCP Server

Ask natural language questions to financial services APIs via Model Context Protocol.
Supports stdio and HTTP/SSE transports for Claude Desktop and production deployments.
"""

from src.mcp_servers.entity_resolution.config import EntityServerConfig
from src.mcp_servers.entity_resolution.context import ClientContext
from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

__all__ = ["EntityResolutionMCPServer", "EntityServerConfig", "ClientContext"]
