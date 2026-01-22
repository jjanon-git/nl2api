"""
MCP Transport Implementations

Provides transport layers for the Entity Resolution MCP Server:
- stdio: For Claude Desktop and local MCP clients
- sse: For HTTP/SSE production deployments
"""

from src.mcp_servers.entity_resolution.transports.sse import create_app, run_sse_server
from src.mcp_servers.entity_resolution.transports.stdio import run_stdio_server

__all__ = ["create_app", "run_sse_server", "run_stdio_server"]
