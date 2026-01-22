"""
stdio Transport for Entity Resolution MCP Server

Uses the official MCP SDK for full protocol compatibility with Claude Desktop.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.mcp_servers.entity_resolution.context import (
    create_stdio_context,
    set_client_context,
)
from src.mcp_servers.entity_resolution.tools import TOOL_DEFINITIONS, ToolHandlers
from src.nl2api.resolution.resolver import ExternalEntityResolver

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """
    Lifespan context manager for the MCP server.

    Initializes database pool and resolver on startup, cleans up on shutdown.
    """
    logger.info("Initializing server resources...")

    # Get config from environment
    import os
    postgres_url = os.environ.get(
        "ENTITY_MCP_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api"
    )

    # Initialize database pool
    db_pool: asyncpg.Pool | None = None
    try:
        import asyncpg
        db_pool = await asyncpg.create_pool(
            postgres_url,
            min_size=1,
            max_size=5,
        )
        logger.info("Database pool initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize database pool: {e}")

    # Initialize resolver
    resolver = ExternalEntityResolver(
        api_endpoint=None,
        api_key=None,
        use_cache=True,
        timeout_seconds=5.0,
        db_pool=db_pool,
    )
    logger.info("Entity resolver initialized")

    # Initialize tool handlers
    tool_handlers = ToolHandlers(resolver)

    # Set up client context
    ctx = create_stdio_context()
    set_client_context(ctx)
    logger.info(f"Client context: session_id={ctx.session_id}")

    try:
        yield {"tool_handlers": tool_handlers, "db_pool": db_pool}
    finally:
        if db_pool:
            await db_pool.close()
            logger.info("Database pool closed")
        logger.info("Server resources cleaned up")


def create_mcp_server() -> Server:
    """Create and configure the MCP server with tool handlers."""

    server = Server(
        name="entity-resolution",
        version="1.0.0",
        lifespan=server_lifespan,
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return available tools."""
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        # Get tool handlers from lifespan context
        ctx = server.request_context
        tool_handlers: ToolHandlers = ctx.lifespan_context["tool_handlers"]

        try:
            result = await tool_handlers.handle_tool_call(name, arguments)
            return [TextContent(type="text", text=str(result))]
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        except Exception as e:
            logger.exception(f"Error calling tool {name}: {e}")
            return [TextContent(type="text", text=f"Internal error: {e}")]

    return server


async def run_stdio_server_sdk() -> None:
    """
    Run the MCP server using the official SDK.

    This provides full protocol compatibility with Claude Desktop.
    """
    logger.info("Starting stdio transport (MCP SDK Server)")

    server = create_mcp_server()

    async with stdio_server() as (read_stream, write_stream):
        logger.info("stdio transport connected")
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)

    logger.info("stdio transport stopped")


async def run_stdio_server(server: "EntityResolutionMCPServer") -> None:
    """
    Run the MCP server using stdio transport.

    Note: The 'server' parameter is ignored - we use the SDK's Server class
    for full protocol compatibility.
    """
    await run_stdio_server_sdk()


# Allow running directly for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    anyio.run(run_stdio_server_sdk)
