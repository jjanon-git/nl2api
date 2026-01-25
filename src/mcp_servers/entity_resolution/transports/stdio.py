"""
stdio Transport for NL2API MCP Server

Uses the official MCP SDK for full protocol compatibility with Claude Desktop.
Exposes tools for asking natural language questions to financial services APIs
(Datastream, Estimates, Fundamentals, Officers, Screening) plus entity resolution.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.mcp_servers.entity_resolution.context import (
    create_stdio_context,
    set_client_context,
)
from src.mcp_servers.entity_resolution.nl2api_tools import (
    NL2API_TOOL_DEFINITIONS,
    NL2APIToolHandlers,
)
from src.mcp_servers.entity_resolution.tools import TOOL_DEFINITIONS, ToolHandlers
from src.nl2api.resolution.resolver import ExternalEntityResolver

if TYPE_CHECKING:
    import asyncpg

    from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """
    Lifespan context manager for the MCP server.

    Initializes:
    - Database pool for entity resolution
    - Entity resolver
    - LLM provider (if ANTHROPIC_API_KEY available)
    - Domain agents and orchestrator (if LLM available)
    """
    logger.info("Initializing server resources...")

    # Load .env file if it exists (for ANTHROPIC_API_KEY etc.)
    try:
        from dotenv import load_dotenv

        # Look for .env in the project root (where we're running from)
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env loading")

    # Get config from environment
    postgres_url = os.environ.get(
        "ENTITY_MCP_POSTGRES_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api"
    )
    # Check both prefixed and non-prefixed env var names
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "NL2API_ANTHROPIC_API_KEY"
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
        _internal=True,
    )
    logger.info("Entity resolver initialized")

    # Initialize entity resolution tool handlers
    entity_tool_handlers = ToolHandlers(resolver)

    # Initialize NL2API components (if API key available)
    nl2api_handlers: NL2APIToolHandlers | None = None
    nl2api_enabled = False

    if anthropic_api_key:
        try:
            from src.nl2api.agents.datastream import DatastreamAgent
            from src.nl2api.agents.estimates import EstimatesAgent
            from src.nl2api.agents.fundamentals import FundamentalsAgent
            from src.nl2api.agents.officers import OfficersAgent
            from src.nl2api.agents.screening import ScreeningAgent
            from src.nl2api.llm.claude import ClaudeProvider
            from src.nl2api.orchestrator import NL2APIOrchestrator

            # Initialize LLM provider (use Haiku for placeholder generation)
            llm = ClaudeProvider(
                api_key=anthropic_api_key,
                model="claude-3-5-haiku-latest",
            )
            logger.info("LLM provider initialized (claude-3-5-haiku-latest)")

            # Initialize domain agents
            agents = {
                "datastream": DatastreamAgent(llm),
                "estimates": EstimatesAgent(llm),
                "fundamentals": FundamentalsAgent(llm),
                "officers": OfficersAgent(llm),
                "screening": ScreeningAgent(llm),
            }
            logger.info(f"Domain agents initialized: {list(agents.keys())}")

            # Initialize router with the same LLM (avoids orchestrator creating
            # a new NL2APIConfig that would look for NL2API_ANTHROPIC_API_KEY)
            from src.nl2api.routing.llm_router import LLMToolRouter
            from src.nl2api.routing.providers import AgentToolProvider

            router = LLMToolRouter(
                llm=llm,
                tool_providers=[AgentToolProvider(agent) for agent in agents.values()],
            )
            logger.info("Query router initialized")

            # Initialize orchestrator with pre-configured router
            orchestrator = NL2APIOrchestrator(
                llm=llm,
                agents=agents,
                entity_resolver=resolver,
                router=router,
            )
            logger.info("NL2API orchestrator initialized")

            # Initialize NL2API tool handlers
            nl2api_handlers = NL2APIToolHandlers(
                orchestrator=orchestrator,
                agents=agents,
                llm=llm,
                resolver=resolver,
            )
            nl2api_enabled = True
            logger.info("NL2API tools enabled")

        except Exception as e:
            logger.warning(f"Failed to initialize NL2API components: {e}")
            logger.info("NL2API tools will be disabled, entity resolution only")
    else:
        logger.info("ANTHROPIC_API_KEY not set - NL2API tools disabled, entity resolution only")

    # Set up client context
    ctx = create_stdio_context()
    set_client_context(ctx)
    logger.info(f"Client context: session_id={ctx.session_id}")

    try:
        yield {
            "entity_tool_handlers": entity_tool_handlers,
            "nl2api_handlers": nl2api_handlers,
            "nl2api_enabled": nl2api_enabled,
            "db_pool": db_pool,
        }
    finally:
        if db_pool:
            await db_pool.close()
            logger.info("Database pool closed")
        logger.info("Server resources cleaned up")


def create_mcp_server() -> Server:
    """Create and configure the MCP server with all tool handlers."""

    server = Server(
        name="nl2api",
        version="1.0.0",
        lifespan=server_lifespan,
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return available tools based on what's initialized."""
        ctx = server.request_context
        nl2api_enabled = ctx.lifespan_context.get("nl2api_enabled", False)

        # Always include entity resolution tools
        tools = [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOL_DEFINITIONS
        ]

        # Add NL2API tools if enabled
        if nl2api_enabled:
            tools.extend(
                [
                    Tool(
                        name=t["name"],
                        description=t["description"],
                        inputSchema=t["inputSchema"],
                    )
                    for t in NL2API_TOOL_DEFINITIONS
                ]
            )

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls, routing to appropriate handler."""
        ctx = server.request_context
        entity_handlers: ToolHandlers = ctx.lifespan_context["entity_tool_handlers"]
        nl2api_handlers: NL2APIToolHandlers | None = ctx.lifespan_context.get("nl2api_handlers")
        nl2api_enabled = ctx.lifespan_context.get("nl2api_enabled", False)

        # Entity resolution tools
        entity_tool_names = {t["name"] for t in TOOL_DEFINITIONS}
        # NL2API tools
        nl2api_tool_names = {t["name"] for t in NL2API_TOOL_DEFINITIONS}

        try:
            if name in entity_tool_names:
                result = await entity_handlers.handle_tool_call(name, arguments)
                return [TextContent(type="text", text=str(result))]

            elif name in nl2api_tool_names:
                if not nl2api_enabled or not nl2api_handlers:
                    return [
                        TextContent(
                            type="text",
                            text="Error: NL2API tools require ANTHROPIC_API_KEY to be set",
                        )
                    ]
                result = await nl2api_handlers.handle_tool_call(name, arguments)
                # Return as formatted JSON for readability
                import json

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                return [TextContent(type="text", text=f"Error: Unknown tool: {name}")]

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


async def run_stdio_server(server: EntityResolutionMCPServer) -> None:
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
