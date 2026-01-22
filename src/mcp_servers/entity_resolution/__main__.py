"""
Entity Resolution MCP Server CLI Entry Point

Run with:
    python -m src.mcp_servers.entity_resolution [OPTIONS]

Examples:
    # Run with SSE/HTTP transport (default)
    python -m src.mcp_servers.entity_resolution --transport sse --port 8080

    # Run with stdio transport (for Claude Desktop)
    python -m src.mcp_servers.entity_resolution --transport stdio

    # Use custom database
    python -m src.mcp_servers.entity_resolution --postgres-url postgresql://user:pass@host:5432/db
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Literal

import typer

from src.mcp_servers.entity_resolution.config import EntityServerConfig

app = typer.Typer(
    name="entity-resolution-mcp",
    help="Entity Resolution MCP Server",
    add_completion=False,
)


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Use stderr for logs to keep stdout clean for stdio transport
    )


@app.command()
def main(
    transport: Literal["stdio", "sse"] = typer.Option(
        "sse",
        "--transport", "-t",
        help="Transport mode: 'stdio' for Claude Desktop, 'sse' for HTTP",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host", "-h",
        help="Host to bind for SSE transport",
    ),
    port: int = typer.Option(
        8080,
        "--port", "-p",
        help="Port for SSE transport",
    ),
    postgres_url: str = typer.Option(
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
        "--postgres-url",
        help="PostgreSQL connection URL",
        envvar="ENTITY_MCP_POSTGRES_URL",
    ),
    redis_url: str = typer.Option(
        "redis://localhost:6379/0",
        "--redis-url",
        help="Redis connection URL",
        envvar="ENTITY_MCP_REDIS_URL",
    ),
    redis_enabled: bool = typer.Option(
        True,
        "--redis/--no-redis",
        help="Enable/disable Redis cache",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level",
    ),
) -> None:
    """
    Run the Entity Resolution MCP Server.

    Supports two transport modes:
    - sse: HTTP server with REST and SSE endpoints (default)
    - stdio: JSON-RPC over stdin/stdout for Claude Desktop
    """
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Entity Resolution MCP Server (transport={transport})")

    # Build config from CLI arguments
    config = EntityServerConfig(
        transport=transport,
        host=host,
        port=port,
        postgres_url=postgres_url,
        redis_url=redis_url,
        redis_enabled=redis_enabled,
        log_level=log_level,
    )

    if transport == "stdio":
        asyncio.run(run_stdio(config))
    else:
        asyncio.run(run_sse(config))


async def run_stdio(config: EntityServerConfig) -> None:
    """Run with stdio transport."""
    from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer
    from src.mcp_servers.entity_resolution.transports.stdio import run_stdio_server

    server = EntityResolutionMCPServer(config)
    await run_stdio_server(server)


async def run_sse(config: EntityServerConfig) -> None:
    """Run with SSE/HTTP transport."""
    from src.mcp_servers.entity_resolution.transports.sse import run_sse_server

    await run_sse_server(config)


if __name__ == "__main__":
    app()
