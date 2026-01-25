"""
Entity Resolution Service - CLI Entry Point

Usage:
    python -m src.services.entity_resolution [--transport http|mcp|stdio] [options]

Examples:
    # Start HTTP server (default)
    python -m src.services.entity_resolution --port 8085

    # Start with custom config
    python -m src.services.entity_resolution --host 0.0.0.0 --port 8085 --no-redis
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import EntityServiceConfig


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Entity Resolution Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--transport",
        choices=["http", "mcp", "stdio"],
        default="http",
        help="Transport type (default: http)",
    )

    # HTTP options
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind (default: from config or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: from config or 8085)",
    )

    # Database options
    parser.add_argument(
        "--postgres-url",
        default=None,
        help="PostgreSQL connection URL",
    )

    # Redis options
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis connection URL",
    )
    parser.add_argument(
        "--no-redis",
        action="store_true",
        help="Disable Redis caching",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level (default: info)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EntityServiceConfig:
    """Build configuration from args and environment."""
    overrides = {}

    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    if args.postgres_url:
        overrides["postgres_url"] = args.postgres_url
    if args.redis_url:
        overrides["redis_url"] = args.redis_url
    if args.no_redis:
        overrides["redis_enabled"] = False
    if args.log_level:
        overrides["log_level"] = args.log_level.upper()

    return EntityServiceConfig(**overrides)


async def run_http(config: EntityServiceConfig) -> None:
    """Run HTTP server."""
    from .transports.http import run_http_server

    await run_http_server(config)


async def run_mcp(config: EntityServiceConfig) -> None:
    """Run MCP server (stdio transport)."""
    # MCP transport not yet implemented
    raise NotImplementedError("MCP transport not yet implemented")


async def run_stdio(config: EntityServiceConfig) -> None:
    """Run stdio transport for Claude Desktop."""
    # stdio transport not yet implemented
    raise NotImplementedError("stdio transport not yet implemented")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    config = build_config(args)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting Entity Resolution Service ({args.transport} transport)")

    try:
        if args.transport == "http":
            asyncio.run(run_http(config))
        elif args.transport == "mcp":
            asyncio.run(run_mcp(config))
        elif args.transport == "stdio":
            asyncio.run(run_stdio(config))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
