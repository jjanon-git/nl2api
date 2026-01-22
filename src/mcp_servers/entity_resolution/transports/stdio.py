"""
stdio Transport for Entity Resolution MCP Server

Implements the stdio transport for use with Claude Desktop and other
local MCP clients that communicate via stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING

from src.mcp_servers.entity_resolution.context import (
    create_stdio_context,
    set_client_context,
)

if TYPE_CHECKING:
    from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

logger = logging.getLogger(__name__)


async def read_message() -> dict | None:
    """
    Read a JSON-RPC message from stdin.

    Returns:
        Parsed JSON message or None if EOF
    """
    loop = asyncio.get_event_loop()

    try:
        # Read line from stdin (blocking operation wrapped in executor)
        line = await loop.run_in_executor(None, sys.stdin.readline)

        logger.debug(f"Read line from stdin: {repr(line)[:100]}")

        if not line:
            logger.debug("Empty line received (EOF)")
            return None

        stripped = line.strip()
        if not stripped:
            # Empty line (just whitespace), skip and read next
            logger.debug("Whitespace-only line, reading next")
            return await read_message()

        return json.loads(stripped)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON received: {e}, line was: {repr(line)[:100]}")
        return None
    except Exception as e:
        logger.error(f"Error reading message: {e}")
        return None


def write_message(message: dict) -> None:
    """
    Write a JSON-RPC message to stdout.

    Args:
        message: Message to write
    """
    try:
        line = json.dumps(message)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Error writing message: {e}")


async def run_stdio_server(server: "EntityResolutionMCPServer") -> None:
    """
    Run the MCP server using stdio transport.

    Reads JSON-RPC messages from stdin and writes responses to stdout.
    Used by Claude Desktop and other local MCP clients.

    Args:
        server: The MCP server instance to run
    """
    logger.info("Starting stdio transport")

    # Set up client context for this stdio session
    # (single client per process, session persists for duration)
    ctx = create_stdio_context()
    set_client_context(ctx)
    logger.info(f"Client context: session_id={ctx.session_id}, transport=stdio")

    try:
        # Initialize server
        await server.initialize()

        # Message loop
        while True:
            message = await read_message()

            if message is None:
                logger.info("EOF received, shutting down")
                break

            logger.debug(f"Received message: {message.get('method', 'unknown')}")

            # Handle the message
            response = await server.handle_message(message)

            # Write response (unless it's a notification with no response)
            if response is not None:
                write_message(response)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down")
    except Exception as e:
        logger.exception(f"Error in stdio server: {e}")
    finally:
        await server.shutdown()
        logger.info("stdio transport stopped")
