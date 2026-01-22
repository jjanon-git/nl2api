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

        if not line:
            return None

        return json.loads(line.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON received: {e}")
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

            # Write response
            write_message(response)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down")
    except Exception as e:
        logger.exception(f"Error in stdio server: {e}")
    finally:
        await server.shutdown()
        logger.info("stdio transport stopped")
