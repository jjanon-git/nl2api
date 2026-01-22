"""
stdio Transport for Entity Resolution MCP Server

Uses the official MCP SDK's stdio transport for compatibility with
Claude Desktop and other MCP clients.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import anyio
from mcp.server.stdio import stdio_server
from mcp.shared.session import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCRequest, JSONRPCResponse, JSONRPCError

from src.mcp_servers.entity_resolution.context import (
    create_stdio_context,
    set_client_context,
)

if TYPE_CHECKING:
    from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

logger = logging.getLogger(__name__)


async def run_stdio_server(server: "EntityResolutionMCPServer") -> None:
    """
    Run the MCP server using stdio transport.

    Uses the official MCP SDK's stdio_server for proper async I/O handling,
    ensuring compatibility with Claude Desktop.

    Args:
        server: The MCP server instance to run
    """
    logger.info("Starting stdio transport (using MCP SDK)")

    # Set up client context for this stdio session
    ctx = create_stdio_context()
    set_client_context(ctx)
    logger.info(f"Client context: session_id={ctx.session_id}, transport=stdio")

    try:
        # Initialize server
        await server.initialize()

        # Use the official MCP SDK's stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.debug("stdio transport connected")

            async for session_message in read_stream:
                # Handle exceptions from the read stream
                if isinstance(session_message, Exception):
                    logger.error(f"Error from read stream: {session_message}")
                    continue

                # Extract the JSON-RPC message
                message = session_message.message

                # Convert to dict for our handler
                if isinstance(message, JSONRPCRequest):
                    method = message.method
                    params = message.params.model_dump() if message.params else {}
                    request_id = message.id if hasattr(message, 'id') else None

                    logger.debug(f"Received request: {method} (id={request_id})")

                    message_dict = {
                        "jsonrpc": "2.0",
                        "method": method,
                        "params": params,
                    }
                    if request_id is not None:
                        message_dict["id"] = request_id
                else:
                    # Handle other message types (notifications, etc.)
                    message_dict = message.model_dump()
                    logger.debug(f"Received message: {message_dict.get('method', 'unknown')}")

                # Handle the message with our server
                response = await server.handle_message(message_dict)

                # Send response (unless it's a notification - None response)
                if response is not None:
                    # Convert response dict to JSONRPCMessage
                    if "error" in response:
                        rpc_response = JSONRPCMessage(
                            JSONRPCError(
                                jsonrpc="2.0",
                                id=response.get("id"),
                                error=response["error"],
                            )
                        )
                    else:
                        rpc_response = JSONRPCMessage(
                            JSONRPCResponse(
                                jsonrpc="2.0",
                                id=response.get("id"),
                                result=response.get("result", {}),
                            )
                        )

                    await write_stream.send(SessionMessage(rpc_response.root))
                    logger.debug(f"Sent response for id={response.get('id')}")

    except anyio.ClosedResourceError:
        logger.info("Client disconnected")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down")
    except Exception as e:
        logger.exception(f"Error in stdio server: {e}")
    finally:
        await server.shutdown()
        logger.info("stdio transport stopped")
