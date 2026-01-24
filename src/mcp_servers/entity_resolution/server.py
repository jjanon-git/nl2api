"""
Entity Resolution MCP Server

Main server class that implements the MCP protocol for entity resolution.
Supports stdio and HTTP/SSE transports.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.evalkit.common.telemetry import get_tracer
from src.mcp_servers.entity_resolution.config import EntityServerConfig
from src.mcp_servers.entity_resolution.context import (
    ClientContext,
    get_client_context,
)
from src.mcp_servers.entity_resolution.resources import (
    RESOURCE_DEFINITIONS,
    ResourceHandlers,
)
from src.mcp_servers.entity_resolution.tools import TOOL_DEFINITIONS, ToolHandlers
from src.nl2api.resolution.resolver import ExternalEntityResolver

if TYPE_CHECKING:
    import asyncpg

    from src.evalkit.common.cache import RedisCache

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class EntityResolutionMCPServer:
    """
    MCP Server for Entity Resolution.

    Exposes entity resolution capabilities via MCP protocol:
    - Tools: resolve_entity, resolve_entities_batch, extract_and_resolve
    - Resources: stats, health, exchanges

    Supports both stdio (Claude Desktop) and SSE (HTTP) transports.
    """

    def __init__(
        self,
        config: EntityServerConfig | None = None,
    ):
        """
        Initialize the MCP server.

        Args:
            config: Server configuration. If None, loads from environment.
        """
        self._config = config or EntityServerConfig()
        self._db_pool: asyncpg.Pool | None = None
        self._redis_cache: RedisCache | None = None
        self._resolver: ExternalEntityResolver | None = None
        self._tool_handlers: ToolHandlers | None = None
        self._resource_handlers: ResourceHandlers | None = None
        self._initialized = False

    @property
    def config(self) -> EntityServerConfig:
        """Get server configuration."""
        return self._config

    @property
    def server_info(self) -> dict[str, Any]:
        """Get MCP server information."""
        return {
            "name": self._config.server_name,
            "version": self._config.server_version,
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
        }

    async def initialize(self) -> None:
        """
        Initialize server components.

        Sets up database pool, cache, and resolver.
        Must be called before handling requests.
        """
        if self._initialized:
            return

        with tracer.start_as_current_span("mcp.server.initialize") as span:
            span.set_attribute("server.name", self._config.server_name)
            self._add_client_context_to_span(span)

            # Initialize database pool
            try:
                import asyncpg

                self._db_pool = await asyncpg.create_pool(
                    self._config.postgres_url,
                    min_size=self._config.postgres_pool_min,
                    max_size=self._config.postgres_pool_max,
                )
                logger.info("Database pool initialized")
                span.set_attribute("database.connected", True)
            except Exception as e:
                logger.warning(f"Failed to initialize database pool: {e}")
                span.set_attribute("database.connected", False)
                span.set_attribute("database.error", str(e))

            # Initialize Redis cache
            if self._config.redis_enabled:
                try:
                    from src.evalkit.common.cache import RedisCache

                    self._redis_cache = RedisCache(
                        redis_url=self._config.redis_url,
                        default_ttl_seconds=self._config.redis_cache_ttl_seconds,
                        key_prefix="entity:",
                    )
                    await self._redis_cache.connect()
                    logger.info("Redis cache initialized")
                    span.set_attribute("cache.connected", True)
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis cache: {e}")
                    span.set_attribute("cache.connected", False)
                    span.set_attribute("cache.error", str(e))

            # Initialize resolver
            self._resolver = ExternalEntityResolver(
                api_endpoint=None,  # Use OpenFIGI directly
                api_key=self._config.openfigi_api_key,
                use_cache=True,
                timeout_seconds=self._config.timeout_seconds,
                circuit_failure_threshold=self._config.circuit_failure_threshold,
                circuit_recovery_seconds=self._config.circuit_recovery_seconds,
                retry_max_attempts=self._config.retry_max_attempts,
                redis_cache=self._redis_cache,
                redis_cache_ttl_seconds=self._config.redis_cache_ttl_seconds,
                db_pool=self._db_pool,
            )
            logger.info("Entity resolver initialized")

            # Initialize handlers
            self._tool_handlers = ToolHandlers(self._resolver)
            self._resource_handlers = ResourceHandlers(
                resolver=self._resolver,
                db_pool=self._db_pool,
                redis_cache=self._redis_cache,
            )

            self._initialized = True
            span.set_attribute("server.initialized", True)
            logger.info(f"MCP Server '{self._config.server_name}' initialized successfully")

    async def shutdown(self) -> None:
        """
        Shutdown server and release resources.
        """
        with tracer.start_as_current_span("mcp.server.shutdown") as span:
            span.set_attribute("server.name", self._config.server_name)
            self._add_client_context_to_span(span)

            if self._db_pool:
                await self._db_pool.close()
                logger.info("Database pool closed")

            if self._redis_cache:
                await self._redis_cache.close()
                logger.info("Redis cache closed")

            self._initialized = False
            logger.info(f"MCP Server '{self._config.server_name}' shut down")

    # =========================================================================
    # MCP Protocol Methods
    # =========================================================================

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools (MCP tools/list).

        Returns:
            List of tool definitions
        """
        return TOOL_DEFINITIONS

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call a tool (MCP tools/call).

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            RuntimeError: If server not initialized
            ValueError: If tool not found
        """
        if not self._initialized or not self._tool_handlers:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        with tracer.start_as_current_span("mcp.call_tool") as span:
            span.set_attribute("tool.name", name)
            self._add_client_context_to_span(span)

            result = await self._tool_handlers.handle_tool_call(name, arguments)

            span.set_attribute("tool.success", True)
            return result

    async def list_resources(self) -> list[dict[str, Any]]:
        """
        List available resources (MCP resources/list).

        Returns:
            List of resource definitions
        """
        return RESOURCE_DEFINITIONS

    async def read_resource(
        self,
        uri: str,
    ) -> dict[str, Any]:
        """
        Read a resource (MCP resources/read).

        Args:
            uri: Resource URI

        Returns:
            Resource content

        Raises:
            RuntimeError: If server not initialized
            ValueError: If resource not found
        """
        if not self._initialized or not self._resource_handlers:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        with tracer.start_as_current_span("mcp.read_resource") as span:
            span.set_attribute("resource.uri", uri)
            self._add_client_context_to_span(span)

            result = await self._resource_handlers.handle_resource_read(uri)

            span.set_attribute("resource.success", True)
            return result

    # =========================================================================
    # JSON-RPC Message Handling
    # =========================================================================

    def _add_client_context_to_span(
        self,
        span: Any,
        client_ctx: ClientContext | None = None,
    ) -> None:
        """Add client context attributes to a span."""
        ctx = client_ctx or get_client_context()
        if ctx:
            for key, value in ctx.to_span_attributes().items():
                span.set_attribute(key, value)

    async def handle_message(
        self,
        message: dict[str, Any],
        client_ctx: ClientContext | None = None,
    ) -> dict[str, Any] | None:
        """
        Handle an incoming JSON-RPC message.

        Args:
            message: JSON-RPC request message
            client_ctx: Client context for observability (optional, falls back to context var)

        Returns:
            JSON-RPC response message, or None for notifications
        """
        method = message.get("method", "")
        params = message.get("params", {})
        request_id = message.get("id")

        # JSON-RPC notifications have no id and expect no response
        is_notification = request_id is None

        # Create request-specific context if we have a base context
        ctx = client_ctx or get_client_context()
        if ctx and request_id:
            ctx = ctx.with_request(str(request_id))

        with tracer.start_as_current_span("mcp.handle_message") as span:
            span.set_attribute("jsonrpc.method", method)
            span.set_attribute("jsonrpc.id", str(request_id))
            span.set_attribute("jsonrpc.is_notification", is_notification)
            self._add_client_context_to_span(span, ctx)

            try:
                if method == "initialize":
                    await self.initialize()
                    result = self.server_info

                elif method == "initialized":
                    # Client acknowledgment - this is a notification, no response
                    logger.debug("Received 'initialized' notification from client")
                    return None

                elif method == "tools/list":
                    result = {"tools": await self.list_tools()}

                elif method == "tools/call":
                    tool_name = params.get("name", "")
                    arguments = params.get("arguments", {})
                    tool_result = await self.call_tool(tool_name, arguments)
                    result = {
                        "content": [
                            {
                                "type": "text",
                                "text": str(tool_result),
                            }
                        ],
                        "isError": False,
                    }

                elif method == "resources/list":
                    result = {"resources": await self.list_resources()}

                elif method == "resources/read":
                    uri = params.get("uri", "")
                    resource_content = await self.read_resource(uri)
                    result = {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": str(resource_content),
                            }
                        ],
                    }

                elif method == "ping":
                    result = {}

                elif method == "shutdown":
                    await self.shutdown()
                    result = {}

                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}",
                        },
                    }

                span.set_attribute("jsonrpc.success", True)
                return {"jsonrpc": "2.0", "id": request_id, "result": result}

            except ValueError as e:
                span.set_attribute("jsonrpc.error", str(e))
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": str(e),
                    },
                }
            except Exception as e:
                logger.exception(f"Error handling message: {e}")
                span.set_attribute("jsonrpc.error", str(e))
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {e}",
                    },
                }

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> EntityResolutionMCPServer:
        """Enter async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.shutdown()


async def create_server(
    config: EntityServerConfig | None = None,
) -> EntityResolutionMCPServer:
    """
    Create and initialize an MCP server instance.

    Args:
        config: Server configuration

    Returns:
        Initialized server instance
    """
    server = EntityResolutionMCPServer(config)
    await server.initialize()
    return server
