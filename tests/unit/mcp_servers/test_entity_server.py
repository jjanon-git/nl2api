"""
Unit Tests for Entity Resolution MCP Server

Tests the MCP server components:
- Configuration
- Tool handlers
- Resource handlers
- Server protocol handling
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_servers.entity_resolution.config import EntityServerConfig
from src.mcp_servers.entity_resolution.resources import (
    EXCHANGE_RIC_SUFFIXES,
    RESOURCE_DEFINITIONS,
    ResourceHandlers,
)
from src.mcp_servers.entity_resolution.tools import (
    TOOL_DEFINITIONS,
    BatchResolveResult,
    ExtractAndResolveResult,
    ResolveEntityResult,
    ToolHandlers,
)
from src.nl2api.resolution.protocols import ResolvedEntity

# =============================================================================
# Configuration Tests
# =============================================================================


class TestEntityServerConfig:
    """Tests for EntityServerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EntityServerConfig()

        assert config.server_name == "entity-resolution"
        assert config.transport == "sse"
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.redis_enabled is True
        assert config.circuit_failure_threshold == 5
        assert config.timeout_seconds == 5.0

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration from environment variables."""
        monkeypatch.setenv("ENTITY_MCP_SERVER_NAME", "test-server")
        monkeypatch.setenv("ENTITY_MCP_TRANSPORT", "stdio")
        monkeypatch.setenv("ENTITY_MCP_PORT", "9090")
        monkeypatch.setenv("ENTITY_MCP_REDIS_ENABLED", "false")

        config = EntityServerConfig()

        assert config.server_name == "test-server"
        assert config.transport == "stdio"
        assert config.port == 9090
        assert config.redis_enabled is False

    def test_config_validation(self) -> None:
        """Test configuration with custom values."""
        config = EntityServerConfig(
            server_name="custom-server",
            transport="stdio",
            postgres_url="postgresql://test:test@db:5432/test",
            redis_enabled=False,
            timeout_seconds=10.0,
        )

        assert config.server_name == "custom-server"
        assert config.transport == "stdio"
        assert config.postgres_url == "postgresql://test:test@db:5432/test"
        assert config.redis_enabled is False
        assert config.timeout_seconds == 10.0


# =============================================================================
# Tool Definition Tests
# =============================================================================


class TestToolDefinitions:
    """Tests for MCP tool definitions."""

    def test_tool_definitions_exist(self) -> None:
        """Test that all tool definitions are present."""
        tool_names = {t["name"] for t in TOOL_DEFINITIONS}

        assert "resolve_entity" in tool_names
        assert "resolve_entities_batch" in tool_names
        assert "extract_and_resolve" in tool_names

    def test_resolve_entity_definition(self) -> None:
        """Test resolve_entity tool definition."""
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "resolve_entity")

        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"
        assert "entity" in tool["inputSchema"]["properties"]
        assert "entity" in tool["inputSchema"]["required"]

    def test_resolve_entities_batch_definition(self) -> None:
        """Test resolve_entities_batch tool definition."""
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "resolve_entities_batch")

        assert "entities" in tool["inputSchema"]["properties"]
        assert tool["inputSchema"]["properties"]["entities"]["type"] == "array"

    def test_extract_and_resolve_definition(self) -> None:
        """Test extract_and_resolve tool definition."""
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "extract_and_resolve")

        assert "query" in tool["inputSchema"]["properties"]
        assert "query" in tool["inputSchema"]["required"]


# =============================================================================
# Tool Handler Tests
# =============================================================================


class TestToolHandlers:
    """Tests for MCP tool handlers."""

    @pytest.fixture
    def mock_resolver(self) -> MagicMock:
        """Create a mock resolver."""
        resolver = MagicMock()
        resolver.resolve_single = AsyncMock()
        resolver.resolve_batch = AsyncMock()
        resolver.resolve = AsyncMock()
        return resolver

    @pytest.fixture
    def handlers(self, mock_resolver: MagicMock) -> ToolHandlers:
        """Create tool handlers with mock resolver."""
        return ToolHandlers(mock_resolver)

    @pytest.mark.asyncio
    async def test_resolve_entity_found(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test resolve_entity when entity is found."""
        mock_resolver.resolve_single.return_value = ResolvedEntity(
            original="Apple",
            identifier="AAPL.O",
            entity_type="company",
            confidence=0.98,
            alternatives=("AAPL.OQ",),
            metadata={"exchange": "NASDAQ"},
        )

        result = await handlers.handle_tool_call(
            "resolve_entity",
            {"entity": "Apple"},
        )

        assert result["found"] is True
        assert result["original"] == "Apple"
        assert result["identifier"] == "AAPL.O"
        assert result["confidence"] == 0.98
        assert result["entity_type"] == "company"
        assert "AAPL.OQ" in result["alternatives"]

    @pytest.mark.asyncio
    async def test_resolve_entity_not_found(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test resolve_entity when entity is not found."""
        mock_resolver.resolve_single.return_value = None

        result = await handlers.handle_tool_call(
            "resolve_entity",
            {"entity": "UnknownCompany"},
        )

        assert result["found"] is False
        assert result["original"] == "UnknownCompany"
        assert result["identifier"] is None

    @pytest.mark.asyncio
    async def test_resolve_entity_with_type_hint(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test resolve_entity with entity type hint."""
        mock_resolver.resolve_single.return_value = ResolvedEntity(
            original="AAPL",
            identifier="AAPL.O",
            entity_type="ticker",
            confidence=0.99,
        )

        result = await handlers.handle_tool_call(
            "resolve_entity",
            {"entity": "AAPL", "entity_type": "ticker"},
        )

        mock_resolver.resolve_single.assert_called_with("AAPL", "ticker")
        assert result["found"] is True
        assert result["entity_type"] == "ticker"

    @pytest.mark.asyncio
    async def test_resolve_entities_batch(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test resolve_entities_batch."""
        mock_resolver.resolve_batch.return_value = [
            ResolvedEntity(
                original="Apple",
                identifier="AAPL.O",
                entity_type="company",
                confidence=0.98,
            ),
            ResolvedEntity(
                original="Microsoft",
                identifier="MSFT.O",
                entity_type="company",
                confidence=0.97,
            ),
        ]

        result = await handlers.handle_tool_call(
            "resolve_entities_batch",
            {"entities": ["Apple", "Microsoft", "UnknownCorp"]},
        )

        assert result["total_requested"] == 3
        assert result["total_resolved"] == 2
        assert len(result["results"]) == 3  # Includes unresolved

    @pytest.mark.asyncio
    async def test_extract_and_resolve(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test extract_and_resolve."""
        mock_resolver.resolve.return_value = {
            "Apple": "AAPL.O",
            "Microsoft": "MSFT.O",
        }

        result = await handlers.handle_tool_call(
            "extract_and_resolve",
            {"query": "Compare Apple and Microsoft revenue"},
        )

        assert "Apple" in result["extracted_entities"]
        assert "Microsoft" in result["extracted_entities"]
        assert result["resolved"]["Apple"] == "AAPL.O"
        assert result["resolved"]["Microsoft"] == "MSFT.O"

    @pytest.mark.asyncio
    async def test_unknown_tool(self, handlers: ToolHandlers) -> None:
        """Test handling unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await handlers.handle_tool_call("unknown_tool", {})


# =============================================================================
# Resource Tests
# =============================================================================


class TestResourceDefinitions:
    """Tests for MCP resource definitions."""

    def test_resource_definitions_exist(self) -> None:
        """Test that all resource definitions are present."""
        resource_uris = {r["uri"] for r in RESOURCE_DEFINITIONS}

        assert "entity://stats" in resource_uris
        assert "entity://health" in resource_uris
        assert "entity://exchanges" in resource_uris

    def test_exchange_data(self) -> None:
        """Test exchange reference data."""
        assert "NASDAQ" in EXCHANGE_RIC_SUFFIXES
        assert EXCHANGE_RIC_SUFFIXES["NASDAQ"]["suffix"] == ".O"
        assert "NYSE" in EXCHANGE_RIC_SUFFIXES
        assert EXCHANGE_RIC_SUFFIXES["NYSE"]["suffix"] == ".N"


class TestResourceHandlers:
    """Tests for MCP resource handlers."""

    @pytest.fixture
    def mock_resolver(self) -> MagicMock:
        """Create a mock resolver."""
        resolver = MagicMock()
        resolver.circuit_breaker_stats = {
            "state": "closed",
            "failure_count": 0,
            "total_calls": 100,
        }
        resolver._cache = {"apple": MagicMock()}
        return resolver

    @pytest.fixture
    def handlers(self, mock_resolver: MagicMock) -> ResourceHandlers:
        """Create resource handlers with mock resolver."""
        return ResourceHandlers(
            resolver=mock_resolver,
            db_pool=None,
            redis_cache=None,
        )

    @pytest.mark.asyncio
    async def test_get_exchanges(self, handlers: ResourceHandlers) -> None:
        """Test exchanges resource."""
        result = await handlers.handle_resource_read("entity://exchanges")

        assert "exchanges" in result
        assert "NASDAQ" in result["exchanges"]
        assert result["total_exchanges"] > 0

    @pytest.mark.asyncio
    async def test_get_health_no_db(self, handlers: ResourceHandlers) -> None:
        """Test health resource without database."""
        result = await handlers.handle_resource_read("entity://health")

        assert result["status"] in ("healthy", "degraded", "unhealthy")
        assert "checks" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_get_stats(self, handlers: ResourceHandlers) -> None:
        """Test stats resource."""
        result = await handlers.handle_resource_read("entity://stats")

        assert "database" in result
        assert "cache" in result
        assert "circuit_breaker" in result
        assert result["circuit_breaker"]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_unknown_resource(self, handlers: ResourceHandlers) -> None:
        """Test handling unknown resource URI."""
        with pytest.raises(ValueError, match="Unknown resource URI"):
            await handlers.handle_resource_read("entity://unknown")


# =============================================================================
# Response Model Tests
# =============================================================================


class TestResponseModels:
    """Tests for response data models."""

    def test_resolve_entity_result_to_dict(self) -> None:
        """Test ResolveEntityResult serialization."""
        result = ResolveEntityResult(
            found=True,
            original="Apple",
            identifier="AAPL.O",
            confidence=0.98,
            entity_type="company",
            alternatives=("AAPL.OQ",),
        )

        data = result.to_dict()

        assert data["found"] is True
        assert data["original"] == "Apple"
        assert data["identifier"] == "AAPL.O"
        assert data["alternatives"] == ["AAPL.OQ"]  # Converted to list

    def test_batch_resolve_result_to_dict(self) -> None:
        """Test BatchResolveResult serialization."""
        result = BatchResolveResult(
            results=(
                ResolveEntityResult(found=True, original="Apple", identifier="AAPL.O"),
                ResolveEntityResult(found=False, original="Unknown"),
            ),
            total_requested=2,
            total_resolved=1,
        )

        data = result.to_dict()

        assert data["total_requested"] == 2
        assert data["total_resolved"] == 1
        assert len(data["results"]) == 2

    def test_extract_and_resolve_result_to_dict(self) -> None:
        """Test ExtractAndResolveResult serialization."""
        result = ExtractAndResolveResult(
            extracted_entities=("Apple", "Microsoft"),
            resolved={"Apple": "AAPL.O", "Microsoft": "MSFT.O"},
            unresolved=(),
        )

        data = result.to_dict()

        assert data["extracted_entities"] == ["Apple", "Microsoft"]
        assert data["resolved"]["Apple"] == "AAPL.O"


# =============================================================================
# Server Protocol Tests
# =============================================================================


class TestServerProtocol:
    """Tests for MCP server protocol handling."""

    @pytest.fixture
    def server(self) -> MagicMock:
        """Create a mock server."""
        from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

        # Create server with minimal config
        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
        )
        return EntityResolutionMCPServer(config)

    def test_server_info(self, server: MagicMock) -> None:
        """Test server info property."""
        info = server.server_info

        assert info["name"] == "entity-resolution"
        assert "version" in info
        assert "protocolVersion" in info
        assert "capabilities" in info
        assert "tools" in info["capabilities"]
        assert "resources" in info["capabilities"]

    @pytest.mark.asyncio
    async def test_list_tools(self, server: MagicMock) -> None:
        """Test tools/list method."""
        tools = await server.list_tools()

        assert len(tools) == 3
        tool_names = {t["name"] for t in tools}
        assert "resolve_entity" in tool_names

    @pytest.mark.asyncio
    async def test_list_resources(self, server: MagicMock) -> None:
        """Test resources/list method."""
        resources = await server.list_resources()

        assert len(resources) == 3
        resource_uris = {r["uri"] for r in resources}
        assert "entity://health" in resource_uris

    @pytest.mark.asyncio
    async def test_handle_ping(self, server: MagicMock) -> None:
        """Test ping message handling."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
            }
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server: MagicMock) -> None:
        """Test tools/list message handling."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            }
        )

        assert response["jsonrpc"] == "2.0"
        assert "tools" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server: MagicMock) -> None:
        """Test unknown method handling."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "unknown/method",
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

    @pytest.mark.asyncio
    async def test_handle_resources_list(self, server: MagicMock) -> None:
        """Test resources/list message handling."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "resources/list",
            }
        )

        assert response["jsonrpc"] == "2.0"
        assert "resources" in response["result"]


# =============================================================================
# Integration-Like Tests (Mocked Dependencies)
# =============================================================================


class TestServerIntegration:
    """Integration-like tests with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_resolve_flow(self) -> None:
        """Test full resolve entity flow with mocked resolver."""
        from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

        # Create server
        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
        )
        server = EntityResolutionMCPServer(config)

        # Mock the resolver initialization
        with patch.object(server, "_db_pool", None):
            with patch.object(server, "_redis_cache", None):
                mock_resolver = MagicMock()
                mock_resolver.resolve_single = AsyncMock(
                    return_value=ResolvedEntity(
                        original="Apple",
                        identifier="AAPL.O",
                        entity_type="company",
                        confidence=0.98,
                    )
                )
                mock_resolver.circuit_breaker_stats = {"state": "closed"}
                mock_resolver._cache = {}

                server._resolver = mock_resolver
                server._tool_handlers = ToolHandlers(mock_resolver)
                server._resource_handlers = ResourceHandlers(mock_resolver)
                server._initialized = True

                # Test tool call
                result = await server.call_tool(
                    "resolve_entity",
                    {"entity": "Apple"},
                )

                assert result["found"] is True
                assert result["identifier"] == "AAPL.O"


# =============================================================================
# Client Context Tests
# =============================================================================


class TestClientContext:
    """Tests for client context and observability."""

    def test_client_context_default_values(self) -> None:
        """Test ClientContext has sensible defaults."""
        from src.mcp_servers.entity_resolution.context import ClientContext

        ctx = ClientContext()

        assert ctx.session_id is not None
        assert len(ctx.session_id) == 36  # UUID format
        assert ctx.client_id is None
        assert ctx.client_name is None
        assert ctx.transport == "unknown"
        assert ctx.request_id is None

    def test_client_context_with_request(self) -> None:
        """Test creating a context with a specific request ID."""
        from src.mcp_servers.entity_resolution.context import ClientContext

        ctx = ClientContext(
            session_id="test-session",
            client_name="test-client",
            transport="sse",
        )

        req_ctx = ctx.with_request("req-123")

        # Should preserve session but add request ID
        assert req_ctx.session_id == "test-session"
        assert req_ctx.client_name == "test-client"
        assert req_ctx.transport == "sse"
        assert req_ctx.request_id == "req-123"

        # Original should be unchanged
        assert ctx.request_id is None

    def test_client_context_to_span_attributes(self) -> None:
        """Test converting context to OTEL span attributes."""
        from src.mcp_servers.entity_resolution.context import ClientContext

        ctx = ClientContext(
            session_id="sess-456",
            client_id="client-789",
            client_name="test-client",
            transport="sse",
            request_id="req-123",
        )

        attrs = ctx.to_span_attributes()

        assert attrs["client.session_id"] == "sess-456"
        assert attrs["client.id"] == "client-789"
        assert attrs["client.name"] == "test-client"
        assert attrs["client.transport"] == "sse"
        assert attrs["request.correlation_id"] == "req-123"

    def test_client_context_to_span_attributes_minimal(self) -> None:
        """Test span attributes with minimal context (no optional fields)."""
        from src.mcp_servers.entity_resolution.context import ClientContext

        ctx = ClientContext(session_id="sess-123", transport="stdio")

        attrs = ctx.to_span_attributes()

        assert attrs["client.session_id"] == "sess-123"
        assert attrs["client.transport"] == "stdio"
        assert "client.id" not in attrs
        assert "client.name" not in attrs
        assert "request.correlation_id" not in attrs

    def test_create_sse_context(self) -> None:
        """Test creating SSE context from headers."""
        from src.mcp_servers.entity_resolution.context import create_sse_context

        ctx = create_sse_context(
            client_id="my-client-id",
            client_name="my-client",
            user_agent=None,
        )

        assert ctx.transport == "sse"
        assert ctx.client_id == "my-client-id"
        assert ctx.client_name == "my-client"
        assert ctx.session_id is not None

    def test_create_sse_context_from_user_agent(self) -> None:
        """Test inferring client name from User-Agent."""
        from src.mcp_servers.entity_resolution.context import create_sse_context

        ctx = create_sse_context(
            client_id=None,
            client_name=None,
            user_agent="Claude-Desktop/1.0",
        )

        assert ctx.client_name == "claude-desktop"

    def test_create_stdio_context(self) -> None:
        """Test creating stdio context."""
        from src.mcp_servers.entity_resolution.context import create_stdio_context

        ctx = create_stdio_context()

        assert ctx.transport == "stdio"
        assert ctx.client_name == "stdio-client"
        assert ctx.session_id is not None

    def test_context_var_get_set(self) -> None:
        """Test getting and setting context via context vars."""
        from src.mcp_servers.entity_resolution.context import (
            ClientContext,
            clear_client_context,
            get_client_context,
            set_client_context,
        )

        # Initially None
        clear_client_context()
        assert get_client_context() is None

        # Set context
        ctx = ClientContext(session_id="test", transport="test")
        set_client_context(ctx)

        # Should be retrievable
        retrieved = get_client_context()
        assert retrieved is not None
        assert retrieved.session_id == "test"

        # Clear
        clear_client_context()
        assert get_client_context() is None


# =============================================================================
# Server Initialization and Lifecycle Tests
# =============================================================================


class TestServerLifecycle:
    """Tests for server initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_server_not_initialized_call_tool_raises(self) -> None:
        """Test that call_tool raises when server not initialized."""
        from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
        )
        server = EntityResolutionMCPServer(config)

        with pytest.raises(RuntimeError, match="Server not initialized"):
            await server.call_tool("resolve_entity", {"entity": "Apple"})

    @pytest.mark.asyncio
    async def test_server_not_initialized_read_resource_raises(self) -> None:
        """Test that read_resource raises when server not initialized."""
        from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
        )
        server = EntityResolutionMCPServer(config)

        with pytest.raises(RuntimeError, match="Server not initialized"):
            await server.read_resource("entity://health")

    @pytest.mark.asyncio
    async def test_server_config_property(self) -> None:
        """Test server config property."""
        from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

        config = EntityServerConfig(
            server_name="test-server",
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
        )
        server = EntityResolutionMCPServer(config)

        assert server.config.server_name == "test-server"

    @pytest.mark.asyncio
    async def test_server_default_config(self) -> None:
        """Test server with default config from environment."""
        from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

        # Test that server can be created without explicit config
        server = EntityResolutionMCPServer()
        assert server.config.server_name == "entity-resolution"


# =============================================================================
# JSON-RPC Message Handling Tests
# =============================================================================


class TestJsonRpcMessageHandling:
    """Tests for JSON-RPC message handling."""

    @pytest.fixture
    def server(self) -> MagicMock:
        """Create a mock server."""
        from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
        )
        return EntityResolutionMCPServer(config)

    @pytest.mark.asyncio
    async def test_handle_initialize_message(self, server: MagicMock) -> None:
        """Test initialize message triggers initialization."""
        # Mock the actual database connection
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_pool:
            mock_pool.return_value = MagicMock()
            mock_pool.return_value.close = AsyncMock()

            response = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                }
            )

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 1
            assert "result" in response
            assert response["result"]["name"] == "entity-resolution"

    @pytest.mark.asyncio
    async def test_handle_initialized_message(self, server: MagicMock) -> None:
        """Test initialized notification (no id, no response per JSON-RPC spec)."""
        # 'initialized' is a notification in MCP - it has no id and expects no response
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "method": "initialized",
                # Note: no 'id' field - this is a notification
            }
        )

        # Notifications return None (no response should be sent)
        assert response is None

    @pytest.mark.asyncio
    async def test_handle_shutdown_message(self, server: MagicMock) -> None:
        """Test shutdown message triggers cleanup."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "shutdown",
            }
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"] == {}

    @pytest.mark.asyncio
    async def test_handle_tools_call_uninitialized(self, server: MagicMock) -> None:
        """Test tools/call returns error when not initialized."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "resolve_entity",
                    "arguments": {"entity": "Apple"},
                },
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32603  # Internal error

    @pytest.mark.asyncio
    async def test_handle_resources_read_uninitialized(self, server: MagicMock) -> None:
        """Test resources/read returns error when not initialized."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "resources/read",
                "params": {"uri": "entity://health"},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32603  # Internal error

    @pytest.mark.asyncio
    async def test_handle_message_with_none_id(self, server: MagicMock) -> None:
        """Test handling message with None id (notification)."""
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": None,
                "method": "ping",
            }
        )

        assert response["id"] is None
        assert "result" in response


# =============================================================================
# Transport Helper Tests
# =============================================================================


# =============================================================================
# SSE Transport Helper Tests
# =============================================================================
# Note: stdio transport tests removed - now using official MCP SDK's stdio_server
# which handles all I/O internally. The SDK is well-tested upstream.


class TestSSETransportHelpers:
    """Tests for SSE transport helper functions."""

    def test_get_server_raises_when_not_initialized(self) -> None:
        """Test get_server raises when no server is set."""
        from src.mcp_servers.entity_resolution.transports import sse

        # Reset global server
        sse._server = None

        with pytest.raises(RuntimeError, match="Server not initialized"):
            sse.get_server()

    def test_create_app_returns_fastapi_app(self) -> None:
        """Test create_app returns a FastAPI application."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
        )
        app = create_app(config)

        # Check it's a FastAPI app
        assert hasattr(app, "routes")
        assert hasattr(app, "openapi")

        # Check expected routes exist
        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths
        assert "/mcp" in route_paths
        assert "/sse" in route_paths
        assert "/api/resolve" in route_paths


# =============================================================================
# Resource Handler Edge Cases
# =============================================================================


class TestResourceHandlerEdgeCases:
    """Tests for resource handler edge cases."""

    @pytest.mark.asyncio
    async def test_get_stats_with_db_pool(self) -> None:
        """Test stats resource with database pool fetches entity counts."""
        mock_resolver = MagicMock()
        mock_resolver.circuit_breaker_stats = {"state": "closed", "failure_count": 0}
        mock_resolver._cache = {"apple": MagicMock()}  # Has one cached entry

        # Mock database pool with async context manager
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "total_entities": 100,
                "total_aliases": 50,
                "entities_with_ric": 80,
            }
        )

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        handlers = ResourceHandlers(
            resolver=mock_resolver,
            db_pool=mock_pool,
            redis_cache=None,
        )

        result = await handlers.handle_resource_read("entity://stats")

        # Stats uses StatsResponse.to_dict() structure
        assert result["database"]["total_entities"] == 100
        assert result["database"]["total_aliases"] == 50
        assert result["database"]["entities_with_ric"] == 80
        assert result["cache"]["l1_cache_size"] == 1  # From resolver._cache
        assert result["circuit_breaker"]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_get_stats_with_redis_cache(self) -> None:
        """Test stats resource with Redis cache shows L2 status."""
        mock_resolver = MagicMock()
        mock_resolver.circuit_breaker_stats = {"state": "closed"}
        mock_resolver._cache = {}

        mock_redis = MagicMock()
        mock_redis.is_connected = True

        handlers = ResourceHandlers(
            resolver=mock_resolver,
            db_pool=None,
            redis_cache=mock_redis,
        )

        result = await handlers.handle_resource_read("entity://stats")

        # Stats uses l2_cache_connected, not redis_enabled
        assert result["cache"]["l2_cache_connected"] is True

    @pytest.mark.asyncio
    async def test_health_degraded_with_failed_db(self) -> None:
        """Test health returns degraded when DB check fails."""
        mock_resolver = MagicMock()
        mock_resolver.circuit_breaker_stats = {"state": "closed"}
        mock_resolver._cache = {}

        # Mock pool that raises on acquire
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        handlers = ResourceHandlers(
            resolver=mock_resolver,
            db_pool=mock_pool,
            redis_cache=None,
        )

        result = await handlers.handle_resource_read("entity://health")

        # Should be degraded or unhealthy
        assert result["status"] in ("degraded", "unhealthy")


# =============================================================================
# Tool Handler Edge Cases
# =============================================================================


class TestToolHandlerEdgeCases:
    """Tests for tool handler edge cases."""

    @pytest.fixture
    def mock_resolver(self) -> MagicMock:
        """Create a mock resolver."""
        resolver = MagicMock()
        resolver.resolve_single = AsyncMock()
        resolver.resolve_batch = AsyncMock()
        resolver.resolve = AsyncMock()
        return resolver

    @pytest.fixture
    def handlers(self, mock_resolver: MagicMock) -> ToolHandlers:
        """Create tool handlers with mock resolver."""
        return ToolHandlers(mock_resolver)

    @pytest.mark.asyncio
    async def test_resolve_entity_with_metadata(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test resolve_entity includes metadata in response."""
        mock_resolver.resolve_single.return_value = ResolvedEntity(
            original="Apple",
            identifier="AAPL.O",
            entity_type="company",
            confidence=0.98,
            metadata={"exchange": "NASDAQ", "primary_name": "Apple Inc."},
        )

        result = await handlers.handle_tool_call(
            "resolve_entity",
            {"entity": "Apple"},
        )

        assert result["metadata"]["exchange"] == "NASDAQ"
        assert result["metadata"]["primary_name"] == "Apple Inc."

    @pytest.mark.asyncio
    async def test_batch_resolve_empty_input(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test batch resolve with empty input."""
        mock_resolver.resolve_batch.return_value = []

        result = await handlers.handle_tool_call(
            "resolve_entities_batch",
            {"entities": []},
        )

        assert result["total_requested"] == 0
        assert result["total_resolved"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_extract_and_resolve_no_entities(
        self, handlers: ToolHandlers, mock_resolver: MagicMock
    ) -> None:
        """Test extract_and_resolve when no entities found."""
        mock_resolver.resolve.return_value = {}

        result = await handlers.handle_tool_call(
            "extract_and_resolve",
            {"query": "What is the weather today?"},
        )

        assert result["extracted_entities"] == []
        assert result["resolved"] == {}
        assert result["unresolved"] == []


# =============================================================================
# Client ID Enforcement Tests
# =============================================================================


class TestClientIdEnforcement:
    """Tests for mandatory client_id enforcement in SSE transport."""

    def test_config_require_client_id_default_true(self) -> None:
        """Test require_client_id defaults to True."""
        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
        )
        assert config.require_client_id is True

    def test_config_require_client_id_can_be_disabled(self) -> None:
        """Test require_client_id can be set to False."""
        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            require_client_id=False,
        )
        assert config.require_client_id is False

    def test_exempt_paths_allow_requests_without_client_id(self) -> None:
        """Test exempt paths (health, root) don't require client_id."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # Health check should work without client_id
            response = client.get("/health")
            # May fail due to no DB, but should not be 401
            assert response.status_code != 401

            # Root should work without client_id
            response = client.get("/")
            assert response.status_code != 401

    def test_api_endpoints_require_client_id(self) -> None:
        """Test API endpoints return 401 without X-Client-ID header."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # API resolve without client_id should fail
            response = client.post(
                "/api/resolve",
                json={"entity": "Apple"},
            )
            assert response.status_code == 401
            assert response.json()["error"] == "missing_client_id"

    def test_api_endpoints_work_with_client_id(self) -> None:
        """Test API endpoints work with X-Client-ID header."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # API resolve with client_id should not get 401
            response = client.post(
                "/api/resolve",
                json={"entity": "Apple"},
                headers={"X-Client-ID": "test-client-123"},
            )
            # May fail due to no DB, but should not be 401
            assert response.status_code != 401

    def test_api_endpoints_work_when_enforcement_disabled(self) -> None:
        """Test API endpoints work without client_id when enforcement disabled."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=False,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # API resolve without client_id should not get 401
            response = client.post(
                "/api/resolve",
                json={"entity": "Apple"},
            )
            # May fail due to no DB, but should not be 401
            assert response.status_code != 401

    def test_empty_client_id_rejected(self) -> None:
        """Test empty string X-Client-ID is rejected."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/resolve",
                json={"entity": "Apple"},
                headers={"X-Client-ID": ""},
            )
            assert response.status_code == 401
            assert response.json()["error"] == "missing_client_id"

    def test_whitespace_client_id_rejected(self) -> None:
        """Test whitespace-only X-Client-ID is rejected."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/resolve",
                json={"entity": "Apple"},
                headers={"X-Client-ID": "   "},
            )
            assert response.status_code == 401
            assert response.json()["error"] == "missing_client_id"

    def test_mcp_endpoint_requires_client_id(self) -> None:
        """Test /mcp endpoint requires X-Client-ID."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # Without client_id
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            assert response.status_code == 401

            # With client_id
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"X-Client-ID": "test-client"},
            )
            assert response.status_code != 401

    def test_sse_endpoint_requires_client_id(self) -> None:
        """Test /sse endpoint requires X-Client-ID (rejection case only).

        Note: We only test the rejection case (401) because the success case
        starts an infinite SSE stream that hangs the test. The middleware check
        happens before the streaming starts, so testing 401 is sufficient to
        verify the enforcement works.
        """
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # Without client_id - should get 401
            response = client.get("/sse")
            assert response.status_code == 401
            assert response.json()["error"] == "missing_client_id"

    def test_batch_endpoint_requires_client_id(self) -> None:
        """Test /api/resolve/batch endpoint requires X-Client-ID."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # Without client_id
            response = client.post(
                "/api/resolve/batch",
                json={"entities": ["Apple", "Microsoft"]},
            )
            assert response.status_code == 401

            # With client_id
            response = client.post(
                "/api/resolve/batch",
                json={"entities": ["Apple", "Microsoft"]},
                headers={"X-Client-ID": "test-client"},
            )
            assert response.status_code != 401

    def test_extract_endpoint_requires_client_id(self) -> None:
        """Test /api/extract endpoint requires X-Client-ID."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as client:
            # Without client_id
            response = client.post(
                "/api/extract",
                json={"query": "Compare Apple and Microsoft"},
            )
            assert response.status_code == 401

            # With client_id
            response = client.post(
                "/api/extract",
                json={"query": "Compare Apple and Microsoft"},
                headers={"X-Client-ID": "test-client"},
            )
            assert response.status_code != 401

    def test_config_from_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test require_client_id can be set via environment variable."""
        monkeypatch.setenv("ENTITY_MCP_REQUIRE_CLIENT_ID", "false")

        # Create fresh config to pick up env var
        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
        )
        assert config.require_client_id is False

        # Also test true
        monkeypatch.setenv("ENTITY_MCP_REQUIRE_CLIENT_ID", "true")
        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
        )
        assert config.require_client_id is True

    def test_client_context_propagation(self) -> None:
        """Test client_id is properly propagated to context."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.context import get_client_context
        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        # Add a test endpoint that returns the context
        from fastapi.responses import JSONResponse

        @app.get("/test/context")
        async def get_context() -> JSONResponse:
            ctx = get_client_context()
            if ctx:
                return JSONResponse(
                    content={
                        "client_id": ctx.client_id,
                        "client_name": ctx.client_name,
                        "transport": ctx.transport,
                        "session_id": ctx.session_id,
                    }
                )
            return JSONResponse(content={"error": "no context"}, status_code=500)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get(
                "/test/context",
                headers={
                    "X-Client-ID": "my-app-123",
                    "X-Client-Name": "My Application",
                },
            )
            # /test/context is not in exempt paths, so needs client_id
            # But we're providing it, so should work
            assert response.status_code == 200
            data = response.json()
            assert data["client_id"] == "my-app-123"
            assert data["client_name"] == "My Application"
            assert data["transport"] == "sse"
            assert data["session_id"] is not None

    def test_long_client_id_accepted(self) -> None:
        """Test very long client_id values are accepted."""
        pytest.importorskip("fastapi", reason="FastAPI not installed")
        from fastapi.testclient import TestClient

        from src.mcp_servers.entity_resolution.transports.sse import create_app

        config = EntityServerConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            redis_enabled=False,
            require_client_id=True,
        )
        app = create_app(config)

        long_client_id = "a" * 1000  # 1000 character client_id

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/resolve",
                json={"entity": "Apple"},
                headers={"X-Client-ID": long_client_id},
            )
            # Should not be rejected for being too long
            assert response.status_code != 401
