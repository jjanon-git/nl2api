"""
Tests for MCP Client

Tests the MCP client implementation including connection management,
tool discovery, and caching.
"""

import pytest
from src.nl2api.mcp.client import (
    MCPClient,
    MCPClientConfig,
    MCPConnectionError,
)
from src.nl2api.mcp.protocols import MCPCapabilities, MCPServer


class TestMCPClientConfig:
    """Tests for MCPClientConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MCPClientConfig()

        assert config.default_timeout_seconds == 30
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.connection_pool_size == 10
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 300

    def test_custom_config(self):
        """Test custom configuration."""
        config = MCPClientConfig(
            default_timeout_seconds=60,
            max_retries=5,
            enable_caching=False,
        )

        assert config.default_timeout_seconds == 60
        assert config.max_retries == 5
        assert config.enable_caching is False


class TestMCPClient:
    """Tests for MCPClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return MCPClient(MCPClientConfig(enable_caching=False))

    @pytest.fixture
    def test_server(self):
        """Create a test server config."""
        return MCPServer(
            uri="mcp://test.lseg.com",
            name="test-server",
            description="Test MCP server",
        )

    @pytest.mark.asyncio
    async def test_connect_to_server(self, client, test_server):
        """Test connecting to an MCP server."""
        result = await client.connect(test_server)

        assert result is True
        assert client.is_connected("mcp://test.lseg.com")
        assert "mcp://test.lseg.com" in client.connected_servers

    @pytest.mark.asyncio
    async def test_connect_same_server_twice(self, client, test_server):
        """Test that connecting twice returns True without error."""
        await client.connect(test_server)
        result = await client.connect(test_server)

        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_from_server(self, client, test_server):
        """Test disconnecting from a server."""
        await client.connect(test_server)
        assert client.is_connected("mcp://test.lseg.com")

        await client.disconnect("mcp://test.lseg.com")

        assert not client.is_connected("mcp://test.lseg.com")
        assert "mcp://test.lseg.com" not in client.connected_servers

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, client):
        """Test disconnecting from non-connected server is safe."""
        # Should not raise
        await client.disconnect("mcp://not-connected.com")

    @pytest.mark.asyncio
    async def test_disconnect_all(self, client):
        """Test disconnecting from all servers."""
        server1 = MCPServer(uri="mcp://server1.com", name="server1")
        server2 = MCPServer(uri="mcp://server2.com", name="server2")

        await client.connect(server1)
        await client.connect(server2)
        assert len(client.connected_servers) == 2

        await client.disconnect_all()

        assert len(client.connected_servers) == 0

    @pytest.mark.asyncio
    async def test_get_server(self, client, test_server):
        """Test getting server configuration."""
        await client.connect(test_server)

        server = client.get_server("mcp://test.lseg.com")

        assert server is not None
        assert server.name == "test-server"
        assert server.capabilities.tools is True  # Set during negotiation

    @pytest.mark.asyncio
    async def test_get_server_not_connected(self, client):
        """Test getting server that's not connected."""
        server = client.get_server("mcp://not-connected.com")

        assert server is None

    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self, client):
        """Test listing tools from non-connected server raises error."""
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await client.list_tools("mcp://not-connected.com")

    @pytest.mark.asyncio
    async def test_list_tools_returns_empty(self, client, test_server):
        """Test listing tools returns empty list (placeholder impl)."""
        await client.connect(test_server)

        tools = await client.list_tools("mcp://test.lseg.com")

        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, client):
        """Test calling tool on non-connected server raises error."""
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await client.call_tool(
                "mcp://not-connected.com",
                "get_price",
                {"ric": "AAPL.O"},
            )

    @pytest.mark.asyncio
    async def test_call_tool_returns_error(self, client, test_server):
        """Test tool execution returns error (placeholder impl)."""
        await client.connect(test_server)

        result = await client.call_tool(
            "mcp://test.lseg.com",
            "get_price",
            {"ric": "AAPL.O"},
        )

        assert result.is_error is True
        assert "not yet implemented" in result.error_message.lower()
        assert result.tool_name == "get_price"

    @pytest.mark.asyncio
    async def test_list_resources_not_connected(self, client):
        """Test listing resources from non-connected server raises error."""
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await client.list_resources("mcp://not-connected.com")

    @pytest.mark.asyncio
    async def test_list_resources_returns_empty(self, client, test_server):
        """Test listing resources returns empty list (placeholder impl)."""
        await client.connect(test_server)

        resources = await client.list_resources("mcp://test.lseg.com")

        assert resources == []

    @pytest.mark.asyncio
    async def test_read_resource_not_connected(self, client):
        """Test reading resource from non-connected server raises error."""
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await client.read_resource(
                "mcp://not-connected.com",
                "mcp://not-connected.com/resource/1",
            )

    @pytest.mark.asyncio
    async def test_read_resource(self, client, test_server):
        """Test reading resource (placeholder impl)."""
        await client.connect(test_server)

        resource = await client.read_resource(
            "mcp://test.lseg.com",
            "mcp://test.lseg.com/docs/field_codes",
        )

        assert resource.uri == "mcp://test.lseg.com/docs/field_codes"
        assert resource.content is None  # Placeholder returns no content

    @pytest.mark.asyncio
    async def test_health_check_connected(self, client, test_server):
        """Test health check for connected server."""
        await client.connect(test_server)

        result = await client.health_check("mcp://test.lseg.com")

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, client):
        """Test health check for non-connected server."""
        result = await client.health_check("mcp://not-connected.com")

        assert result is False


class TestMCPClientCaching:
    """Tests for MCP client caching behavior."""

    @pytest.fixture
    def caching_client(self):
        """Create a client with caching enabled."""
        return MCPClient(MCPClientConfig(
            enable_caching=True,
            cache_ttl_seconds=60,
        ))

    @pytest.fixture
    def test_server(self):
        """Create a test server config."""
        return MCPServer(
            uri="mcp://cached.lseg.com",
            name="cached-server",
        )

    @pytest.mark.asyncio
    async def test_invalidate_cache_specific_server(self, caching_client, test_server):
        """Test invalidating cache for specific server."""
        await caching_client.connect(test_server)

        # Trigger cache population
        await caching_client.list_tools("mcp://cached.lseg.com")

        # Invalidate
        caching_client.invalidate_cache("mcp://cached.lseg.com")

        # Cache should be invalidated (next call would fetch again)
        # Since implementation is placeholder, just verify no errors
        tools = await caching_client.list_tools("mcp://cached.lseg.com")
        assert tools == []

    @pytest.mark.asyncio
    async def test_invalidate_all_cache(self, caching_client, test_server):
        """Test invalidating all cache entries."""
        await caching_client.connect(test_server)

        # Trigger cache population
        await caching_client.list_tools("mcp://cached.lseg.com")

        # Invalidate all
        caching_client.invalidate_cache()

        # Should not raise
        tools = await caching_client.list_tools("mcp://cached.lseg.com")
        assert tools == []
