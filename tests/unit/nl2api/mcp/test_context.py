"""
Tests for MCP Context Retriever

Tests the MCPContextRetriever.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nl2api.mcp.client import MCPClient
from src.nl2api.mcp.context import (
    MCPContextRetriever,
)
from src.nl2api.mcp.protocols import MCPResource

# =============================================================================
# MCPContextRetriever Tests
# =============================================================================


class TestMCPContextRetriever:
    """Tests for MCPContextRetriever."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = MagicMock(spec=MCPClient)
        client.is_connected.return_value = True
        client.list_resources = AsyncMock(return_value=[])
        return client

    def test_init(self, mock_mcp_client):
        """Test retriever initialization."""
        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"datastream": "mcp://datastream.lseg.com"},
        )

        assert retriever.get_server_uri("datastream") == "mcp://datastream.lseg.com"
        assert retriever.get_server_uri("unknown") is None

    def test_add_domain_mapping(self, mock_mcp_client):
        """Test adding domain mappings."""
        retriever = MCPContextRetriever(mcp_client=mock_mcp_client)

        retriever.add_domain_mapping("estimates", "mcp://estimates.lseg.com")

        assert retriever.get_server_uri("estimates") == "mcp://estimates.lseg.com"

    @pytest.mark.asyncio
    async def test_get_field_codes_no_server_mapping(self, mock_mcp_client):
        """Test get_field_codes with no server mapping."""
        retriever = MCPContextRetriever(mcp_client=mock_mcp_client)

        result = await retriever.get_field_codes("price", "unknown_domain")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_field_codes_not_connected(self, mock_mcp_client):
        """Test get_field_codes when not connected."""
        mock_mcp_client.is_connected.return_value = False
        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"datastream": "mcp://datastream.lseg.com"},
        )

        result = await retriever.get_field_codes("price", "datastream")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_field_codes_from_resources(self, mock_mcp_client):
        """Test retrieving field codes from MCP resources."""
        resources = [
            MCPResource(
                uri="mcp://datastream.lseg.com/field_codes/P",
                name="P",
                description="Price field code",
            ),
            MCPResource(
                uri="mcp://datastream.lseg.com/field_codes/VO",
                name="VO",
                description="Volume field code",
            ),
        ]
        mock_mcp_client.list_resources = AsyncMock(return_value=resources)

        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"datastream": "mcp://datastream.lseg.com"},
        )

        result = await retriever.get_field_codes("price", "datastream", limit=5)

        assert len(result) == 2
        assert result[0]["code"] == "P"
        assert result[0]["description"] == "Price field code"
        assert result[0]["source"] == "mcp"

    @pytest.mark.asyncio
    async def test_get_field_codes_respects_limit(self, mock_mcp_client):
        """Test that get_field_codes respects the limit parameter."""
        resources = [
            MCPResource(uri=f"mcp://test/field_codes/{i}", name=f"F{i}") for i in range(10)
        ]
        mock_mcp_client.list_resources = AsyncMock(return_value=resources)

        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"test": "mcp://test"},
        )

        result = await retriever.get_field_codes("query", "test", limit=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_field_codes_handles_error(self, mock_mcp_client):
        """Test get_field_codes handles errors gracefully."""
        mock_mcp_client.list_resources = AsyncMock(side_effect=Exception("Error"))

        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"test": "mcp://test"},
        )

        result = await retriever.get_field_codes("query", "test")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_query_examples_no_server_mapping(self, mock_mcp_client):
        """Test get_query_examples with no server mapping."""
        retriever = MCPContextRetriever(mcp_client=mock_mcp_client)

        result = await retriever.get_query_examples("price", "unknown_domain")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_query_examples_from_resources(self, mock_mcp_client):
        """Test retrieving examples from MCP resources."""
        resources = [
            MCPResource(
                uri="mcp://test/examples/1",
                name="What is the price?",
                description="API call example",
                content='{"query": "What is the price?", "api_call": {"tool": "get_price"}}',
            ),
        ]
        mock_mcp_client.list_resources = AsyncMock(return_value=resources)

        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"test": "mcp://test"},
        )

        result = await retriever.get_query_examples("price", "test")

        assert len(result) == 1
        assert result[0]["query"] == "What is the price?"
        assert result[0]["api_call"] == {"tool": "get_price"}
        assert result[0]["source"] == "mcp"

    @pytest.mark.asyncio
    async def test_get_query_examples_fallback_to_name(self, mock_mcp_client):
        """Test examples fallback to name/description when no JSON content."""
        resources = [
            MCPResource(
                uri="mcp://test/examples/1",
                name="Example query",
                description="Example response",
            ),
        ]
        mock_mcp_client.list_resources = AsyncMock(return_value=resources)

        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"test": "mcp://test"},
        )

        result = await retriever.get_query_examples("query", "test")

        assert len(result) == 1
        assert result[0]["query"] == "Example query"
        assert result[0]["api_call"] == "Example response"

    @pytest.mark.asyncio
    async def test_caching(self, mock_mcp_client):
        """Test resource caching."""
        resources = [MCPResource(uri="mcp://test/field_codes/P", name="P")]
        mock_mcp_client.list_resources = AsyncMock(return_value=resources)

        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"test": "mcp://test"},
            cache_enabled=True,
        )

        # First call
        await retriever.get_field_codes("query", "test")
        # Second call should use cache
        await retriever.get_field_codes("query", "test")

        # Should only call list_resources once
        assert mock_mcp_client.list_resources.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_mcp_client):
        """Test with caching disabled."""
        resources = [MCPResource(uri="mcp://test/field_codes/P", name="P")]
        mock_mcp_client.list_resources = AsyncMock(return_value=resources)

        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"test": "mcp://test"},
            cache_enabled=False,
        )

        # Multiple calls
        await retriever.get_field_codes("query", "test")
        await retriever.get_field_codes("query", "test")

        # Should call list_resources each time
        assert mock_mcp_client.list_resources.call_count == 2

    def test_invalidate_cache_all(self, mock_mcp_client):
        """Test invalidating all cache."""
        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={"test": "mcp://test"},
        )
        retriever._resource_cache["mcp://test:field_codes/"] = []

        retriever.invalidate_cache()

        assert len(retriever._resource_cache) == 0

    def test_invalidate_cache_domain(self, mock_mcp_client):
        """Test invalidating cache for specific domain."""
        retriever = MCPContextRetriever(
            mcp_client=mock_mcp_client,
            domain_server_map={
                "test1": "mcp://test1",
                "test2": "mcp://test2",
            },
        )
        retriever._resource_cache["mcp://test1:field_codes/"] = []
        retriever._resource_cache["mcp://test2:field_codes/"] = []

        retriever.invalidate_cache("test1")

        assert "mcp://test1:field_codes/" not in retriever._resource_cache
        assert "mcp://test2:field_codes/" in retriever._resource_cache
