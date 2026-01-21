"""
Tests for MCP Context Retriever

Tests the MCPContextRetriever and DualModeContextRetriever.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.nl2api.mcp.client import MCPClient
from src.nl2api.mcp.protocols import MCPResource
from src.nl2api.mcp.context import (
    ContextProvider,
    MCPContextRetriever,
    DualModeContextRetriever,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockRAGRetriever:
    """Mock RAG retriever for testing."""

    def __init__(
        self,
        field_codes: list[dict] | None = None,
        examples: list[dict] | None = None,
    ):
        self._field_codes = field_codes or []
        self._examples = examples or []

    async def get_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[dict]:
        return self._field_codes[:limit]

    async def get_query_examples(
        self,
        query: str,
        domain: str,
        limit: int = 3,
    ) -> list[dict]:
        return self._examples[:limit]


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
            MCPResource(uri=f"mcp://test/field_codes/{i}", name=f"F{i}")
            for i in range(10)
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


# =============================================================================
# DualModeContextRetriever Tests
# =============================================================================


class TestDualModeContextRetriever:
    """Tests for DualModeContextRetriever."""

    @pytest.fixture
    def mock_rag(self):
        """Create mock RAG retriever."""
        return MockRAGRetriever(
            field_codes=[{"code": "RAG_CODE", "description": "From RAG"}],
            examples=[{"query": "RAG query", "api_call": "RAG response"}],
        )

    @pytest.fixture
    def mock_mcp_retriever(self):
        """Create mock MCP retriever."""
        retriever = MagicMock(spec=MCPContextRetriever)
        retriever.get_field_codes = AsyncMock(
            return_value=[{"code": "MCP_CODE", "description": "From MCP"}]
        )
        retriever.get_query_examples = AsyncMock(
            return_value=[{"query": "MCP query", "api_call": "MCP response"}]
        )
        return retriever

    def test_init(self, mock_rag, mock_mcp_retriever):
        """Test initialization."""
        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=mock_mcp_retriever,
            mode="local",
        )

        assert retriever.mode == "local"

    def test_set_mode(self, mock_rag):
        """Test setting mode."""
        retriever = DualModeContextRetriever(rag_retriever=mock_rag, mode="local")

        retriever.set_mode("mcp")
        assert retriever.mode == "mcp"

        retriever.set_mode("hybrid")
        assert retriever.mode == "hybrid"

    def test_set_invalid_mode(self, mock_rag):
        """Test setting invalid mode raises error."""
        retriever = DualModeContextRetriever(rag_retriever=mock_rag, mode="local")

        with pytest.raises(ValueError, match="Invalid mode"):
            retriever.set_mode("invalid")

    @pytest.mark.asyncio
    async def test_local_mode_uses_rag(self, mock_rag, mock_mcp_retriever):
        """Test local mode uses RAG retriever."""
        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=mock_mcp_retriever,
            mode="local",
        )

        field_codes = await retriever.get_field_codes("query", "domain")
        examples = await retriever.get_query_examples("query", "domain")

        assert field_codes[0]["code"] == "RAG_CODE"
        assert examples[0]["query"] == "RAG query"
        mock_mcp_retriever.get_field_codes.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_mode_uses_mcp(self, mock_rag, mock_mcp_retriever):
        """Test MCP mode uses MCP retriever."""
        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=mock_mcp_retriever,
            mode="mcp",
        )

        field_codes = await retriever.get_field_codes("query", "domain")
        examples = await retriever.get_query_examples("query", "domain")

        assert field_codes[0]["code"] == "MCP_CODE"
        assert examples[0]["query"] == "MCP query"

    @pytest.mark.asyncio
    async def test_hybrid_mode_prefers_mcp(self, mock_rag, mock_mcp_retriever):
        """Test hybrid mode prefers MCP."""
        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=mock_mcp_retriever,
            mode="hybrid",
        )

        field_codes = await retriever.get_field_codes("query", "domain")

        assert field_codes[0]["code"] == "MCP_CODE"

    @pytest.mark.asyncio
    async def test_hybrid_mode_falls_back_to_rag(self, mock_rag, mock_mcp_retriever):
        """Test hybrid mode falls back to RAG when MCP returns empty."""
        mock_mcp_retriever.get_field_codes = AsyncMock(return_value=[])

        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=mock_mcp_retriever,
            mode="hybrid",
            fallback_enabled=True,
        )

        field_codes = await retriever.get_field_codes("query", "domain")

        assert field_codes[0]["code"] == "RAG_CODE"

    @pytest.mark.asyncio
    async def test_hybrid_mode_no_fallback(self, mock_rag, mock_mcp_retriever):
        """Test hybrid mode without fallback."""
        mock_mcp_retriever.get_field_codes = AsyncMock(return_value=[])

        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=mock_mcp_retriever,
            mode="hybrid",
            fallback_enabled=False,
        )

        field_codes = await retriever.get_field_codes("query", "domain")

        assert field_codes == []

    @pytest.mark.asyncio
    async def test_local_mode_without_rag(self, mock_mcp_retriever):
        """Test local mode returns empty when RAG not configured."""
        retriever = DualModeContextRetriever(
            rag_retriever=None,
            mcp_retriever=mock_mcp_retriever,
            mode="local",
        )

        field_codes = await retriever.get_field_codes("query", "domain")

        assert field_codes == []

    @pytest.mark.asyncio
    async def test_mcp_mode_without_mcp(self, mock_rag):
        """Test MCP mode returns empty when MCP not configured."""
        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=None,
            mode="mcp",
        )

        field_codes = await retriever.get_field_codes("query", "domain")

        assert field_codes == []

    @pytest.mark.asyncio
    async def test_handles_rag_error(self, mock_mcp_retriever):
        """Test handles RAG errors gracefully."""
        rag = MagicMock()
        rag.get_field_codes = AsyncMock(side_effect=Exception("RAG error"))

        retriever = DualModeContextRetriever(
            rag_retriever=rag,
            mcp_retriever=mock_mcp_retriever,
            mode="local",
        )

        field_codes = await retriever.get_field_codes("query", "domain")

        assert field_codes == []

    @pytest.mark.asyncio
    async def test_handles_mcp_error_with_fallback(self, mock_rag, mock_mcp_retriever):
        """Test handles MCP errors and falls back to RAG."""
        mock_mcp_retriever.get_field_codes = AsyncMock(
            side_effect=Exception("MCP error")
        )

        retriever = DualModeContextRetriever(
            rag_retriever=mock_rag,
            mcp_retriever=mock_mcp_retriever,
            mode="hybrid",
            fallback_enabled=True,
        )

        field_codes = await retriever.get_field_codes("query", "domain")

        assert field_codes[0]["code"] == "RAG_CODE"
