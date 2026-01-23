"""
Tests for MCP Cache

Tests the MCP caching implementations.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.nl2api.mcp.cache import (
    CacheEntry,
    InMemoryMCPCache,
    RedisMCPCache,
)
from src.nl2api.mcp.protocols import (
    MCPResource,
    MCPToolDefinition,
    MCPToolParameter,
    MCPToolResult,
)


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_entry_not_expired(self):
        """Test entry that has not expired."""
        entry = CacheEntry(
            value="test",
            expires_at=time.time() + 100,
        )

        assert entry.is_expired is False

    def test_entry_expired(self):
        """Test entry that has expired."""
        entry = CacheEntry(
            value="test",
            expires_at=time.time() - 1,
        )

        assert entry.is_expired is True


class TestInMemoryMCPCache:
    """Tests for InMemoryMCPCache."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        return InMemoryMCPCache(default_ttl_seconds=60)

    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""
        return [
            MCPToolDefinition(
                name="get_price",
                description="Get stock price",
                parameters=(
                    MCPToolParameter(
                        name="ric",
                        description="Instrument code",
                        type="string",
                    ),
                ),
            ),
            MCPToolDefinition(
                name="get_volume",
                description="Get trading volume",
            ),
        ]

    @pytest.fixture
    def sample_resources(self):
        """Create sample resources for testing."""
        return [
            MCPResource(
                uri="mcp://test/doc1",
                name="Doc 1",
                content="Content 1",
            ),
            MCPResource(
                uri="mcp://test/doc2",
                name="Doc 2",
            ),
        ]

    @pytest.mark.asyncio
    async def test_set_and_get_tools(self, cache, sample_tools):
        """Test caching and retrieving tools."""
        await cache.set_tools("mcp://server1", sample_tools)

        cached = await cache.get_tools("mcp://server1")

        assert cached is not None
        assert len(cached) == 2
        assert cached[0].name == "get_price"

    @pytest.mark.asyncio
    async def test_get_tools_cache_miss(self, cache):
        """Test getting tools that aren't cached."""
        cached = await cache.get_tools("mcp://not-cached")

        assert cached is None

    @pytest.mark.asyncio
    async def test_set_and_get_resources(self, cache, sample_resources):
        """Test caching and retrieving resources."""
        await cache.set_resources("mcp://server1", sample_resources)

        cached = await cache.get_resources("mcp://server1")

        assert cached is not None
        assert len(cached) == 2
        assert cached[0].name == "Doc 1"

    @pytest.mark.asyncio
    async def test_get_resources_cache_miss(self, cache):
        """Test getting resources that aren't cached."""
        cached = await cache.get_resources("mcp://not-cached")

        assert cached is None

    @pytest.mark.asyncio
    async def test_set_and_get_tool_result(self, cache):
        """Test caching and retrieving tool results."""
        result = MCPToolResult(
            tool_name="get_price",
            content={"price": 150.25},
            execution_time_ms=42,
        )

        await cache.set_tool_result(
            "mcp://server1",
            "get_price",
            {"ric": "AAPL.O"},
            result,
        )

        cached = await cache.get_tool_result(
            "mcp://server1",
            "get_price",
            {"ric": "AAPL.O"},
        )

        assert cached is not None
        assert cached.content == {"price": 150.25}

    @pytest.mark.asyncio
    async def test_tool_result_different_args_different_cache(self, cache):
        """Test that different arguments result in different cache entries."""
        result1 = MCPToolResult(tool_name="get_price", content={"price": 150})
        result2 = MCPToolResult(tool_name="get_price", content={"price": 200})

        await cache.set_tool_result("mcp://server1", "get_price", {"ric": "AAPL.O"}, result1)
        await cache.set_tool_result("mcp://server1", "get_price", {"ric": "MSFT.O"}, result2)

        cached1 = await cache.get_tool_result("mcp://server1", "get_price", {"ric": "AAPL.O"})
        cached2 = await cache.get_tool_result("mcp://server1", "get_price", {"ric": "MSFT.O"})

        assert cached1.content == {"price": 150}
        assert cached2.content == {"price": 200}

    @pytest.mark.asyncio
    async def test_does_not_cache_error_results(self, cache):
        """Test that error results are not cached."""
        error_result = MCPToolResult(
            tool_name="get_price",
            content=None,
            is_error=True,
            error_message="Failed",
        )

        await cache.set_tool_result(
            "mcp://server1",
            "get_price",
            {"ric": "INVALID"},
            error_result,
        )

        cached = await cache.get_tool_result(
            "mcp://server1",
            "get_price",
            {"ric": "INVALID"},
        )

        assert cached is None

    @pytest.mark.asyncio
    async def test_invalidate_specific_server(self, cache, sample_tools, sample_resources):
        """Test invalidating cache for a specific server."""
        await cache.set_tools("mcp://server1", sample_tools)
        await cache.set_tools("mcp://server2", sample_tools)
        await cache.set_resources("mcp://server1", sample_resources)

        await cache.invalidate("mcp://server1")

        assert await cache.get_tools("mcp://server1") is None
        assert await cache.get_resources("mcp://server1") is None
        assert await cache.get_tools("mcp://server2") is not None

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache, sample_tools, sample_resources):
        """Test invalidating all cache entries."""
        await cache.set_tools("mcp://server1", sample_tools)
        await cache.set_tools("mcp://server2", sample_tools)
        await cache.set_resources("mcp://server1", sample_resources)

        await cache.invalidate()

        assert await cache.get_tools("mcp://server1") is None
        assert await cache.get_tools("mcp://server2") is None
        assert await cache.get_resources("mcp://server1") is None

    @pytest.mark.asyncio
    async def test_expired_entries_return_none(self, cache, sample_tools):
        """Test that expired entries return None."""
        # Create cache with very short TTL
        short_ttl_cache = InMemoryMCPCache(default_ttl_seconds=0)

        await short_ttl_cache.set_tools("mcp://server1", sample_tools)

        # Wait for expiration
        await asyncio.sleep(0.1)

        cached = await short_ttl_cache.get_tools("mcp://server1")

        assert cached is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache, sample_tools):
        """Test cleaning up expired entries."""
        short_ttl_cache = InMemoryMCPCache(default_ttl_seconds=0)

        await short_ttl_cache.set_tools("mcp://server1", sample_tools)
        await short_ttl_cache.set_tools("mcp://server2", sample_tools)

        await asyncio.sleep(0.1)

        removed = await short_ttl_cache.cleanup_expired()

        assert removed >= 2
        assert short_ttl_cache.stats()["tools_cached"] == 0

    def test_stats(self, cache):
        """Test cache statistics."""
        stats = cache.stats()

        assert stats["tools_cached"] == 0
        assert stats["resources_cached"] == 0
        assert stats["results_cached"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_caching(self, cache, sample_tools, sample_resources):
        """Test cache statistics after adding entries."""
        await cache.set_tools("mcp://server1", sample_tools)
        await cache.set_resources("mcp://server1", sample_resources)
        await cache.set_tool_result(
            "mcp://server1",
            "get_price",
            {"ric": "AAPL"},
            MCPToolResult(tool_name="get_price", content={}),
        )

        stats = cache.stats()

        assert stats["tools_cached"] == 1
        assert stats["resources_cached"] == 1
        assert stats["results_cached"] == 1


class TestRedisMCPCache:
    """Tests for RedisMCPCache."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock()
        redis.scan = AsyncMock(return_value=(0, []))
        redis.delete = AsyncMock()
        return redis

    @pytest.fixture
    def cache(self, mock_redis):
        """Create a Redis cache with mocked client."""
        return RedisMCPCache(
            redis_client=mock_redis,
            key_prefix="test:mcp:",
            default_ttl_seconds=300,
        )

    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""
        return [
            MCPToolDefinition(
                name="get_price",
                description="Get stock price",
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_tools_cache_miss(self, cache, mock_redis):
        """Test getting tools on cache miss."""
        mock_redis.get.return_value = None

        result = await cache.get_tools("mcp://server1")

        assert result is None
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tools_cache_hit(self, cache, mock_redis):
        """Test getting tools on cache hit."""
        cached_data = '[{"name": "get_price", "description": "Get price", "inputSchema": {"type": "object", "properties": {}, "required": []}}]'
        mock_redis.get.return_value = cached_data.encode()

        result = await cache.get_tools("mcp://server1")

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_price"

    @pytest.mark.asyncio
    async def test_set_tools(self, cache, mock_redis, sample_tools):
        """Test caching tools."""
        await cache.set_tools("mcp://server1", sample_tools)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "test:mcp:tools:mcp://server1"
        assert call_args[0][1] == 300  # TTL

    @pytest.mark.asyncio
    async def test_get_resources_cache_miss(self, cache, mock_redis):
        """Test getting resources on cache miss."""
        mock_redis.get.return_value = None

        result = await cache.get_resources("mcp://server1")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_resources_cache_hit(self, cache, mock_redis):
        """Test getting resources on cache hit."""
        cached_data = '[{"uri": "mcp://test/doc", "name": "Doc", "description": "Test doc", "mime_type": "text/plain"}]'
        mock_redis.get.return_value = cached_data.encode()

        result = await cache.get_resources("mcp://server1")

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "Doc"

    @pytest.mark.asyncio
    async def test_set_tool_result(self, cache, mock_redis):
        """Test caching tool result."""
        result = MCPToolResult(
            tool_name="get_price",
            content={"price": 150},
            execution_time_ms=42,
        )

        await cache.set_tool_result(
            "mcp://server1",
            "get_price",
            {"ric": "AAPL"},
            result,
        )

        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_cache_error_result(self, cache, mock_redis):
        """Test that error results are not cached."""
        error_result = MCPToolResult(
            tool_name="get_price",
            content=None,
            is_error=True,
            error_message="Failed",
        )

        await cache.set_tool_result(
            "mcp://server1",
            "get_price",
            {"ric": "INVALID"},
            error_result,
        )

        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache, mock_redis):
        """Test invalidating all cache entries."""
        mock_redis.scan.return_value = (0, [b"test:mcp:tools:server1"])

        await cache.invalidate()

        mock_redis.scan.assert_called()
        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_invalidate_specific_server(self, cache, mock_redis):
        """Test invalidating cache for specific server."""
        mock_redis.scan.return_value = (0, [b"test:mcp:tools:mcp://server1"])

        await cache.invalidate("mcp://server1")

        # Verify scan was called with pattern containing server URI
        call_args = mock_redis.scan.call_args
        assert "server1" in call_args[1]["match"]
