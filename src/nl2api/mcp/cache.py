"""
MCP Cache Implementation

Provides caching for MCP tool definitions and resources to reduce
latency and server load. Supports both in-memory caching for testing
and Redis-based caching for production.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.nl2api.mcp.protocols import MCPResource, MCPToolDefinition, MCPToolResult

logger = logging.getLogger(__name__)


@runtime_checkable
class MCPCache(Protocol):
    """Protocol for MCP caching implementations."""

    async def get_tools(self, server_uri: str) -> list[MCPToolDefinition] | None:
        """Get cached tools for a server."""
        ...

    async def set_tools(
        self,
        server_uri: str,
        tools: list[MCPToolDefinition],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache tools for a server."""
        ...

    async def get_resources(self, server_uri: str) -> list[MCPResource] | None:
        """Get cached resources for a server."""
        ...

    async def set_resources(
        self,
        server_uri: str,
        resources: list[MCPResource],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache resources for a server."""
        ...

    async def get_tool_result(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult | None:
        """Get cached tool result."""
        ...

    async def set_tool_result(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: MCPToolResult,
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache tool execution result."""
        ...

    async def invalidate(self, server_uri: str | None = None) -> None:
        """Invalidate cache entries."""
        ...


@dataclass
class CacheEntry:
    """A cached entry with expiration."""

    value: Any
    expires_at: float

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at


class InMemoryMCPCache:
    """
    In-memory MCP cache for testing and development.

    Supports TTL-based expiration but has no persistence or
    distributed cache support.
    """

    def __init__(self, default_ttl_seconds: int = 300):
        """
        Initialize in-memory cache.

        Args:
            default_ttl_seconds: Default TTL for cache entries
        """
        self._default_ttl = default_ttl_seconds
        self._tools: dict[str, CacheEntry] = {}
        self._resources: dict[str, CacheEntry] = {}
        self._results: dict[str, CacheEntry] = {}

    def _make_result_key(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Generate cache key for tool result."""
        args_hash = hashlib.md5(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest()
        return f"{server_uri}:{tool_name}:{args_hash}"

    async def get_tools(self, server_uri: str) -> list[MCPToolDefinition] | None:
        """Get cached tools for a server."""
        entry = self._tools.get(server_uri)
        if entry and not entry.is_expired:
            logger.debug(f"Cache hit: tools for {server_uri}")
            return entry.value
        if entry and entry.is_expired:
            del self._tools[server_uri]
        return None

    async def set_tools(
        self,
        server_uri: str,
        tools: list[MCPToolDefinition],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache tools for a server."""
        ttl = ttl_seconds or self._default_ttl
        self._tools[server_uri] = CacheEntry(
            value=tools,
            expires_at=time.time() + ttl,
        )
        logger.debug(f"Cached {len(tools)} tools for {server_uri} (TTL={ttl}s)")

    async def get_resources(self, server_uri: str) -> list[MCPResource] | None:
        """Get cached resources for a server."""
        entry = self._resources.get(server_uri)
        if entry and not entry.is_expired:
            logger.debug(f"Cache hit: resources for {server_uri}")
            return entry.value
        if entry and entry.is_expired:
            del self._resources[server_uri]
        return None

    async def set_resources(
        self,
        server_uri: str,
        resources: list[MCPResource],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache resources for a server."""
        ttl = ttl_seconds or self._default_ttl
        self._resources[server_uri] = CacheEntry(
            value=resources,
            expires_at=time.time() + ttl,
        )
        logger.debug(f"Cached {len(resources)} resources for {server_uri} (TTL={ttl}s)")

    async def get_tool_result(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult | None:
        """Get cached tool result."""
        key = self._make_result_key(server_uri, tool_name, arguments)
        entry = self._results.get(key)
        if entry and not entry.is_expired:
            logger.debug(f"Cache hit: result for {tool_name}")
            return entry.value
        if entry and entry.is_expired:
            del self._results[key]
        return None

    async def set_tool_result(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: MCPToolResult,
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache tool execution result."""
        # Don't cache error results
        if result.is_error:
            return

        key = self._make_result_key(server_uri, tool_name, arguments)
        ttl = ttl_seconds or self._default_ttl
        self._results[key] = CacheEntry(
            value=result,
            expires_at=time.time() + ttl,
        )
        logger.debug(f"Cached result for {tool_name} (TTL={ttl}s)")

    async def invalidate(self, server_uri: str | None = None) -> None:
        """
        Invalidate cache entries.

        Args:
            server_uri: If provided, only invalidate entries for this server.
                       If None, clear all entries.
        """
        if server_uri is None:
            self._tools.clear()
            self._resources.clear()
            self._results.clear()
            logger.debug("Cleared all MCP cache entries")
        else:
            self._tools.pop(server_uri, None)
            self._resources.pop(server_uri, None)
            # Remove result entries for this server
            keys_to_remove = [
                k for k in self._results if k.startswith(server_uri)
            ]
            for key in keys_to_remove:
                del self._results[key]
            logger.debug(f"Cleared MCP cache entries for {server_uri}")

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        removed = 0
        now = time.time()

        for cache in [self._tools, self._resources, self._results]:
            expired_keys = [
                k for k, v in cache.items()
                if v.expires_at < now
            ]
            for key in expired_keys:
                del cache[key]
                removed += 1

        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired cache entries")

        return removed

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "tools_cached": len(self._tools),
            "resources_cached": len(self._resources),
            "results_cached": len(self._results),
        }


class RedisMCPCache:
    """
    Redis-based MCP cache for production use.

    Provides persistent, distributed caching with automatic expiration.
    """

    def __init__(
        self,
        redis_client: Any,  # redis.asyncio.Redis
        key_prefix: str = "mcp:",
        default_ttl_seconds: int = 300,
    ):
        """
        Initialize Redis cache.

        Args:
            redis_client: Async Redis client
            key_prefix: Prefix for all cache keys
            default_ttl_seconds: Default TTL for cache entries
        """
        self._redis = redis_client
        self._prefix = key_prefix
        self._default_ttl = default_ttl_seconds

    def _key(self, *parts: str) -> str:
        """Generate prefixed cache key."""
        return self._prefix + ":".join(parts)

    def _make_result_key(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Generate cache key for tool result."""
        args_hash = hashlib.md5(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest()
        return self._key("result", server_uri, tool_name, args_hash)

    async def get_tools(self, server_uri: str) -> list[MCPToolDefinition] | None:
        """Get cached tools for a server."""
        key = self._key("tools", server_uri)
        data = await self._redis.get(key)
        if data:
            try:
                items = json.loads(data)
                return [
                    MCPToolDefinition.from_mcp_response(item)
                    for item in items
                ]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to decode cached tools: {e}")
        return None

    async def set_tools(
        self,
        server_uri: str,
        tools: list[MCPToolDefinition],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache tools for a server."""
        key = self._key("tools", server_uri)
        ttl = ttl_seconds or self._default_ttl

        # Convert to JSON-serializable format
        items = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.to_json_schema(),
            }
            for t in tools
        ]

        await self._redis.setex(key, ttl, json.dumps(items))
        logger.debug(f"Cached {len(tools)} tools for {server_uri} in Redis")

    async def get_resources(self, server_uri: str) -> list[MCPResource] | None:
        """Get cached resources for a server."""
        key = self._key("resources", server_uri)
        data = await self._redis.get(key)
        if data:
            try:
                items = json.loads(data)
                return [
                    MCPResource(
                        uri=item["uri"],
                        name=item["name"],
                        description=item.get("description"),
                        mime_type=item.get("mime_type", "text/plain"),
                        content=item.get("content"),
                    )
                    for item in items
                ]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to decode cached resources: {e}")
        return None

    async def set_resources(
        self,
        server_uri: str,
        resources: list[MCPResource],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache resources for a server."""
        key = self._key("resources", server_uri)
        ttl = ttl_seconds or self._default_ttl

        items = [
            {
                "uri": r.uri,
                "name": r.name,
                "description": r.description,
                "mime_type": r.mime_type,
                "content": r.content,
            }
            for r in resources
        ]

        await self._redis.setex(key, ttl, json.dumps(items))
        logger.debug(f"Cached {len(resources)} resources for {server_uri} in Redis")

    async def get_tool_result(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult | None:
        """Get cached tool result."""
        key = self._make_result_key(server_uri, tool_name, arguments)
        data = await self._redis.get(key)
        if data:
            try:
                item = json.loads(data)
                return MCPToolResult(
                    tool_name=item["tool_name"],
                    content=item["content"],
                    is_error=item.get("is_error", False),
                    error_message=item.get("error_message"),
                    execution_time_ms=item.get("execution_time_ms", 0),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to decode cached result: {e}")
        return None

    async def set_tool_result(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: MCPToolResult,
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache tool execution result."""
        # Don't cache error results
        if result.is_error:
            return

        key = self._make_result_key(server_uri, tool_name, arguments)
        ttl = ttl_seconds or self._default_ttl

        await self._redis.setex(key, ttl, json.dumps(result.to_dict()))
        logger.debug(f"Cached result for {tool_name} in Redis")

    async def invalidate(self, server_uri: str | None = None) -> None:
        """
        Invalidate cache entries.

        Args:
            server_uri: If provided, only invalidate entries for this server.
                       If None, clear all MCP cache entries.
        """
        if server_uri is None:
            # Clear all MCP cache entries
            pattern = f"{self._prefix}*"
        else:
            # Clear entries for specific server
            pattern = f"{self._prefix}*{server_uri}*"

        # Use SCAN to find keys (safer than KEYS for large datasets)
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
            if keys:
                await self._redis.delete(*keys)
            if cursor == 0:
                break

        logger.debug(f"Invalidated MCP cache entries matching: {pattern}")
