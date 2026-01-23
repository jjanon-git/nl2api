"""
MCP Context Retriever

Retrieves field codes and query examples from MCP server resources,
providing the same interface as the RAG retriever for seamless
integration with domain agents.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from src.nl2api.mcp.client import MCPClient
from src.nl2api.mcp.protocols import MCPResource

logger = logging.getLogger(__name__)


@runtime_checkable
class ContextProvider(Protocol):
    """
    Protocol for context providers (RAG or MCP).

    This protocol abstracts the source of context data, allowing
    agents to receive field codes and examples from either:
    - Local RAG retrieval (existing implementation)
    - Remote MCP server resources (new MCP integration)
    """

    async def get_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant field codes for a query.

        Args:
            query: User's natural language query
            domain: Domain name (e.g., "datastream", "estimates")
            limit: Maximum number of field codes to return

        Returns:
            List of field code dictionaries with 'code' and 'description' keys
        """
        ...

    async def get_query_examples(
        self,
        query: str,
        domain: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant query examples for a query.

        Args:
            query: User's natural language query
            domain: Domain name (e.g., "datastream", "estimates")
            limit: Maximum number of examples to return

        Returns:
            List of example dictionaries with 'query' and 'api_call' keys
        """
        ...


class MCPContextRetriever:
    """
    Retrieves context from MCP server resources.

    This retriever fetches field code documentation and query examples
    from MCP servers, converting them to the same format used by the
    RAG retriever.

    Example:
        from src.nl2api.mcp.client import MCPClient, MCPClientConfig
        from src.nl2api.mcp.protocols import MCPServer

        client = MCPClient(MCPClientConfig())
        await client.connect(MCPServer(uri="mcp://datastream.lseg.com", name="datastream"))

        retriever = MCPContextRetriever(
            mcp_client=client,
            domain_server_map={"datastream": "mcp://datastream.lseg.com"},
        )

        field_codes = await retriever.get_field_codes("stock price", "datastream", limit=5)
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        domain_server_map: dict[str, str] | None = None,
        field_code_resource_prefix: str = "field_codes/",
        example_resource_prefix: str = "examples/",
        cache_enabled: bool = True,
    ):
        """
        Initialize the MCP context retriever.

        Args:
            mcp_client: MCP client for server communication
            domain_server_map: Mapping of domain names to MCP server URIs
            field_code_resource_prefix: Prefix for field code resources
            example_resource_prefix: Prefix for example resources
            cache_enabled: Whether to cache retrieved resources
        """
        self._mcp_client = mcp_client
        self._domain_server_map = domain_server_map or {}
        self._field_code_prefix = field_code_resource_prefix
        self._example_prefix = example_resource_prefix
        self._cache_enabled = cache_enabled
        self._resource_cache: dict[str, list[MCPResource]] = {}

    def add_domain_mapping(self, domain: str, server_uri: str) -> None:
        """Add a domain to server URI mapping."""
        self._domain_server_map[domain] = server_uri

    def get_server_uri(self, domain: str) -> str | None:
        """Get the MCP server URI for a domain."""
        return self._domain_server_map.get(domain)

    async def get_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve field codes from MCP server resources.

        Args:
            query: User's natural language query
            domain: Domain name for server lookup
            limit: Maximum number of field codes to return

        Returns:
            List of field code dictionaries matching RAG format
        """
        server_uri = self._domain_server_map.get(domain)
        if not server_uri:
            logger.warning(f"No MCP server configured for domain: {domain}")
            return []

        if not self._mcp_client.is_connected(server_uri):
            logger.warning(f"MCP client not connected to: {server_uri}")
            return []

        try:
            # Fetch field code resources
            resources = await self._get_domain_resources(
                server_uri=server_uri,
                resource_prefix=self._field_code_prefix,
            )

            # Filter and convert to field code format
            field_codes = []
            for resource in resources[:limit]:
                field_code = self._parse_field_code_resource(resource)
                if field_code:
                    field_codes.append(field_code)

            logger.debug(f"Retrieved {len(field_codes)} field codes from MCP for domain={domain}")
            return field_codes

        except Exception as e:
            logger.error(f"Failed to retrieve field codes from MCP: {e}")
            return []

    async def get_query_examples(
        self,
        query: str,
        domain: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve query examples from MCP server resources.

        Args:
            query: User's natural language query
            domain: Domain name for server lookup
            limit: Maximum number of examples to return

        Returns:
            List of example dictionaries matching RAG format
        """
        server_uri = self._domain_server_map.get(domain)
        if not server_uri:
            logger.warning(f"No MCP server configured for domain: {domain}")
            return []

        if not self._mcp_client.is_connected(server_uri):
            logger.warning(f"MCP client not connected to: {server_uri}")
            return []

        try:
            # Fetch example resources
            resources = await self._get_domain_resources(
                server_uri=server_uri,
                resource_prefix=self._example_prefix,
            )

            # Filter and convert to example format
            examples = []
            for resource in resources[:limit]:
                example = self._parse_example_resource(resource)
                if example:
                    examples.append(example)

            logger.debug(f"Retrieved {len(examples)} examples from MCP for domain={domain}")
            return examples

        except Exception as e:
            logger.error(f"Failed to retrieve examples from MCP: {e}")
            return []

    async def _get_domain_resources(
        self,
        server_uri: str,
        resource_prefix: str,
    ) -> list[MCPResource]:
        """
        Get resources from an MCP server with optional caching.

        Args:
            server_uri: MCP server URI
            resource_prefix: Filter resources by this prefix

        Returns:
            List of MCPResource objects
        """
        cache_key = f"{server_uri}:{resource_prefix}"

        # Check cache
        if self._cache_enabled and cache_key in self._resource_cache:
            return self._resource_cache[cache_key]

        # Fetch all resources
        all_resources = await self._mcp_client.list_resources(server_uri)

        # Filter by prefix
        filtered = [
            r
            for r in all_resources
            if r.uri.startswith(f"{server_uri}/{resource_prefix}") or resource_prefix in r.uri
        ]

        # Cache results
        if self._cache_enabled:
            self._resource_cache[cache_key] = filtered

        return filtered

    def _parse_field_code_resource(
        self,
        resource: MCPResource,
    ) -> dict[str, Any] | None:
        """
        Parse an MCP resource into field code format.

        Expected format matches RAG output:
        {"code": "TR.EPSMean", "description": "Mean EPS estimate"}

        Args:
            resource: MCP resource to parse

        Returns:
            Field code dictionary or None if parsing fails
        """
        try:
            # Extract code from resource name/URI
            code = resource.name
            if "/" in code:
                code = code.split("/")[-1]

            description = resource.description or ""
            if resource.content:
                # Use content as description if available
                description = resource.content[:500]  # Limit length

            return {
                "code": code,
                "description": description,
                "source": "mcp",
                "uri": resource.uri,
            }
        except Exception as e:
            logger.warning(f"Failed to parse field code resource: {e}")
            return None

    def _parse_example_resource(
        self,
        resource: MCPResource,
    ) -> dict[str, Any] | None:
        """
        Parse an MCP resource into example format.

        Expected format matches RAG output:
        {"query": "What is Apple's EPS?", "api_call": {...}}

        Args:
            resource: MCP resource to parse

        Returns:
            Example dictionary or None if parsing fails
        """
        try:
            # Try to parse content as JSON if available
            if resource.content:
                import json

                try:
                    data = json.loads(resource.content)
                    if "query" in data:
                        return {
                            "query": data.get("query", ""),
                            "api_call": data.get("api_call", data.get("response", "")),
                            "source": "mcp",
                            "uri": resource.uri,
                        }
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON from resource: {resource.uri}")

            # Fall back to using name/description
            return {
                "query": resource.name,
                "api_call": resource.description or "",
                "source": "mcp",
                "uri": resource.uri,
            }
        except Exception as e:
            logger.warning(f"Failed to parse example resource: {e}")
            return None

    def invalidate_cache(self, domain: str | None = None) -> None:
        """
        Invalidate cached resources.

        Args:
            domain: If provided, only invalidate cache for this domain.
                   If None, invalidate all cached resources.
        """
        if domain is None:
            self._resource_cache.clear()
            logger.debug("Cleared all MCP context cache")
        else:
            server_uri = self._domain_server_map.get(domain)
            if server_uri:
                keys_to_remove = [k for k in self._resource_cache if k.startswith(server_uri)]
                for key in keys_to_remove:
                    del self._resource_cache[key]
                logger.debug(f"Cleared MCP context cache for domain: {domain}")
