"""
MCP Protocol Definitions

Defines the data types and protocols for MCP (Model Context Protocol)
integration. These types mirror the MCP specification while remaining
compatible with the existing LLM tool infrastructure.

Reference: https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.nl2api.llm.protocols import LLMToolDefinition


class MCPResourceType(str, Enum):
    """Type of MCP resource."""

    TEXT = "text"
    BLOB = "blob"
    IMAGE = "image"


@dataclass(frozen=True)
class MCPResource:
    """
    An MCP resource representing contextual data.

    Resources provide read-only access to data that can be used
    as context for LLM operations (e.g., field code documentation,
    API examples, schema definitions).
    """

    uri: str
    name: str
    description: str | None = None
    mime_type: str = "text/plain"
    content: str | None = None
    resource_type: MCPResourceType = MCPResourceType.TEXT

    def to_context_string(self) -> str:
        """Convert resource to a context string for LLM."""
        if self.content:
            return f"[{self.name}]\n{self.content}"
        return f"[{self.name}] (URI: {self.uri})"


@dataclass(frozen=True)
class MCPToolParameter:
    """A parameter definition for an MCP tool."""

    name: str
    description: str
    type: str = "string"
    required: bool = True
    enum: tuple[str, ...] | None = None
    default: Any = None


@dataclass(frozen=True)
class MCPToolDefinition:
    """
    Definition of an MCP tool.

    MCP tools are similar to LLM tools but follow the MCP specification.
    This class provides conversion methods to/from LLMToolDefinition.
    """

    name: str
    description: str
    parameters: tuple[MCPToolParameter, ...] = ()
    server_uri: str | None = None  # Which MCP server provides this tool

    def to_json_schema(self) -> dict[str, Any]:
        """Convert parameters to JSON Schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = list(param.enum)
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_llm_tool_definition(self) -> LLMToolDefinition:
        """Convert to LLMToolDefinition for use with LLM providers."""
        from src.nl2api.llm.protocols import LLMToolDefinition

        return LLMToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.to_json_schema(),
        )

    @classmethod
    def from_mcp_response(cls, data: dict[str, Any]) -> MCPToolDefinition:
        """Create from MCP tools/list response item."""
        params = []
        input_schema = data.get("inputSchema", {})

        if input_schema.get("type") == "object":
            properties = input_schema.get("properties", {})
            required_params = set(input_schema.get("required", []))

            for name, prop in properties.items():
                params.append(
                    MCPToolParameter(
                        name=name,
                        description=prop.get("description", ""),
                        type=prop.get("type", "string"),
                        required=name in required_params,
                        enum=tuple(prop["enum"]) if "enum" in prop else None,
                        default=prop.get("default"),
                    )
                )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=tuple(params),
        )


@dataclass(frozen=True)
class MCPToolResult:
    """
    Result of an MCP tool execution.

    Contains the tool output along with metadata about the execution.
    """

    tool_name: str
    content: Any
    is_error: bool = False
    error_message: str | None = None
    execution_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {
            "tool_name": self.tool_name,
            "content": self.content,
            "is_error": self.is_error,
        }
        if self.is_error and self.error_message:
            result["error_message"] = self.error_message
        if self.execution_time_ms > 0:
            result["execution_time_ms"] = self.execution_time_ms
        return result


@dataclass(frozen=True)
class MCPCapabilities:
    """Capabilities advertised by an MCP server."""

    tools: bool = False
    resources: bool = False
    prompts: bool = False
    sampling: bool = False
    logging: bool = False


@dataclass(frozen=True)
class MCPServer:
    """
    Configuration for an MCP server connection.

    Represents a remote MCP server that provides tools and resources.
    """

    uri: str
    name: str
    description: str | None = None
    capabilities: MCPCapabilities = field(default_factory=MCPCapabilities)
    api_key: str | None = None
    timeout_seconds: int = 30

    @property
    def has_tools(self) -> bool:
        """Check if server supports tools."""
        return self.capabilities.tools

    @property
    def has_resources(self) -> bool:
        """Check if server supports resources."""
        return self.capabilities.resources


@runtime_checkable
class MCPClientProtocol(Protocol):
    """
    Protocol for MCP client implementations.

    Defines the interface for interacting with MCP servers.
    """

    async def connect(self, server: MCPServer) -> bool:
        """
        Connect to an MCP server.

        Returns True if connection successful.
        """
        ...

    async def disconnect(self, server_uri: str) -> None:
        """Disconnect from an MCP server."""
        ...

    async def list_tools(self, server_uri: str) -> list[MCPToolDefinition]:
        """List available tools from an MCP server."""
        ...

    async def call_tool(
        self,
        server_uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """Execute a tool on an MCP server."""
        ...

    async def list_resources(self, server_uri: str) -> list[MCPResource]:
        """List available resources from an MCP server."""
        ...

    async def read_resource(self, server_uri: str, resource_uri: str) -> MCPResource:
        """Read a specific resource from an MCP server."""
        ...


@runtime_checkable
class MCPResourceProvider(Protocol):
    """
    Protocol for providing MCP resources.

    Used by both local implementations and MCP server adapters.
    """

    async def get_resources(self, domain: str) -> list[MCPResource]:
        """Get resources for a specific domain."""
        ...

    async def get_resource_by_uri(self, uri: str) -> MCPResource | None:
        """Get a specific resource by URI."""
        ...

    async def search_resources(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[MCPResource]:
        """Search for resources matching a query."""
        ...
