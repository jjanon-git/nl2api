"""
Tests for MCP Protocol Definitions

Tests the MCP data types and protocol definitions.
"""

import pytest
from src.nl2api.mcp.protocols import (
    MCPCapabilities,
    MCPResource,
    MCPResourceType,
    MCPServer,
    MCPToolDefinition,
    MCPToolParameter,
    MCPToolResult,
)


class TestMCPResource:
    """Tests for MCPResource."""

    def test_basic_resource(self):
        """Test creating a basic resource."""
        resource = MCPResource(
            uri="mcp://datastream/field_codes/price",
            name="Price Field Codes",
            description="Documentation for price-related field codes",
        )

        assert resource.uri == "mcp://datastream/field_codes/price"
        assert resource.name == "Price Field Codes"
        assert resource.mime_type == "text/plain"
        assert resource.resource_type == MCPResourceType.TEXT

    def test_resource_with_content(self):
        """Test resource with content."""
        resource = MCPResource(
            uri="mcp://datastream/examples/basic",
            name="Basic Examples",
            content="P = Price\nVO = Volume\nMV = Market Value",
        )

        assert resource.content == "P = Price\nVO = Volume\nMV = Market Value"

    def test_to_context_string_with_content(self):
        """Test context string generation with content."""
        resource = MCPResource(
            uri="mcp://datastream/docs",
            name="Datastream Docs",
            content="Field documentation here",
        )

        context = resource.to_context_string()

        assert context == "[Datastream Docs]\nField documentation here"

    def test_to_context_string_without_content(self):
        """Test context string generation without content."""
        resource = MCPResource(
            uri="mcp://datastream/docs",
            name="Datastream Docs",
        )

        context = resource.to_context_string()

        assert context == "[Datastream Docs] (URI: mcp://datastream/docs)"

    def test_resource_is_frozen(self):
        """Test that resource is immutable."""
        resource = MCPResource(uri="test://uri", name="Test")

        with pytest.raises(AttributeError):
            resource.name = "Changed"


class TestMCPToolParameter:
    """Tests for MCPToolParameter."""

    def test_basic_parameter(self):
        """Test creating a basic parameter."""
        param = MCPToolParameter(
            name="ric",
            description="Reuters Instrument Code",
            type="string",
            required=True,
        )

        assert param.name == "ric"
        assert param.type == "string"
        assert param.required is True
        assert param.enum is None

    def test_parameter_with_enum(self):
        """Test parameter with enum values."""
        param = MCPToolParameter(
            name="frequency",
            description="Data frequency",
            type="string",
            enum=("daily", "weekly", "monthly"),
        )

        assert param.enum == ("daily", "weekly", "monthly")

    def test_parameter_with_default(self):
        """Test parameter with default value."""
        param = MCPToolParameter(
            name="limit",
            description="Maximum results",
            type="integer",
            required=False,
            default=100,
        )

        assert param.required is False
        assert param.default == 100


class TestMCPToolDefinition:
    """Tests for MCPToolDefinition."""

    def test_basic_tool_definition(self):
        """Test creating a basic tool definition."""
        tool = MCPToolDefinition(
            name="get_price",
            description="Get current stock price",
        )

        assert tool.name == "get_price"
        assert tool.description == "Get current stock price"
        assert tool.parameters == ()

    def test_tool_with_parameters(self):
        """Test tool with parameters."""
        tool = MCPToolDefinition(
            name="get_price",
            description="Get stock price",
            parameters=(
                MCPToolParameter(
                    name="ric",
                    description="Instrument code",
                    type="string",
                    required=True,
                ),
                MCPToolParameter(
                    name="date",
                    description="Price date",
                    type="string",
                    required=False,
                ),
            ),
        )

        assert len(tool.parameters) == 2

    def test_to_json_schema(self):
        """Test JSON schema generation."""
        tool = MCPToolDefinition(
            name="get_data",
            description="Get data",
            parameters=(
                MCPToolParameter(
                    name="ric",
                    description="Instrument code",
                    type="string",
                    required=True,
                ),
                MCPToolParameter(
                    name="limit",
                    description="Max results",
                    type="integer",
                    required=False,
                    default=100,
                ),
            ),
        )

        schema = tool.to_json_schema()

        assert schema["type"] == "object"
        assert "ric" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["required"] == ["ric"]
        assert schema["properties"]["limit"]["default"] == 100

    def test_to_llm_tool_definition(self):
        """Test conversion to LLMToolDefinition."""
        tool = MCPToolDefinition(
            name="get_price",
            description="Get price data",
            parameters=(
                MCPToolParameter(
                    name="ric",
                    description="Instrument code",
                    type="string",
                    required=True,
                ),
            ),
        )

        llm_tool = tool.to_llm_tool_definition()

        assert llm_tool.name == "get_price"
        assert llm_tool.description == "Get price data"
        assert llm_tool.parameters["type"] == "object"
        assert "ric" in llm_tool.parameters["properties"]

    def test_from_mcp_response(self):
        """Test creating from MCP response."""
        response_data = {
            "name": "get_quote",
            "description": "Get stock quote",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol",
                    },
                    "include_extended": {
                        "type": "boolean",
                        "description": "Include extended hours",
                    },
                },
                "required": ["symbol"],
            },
        }

        tool = MCPToolDefinition.from_mcp_response(response_data)

        assert tool.name == "get_quote"
        assert tool.description == "Get stock quote"
        assert len(tool.parameters) == 2

        symbol_param = next(p for p in tool.parameters if p.name == "symbol")
        assert symbol_param.required is True

        extended_param = next(p for p in tool.parameters if p.name == "include_extended")
        assert extended_param.required is False


class TestMCPToolResult:
    """Tests for MCPToolResult."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = MCPToolResult(
            tool_name="get_price",
            content={"price": 150.25, "currency": "USD"},
            execution_time_ms=45,
        )

        assert result.tool_name == "get_price"
        assert result.content == {"price": 150.25, "currency": "USD"}
        assert result.is_error is False
        assert result.execution_time_ms == 45

    def test_error_result(self):
        """Test creating an error result."""
        result = MCPToolResult(
            tool_name="get_price",
            content=None,
            is_error=True,
            error_message="Symbol not found",
        )

        assert result.is_error is True
        assert result.error_message == "Symbol not found"

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = MCPToolResult(
            tool_name="get_data",
            content={"data": [1, 2, 3]},
            execution_time_ms=100,
        )

        d = result.to_dict()

        assert d["tool_name"] == "get_data"
        assert d["content"] == {"data": [1, 2, 3]}
        assert d["is_error"] is False
        assert d["execution_time_ms"] == 100

    def test_error_to_dict(self):
        """Test error result dictionary conversion."""
        result = MCPToolResult(
            tool_name="get_data",
            content=None,
            is_error=True,
            error_message="Failed",
        )

        d = result.to_dict()

        assert d["is_error"] is True
        assert d["error_message"] == "Failed"


class TestMCPServer:
    """Tests for MCPServer."""

    def test_basic_server(self):
        """Test creating a basic server config."""
        server = MCPServer(
            uri="mcp://datastream.lseg.com",
            name="datastream",
            description="LSEG Datastream API",
        )

        assert server.uri == "mcp://datastream.lseg.com"
        assert server.name == "datastream"
        assert server.timeout_seconds == 30

    def test_server_with_capabilities(self):
        """Test server with explicit capabilities."""
        caps = MCPCapabilities(tools=True, resources=True)
        server = MCPServer(
            uri="mcp://test.com",
            name="test",
            capabilities=caps,
        )

        assert server.has_tools is True
        assert server.has_resources is True

    def test_server_without_capabilities(self):
        """Test server with default capabilities."""
        server = MCPServer(uri="mcp://test.com", name="test")

        assert server.has_tools is False
        assert server.has_resources is False

    def test_server_with_auth(self):
        """Test server with API key."""
        server = MCPServer(
            uri="mcp://secure.com",
            name="secure",
            api_key="secret-key-123",
        )

        assert server.api_key == "secret-key-123"


class TestMCPCapabilities:
    """Tests for MCPCapabilities."""

    def test_default_capabilities(self):
        """Test default capabilities are all false."""
        caps = MCPCapabilities()

        assert caps.tools is False
        assert caps.resources is False
        assert caps.prompts is False
        assert caps.sampling is False
        assert caps.logging is False

    def test_custom_capabilities(self):
        """Test custom capabilities."""
        caps = MCPCapabilities(
            tools=True,
            resources=True,
            prompts=True,
        )

        assert caps.tools is True
        assert caps.resources is True
        assert caps.prompts is True
        assert caps.sampling is False
