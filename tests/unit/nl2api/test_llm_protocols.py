"""Tests for LLM protocols and data classes."""

from __future__ import annotations

import pytest

from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    MessageRole,
)


class TestMessageRole:
    """Test suite for MessageRole enum."""

    def test_all_roles_defined(self) -> None:
        """Test all expected roles are defined."""
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_role_string_values(self) -> None:
        """Test roles are string enums."""
        assert isinstance(MessageRole.SYSTEM.value, str)
        assert MessageRole.USER.value == "user"


class TestLLMToolDefinition:
    """Test suite for LLMToolDefinition."""

    def test_basic_creation(self) -> None:
        """Test basic tool definition creation."""
        tool = LLMToolDefinition(
            name="get_data",
            description="Fetch financial data",
        )
        assert tool.name == "get_data"
        assert tool.description == "Fetch financial data"
        assert tool.parameters == {}

    def test_with_parameters(self) -> None:
        """Test tool definition with JSON schema parameters."""
        params = {
            "type": "object",
            "properties": {
                "RICs": {"type": "array", "items": {"type": "string"}},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["RICs", "fields"],
        }
        tool = LLMToolDefinition(
            name="get_data",
            description="Fetch data",
            parameters=params,
        )
        assert tool.parameters == params
        assert "RICs" in tool.parameters["properties"]

    def test_to_openai_format(self) -> None:
        """Test conversion to OpenAI function format."""
        tool = LLMToolDefinition(
            name="search",
            description="Search for items",
            parameters={"type": "object", "properties": {}},
        )
        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "search"
        assert openai_format["function"]["description"] == "Search for items"
        assert openai_format["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_to_anthropic_format(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = LLMToolDefinition(
            name="search",
            description="Search for items",
            parameters={"type": "object", "properties": {}},
        )
        anthropic_format = tool.to_anthropic_format()

        assert anthropic_format["name"] == "search"
        assert anthropic_format["description"] == "Search for items"
        assert anthropic_format["input_schema"] == {"type": "object", "properties": {}}

    def test_frozen_dataclass(self) -> None:
        """Test that LLMToolDefinition is frozen/immutable."""
        tool = LLMToolDefinition(name="test", description="test")
        with pytest.raises(AttributeError):
            tool.name = "changed"


class TestLLMToolCall:
    """Test suite for LLMToolCall."""

    def test_basic_creation(self) -> None:
        """Test basic tool call creation."""
        call = LLMToolCall(
            id="call-123",
            name="get_data",
            arguments={"RICs": ["AAPL.O"]},
        )
        assert call.id == "call-123"
        assert call.name == "get_data"
        assert call.arguments == {"RICs": ["AAPL.O"]}

    def test_default_arguments(self) -> None:
        """Test default empty arguments."""
        call = LLMToolCall(id="call-1", name="test")
        assert call.arguments == {}

    def test_frozen_dataclass(self) -> None:
        """Test that LLMToolCall is frozen/immutable."""
        call = LLMToolCall(id="1", name="test")
        with pytest.raises(AttributeError):
            call.name = "changed"


class TestLLMMessage:
    """Test suite for LLMMessage."""

    def test_system_message(self) -> None:
        """Test creating a system message."""
        msg = LLMMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant.",
        )
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a helpful assistant."

    def test_user_message(self) -> None:
        """Test creating a user message."""
        msg = LLMMessage(
            role=MessageRole.USER,
            content="What is Apple's EPS?",
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "What is Apple's EPS?"

    def test_assistant_message_with_tool_calls(self) -> None:
        """Test assistant message with tool calls."""
        tool_calls = (
            LLMToolCall(id="1", name="get_data", arguments={"x": 1}),
            LLMToolCall(id="2", name="search", arguments={"q": "test"}),
        )
        msg = LLMMessage(
            role=MessageRole.ASSISTANT,
            content="Let me get that data.",
            tool_calls=tool_calls,
        )
        assert msg.role == MessageRole.ASSISTANT
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].name == "get_data"

    def test_tool_result_message(self) -> None:
        """Test tool result message."""
        msg = LLMMessage(
            role=MessageRole.TOOL,
            content='{"result": "success"}',
            tool_call_id="call-123",
            name="get_data",
        )
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call-123"
        assert msg.name == "get_data"

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        msg = LLMMessage(role=MessageRole.USER)
        assert msg.content == ""
        assert msg.tool_calls == ()
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_frozen_dataclass(self) -> None:
        """Test that LLMMessage is frozen/immutable."""
        msg = LLMMessage(role=MessageRole.USER, content="test")
        with pytest.raises(AttributeError):
            msg.content = "changed"


class TestLLMResponse:
    """Test suite for LLMResponse."""

    def test_text_response(self) -> None:
        """Test simple text response."""
        response = LLMResponse(
            content="The EPS estimate is $5.23",
            finish_reason="stop",
        )
        assert response.content == "The EPS estimate is $5.23"
        assert response.finish_reason == "stop"
        assert not response.has_tool_calls

    def test_response_with_tool_calls(self) -> None:
        """Test response with tool calls."""
        tool_calls = (
            LLMToolCall(
                id="call-1",
                name="get_data",
                arguments={"RICs": ["AAPL.O"], "fields": ["TR.EPSMean"]},
            ),
        )
        response = LLMResponse(
            content="",
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_data"

    def test_has_tool_calls_property(self) -> None:
        """Test has_tool_calls property."""
        no_tools = LLMResponse(content="test")
        assert not no_tools.has_tool_calls

        with_tools = LLMResponse(tool_calls=(LLMToolCall(id="1", name="test"),))
        assert with_tools.has_tool_calls

    def test_usage_tracking(self) -> None:
        """Test usage/token tracking."""
        response = LLMResponse(
            content="test",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )
        assert response.usage["total_tokens"] == 150
        assert response.usage["prompt_tokens"] == 100

    def test_default_values(self) -> None:
        """Test default values."""
        response = LLMResponse()
        assert response.content == ""
        assert response.tool_calls == ()
        assert response.finish_reason == "stop"
        assert response.usage == {}

    def test_frozen_dataclass(self) -> None:
        """Test that LLMResponse is frozen/immutable."""
        response = LLMResponse(content="test")
        with pytest.raises(AttributeError):
            response.content = "changed"


class TestLLMProviderProtocol:
    """Test suite for LLMProvider protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that LLMProvider is runtime checkable."""
        from src.nl2api.llm.protocols import LLMProvider

        # Create a minimal implementation
        class MinimalProvider:
            @property
            def model_name(self) -> str:
                return "test-model"

            async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
                return LLMResponse(content="test")

            async def complete_with_retry(
                self, messages, tools=None, temperature=0.0, max_tokens=4096, max_retries=3
            ):
                return LLMResponse(content="test")

        provider = MinimalProvider()
        assert isinstance(provider, LLMProvider)

    def test_non_compliant_class_fails_check(self) -> None:
        """Test that non-compliant class fails isinstance check."""
        from src.nl2api.llm.protocols import LLMProvider

        class NotAProvider:
            pass

        not_provider = NotAProvider()
        assert not isinstance(not_provider, LLMProvider)
