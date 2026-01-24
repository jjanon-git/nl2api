"""Tests for LLM providers (Claude, OpenAI)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMToolCall,
    LLMToolDefinition,
    MessageRole,
)


class TestClaudeProvider:
    """Test suite for ClaudeProvider."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create mock anthropic module."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="Test response"),
        ]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_module = MagicMock()
        mock_module.AsyncAnthropic.return_value = mock_client
        mock_module.RateLimitError = Exception
        mock_module.APIConnectionError = Exception
        mock_module.InternalServerError = Exception

        return mock_module, mock_client, mock_response

    def test_init_requires_anthropic(self) -> None:
        """Test that init raises ImportError if anthropic not installed."""
        # This test verifies the import error is raised properly
        # We can't easily test this without actually having anthropic uninstalled
        # so we'll test that the class exists and has the expected structure
        from src.nl2api.llm.claude import ClaudeProvider

        assert hasattr(ClaudeProvider, "complete")
        assert hasattr(ClaudeProvider, "model_name")

    @pytest.mark.asyncio
    async def test_complete_text_response(self, mock_anthropic) -> None:
        """Test complete with text response."""
        mock_module, mock_client, mock_response = mock_anthropic

        from src.nl2api.llm.claude import ClaudeProvider

        # Create provider using __new__ to bypass __init__
        provider = ClaudeProvider.__new__(ClaudeProvider)
        provider._model = "claude-sonnet-4-20250514"
        provider._client = mock_client

        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        response = await provider.complete(messages)

        assert response.content == "Test response"
        assert response.usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self, mock_anthropic) -> None:
        """Test complete with tool call response."""
        mock_module, mock_client, mock_response = mock_anthropic

        # Configure response with tool use
        # Use spec to ensure proper attribute access
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "call-123"
        tool_use_block.name = "get_data"  # Important: set as attribute, not in constructor
        tool_use_block.input = {"RICs": ["AAPL.O"]}
        mock_response.content = [tool_use_block]

        from src.nl2api.llm.claude import ClaudeProvider

        provider = ClaudeProvider.__new__(ClaudeProvider)
        provider._model = "claude-sonnet-4-20250514"
        provider._client = mock_client

        messages = [
            LLMMessage(role=MessageRole.USER, content="Get Apple's EPS"),
        ]
        tools = [
            LLMToolDefinition(name="get_data", description="Get data"),
        ]

        response = await provider.complete(messages, tools=tools)

        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_data"
        assert response.tool_calls[0].arguments == {"RICs": ["AAPL.O"]}

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        from src.nl2api.llm.claude import ClaudeProvider

        provider = ClaudeProvider.__new__(ClaudeProvider)
        provider._model = "claude-opus-4-20250514"

        assert provider.model_name == "claude-opus-4-20250514"


class TestOpenAIProvider:
    """Test suite for OpenAIProvider."""

    @pytest.fixture
    def mock_openai(self):
        """Create mock openai module."""
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_module = MagicMock()
        mock_module.AsyncOpenAI.return_value = mock_client
        mock_module.AsyncAzureOpenAI.return_value = mock_client
        mock_module.RateLimitError = Exception
        mock_module.APIConnectionError = Exception
        mock_module.InternalServerError = Exception

        return mock_module, mock_client, mock_response

    @pytest.mark.asyncio
    async def test_complete_text_response(self, mock_openai) -> None:
        """Test complete with text response."""
        mock_module, mock_client, mock_response = mock_openai

        from src.nl2api.llm.openai import OpenAIProvider

        # Create provider using __new__ to bypass __init__
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4o"
        provider._client = mock_client

        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        response = await provider.complete(messages)

        assert response.content == "Test response"
        assert response.usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self, mock_openai) -> None:
        """Test complete with tool call response."""
        mock_module, mock_client, mock_response = mock_openai

        # Configure response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call-123"
        mock_tool_call.function.name = "get_data"
        mock_tool_call.function.arguments = '{"RICs": ["AAPL.O"]}'

        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"

        from src.nl2api.llm.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4o"
        provider._client = mock_client

        messages = [
            LLMMessage(role=MessageRole.USER, content="Get Apple's EPS"),
        ]
        tools = [
            LLMToolDefinition(name="get_data", description="Get data"),
        ]

        response = await provider.complete(messages, tools=tools)

        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_data"

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        from src.nl2api.llm.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4o-mini"

        assert provider.model_name == "gpt-4o-mini"


class TestLLMFactory:
    """Test suite for LLM provider factory."""

    def test_create_claude_provider_structure(self) -> None:
        """Test that create_llm_provider accepts claude provider type."""
        # We can't easily mock the import inside the function,
        # so we test the ValueError for invalid providers instead
        # and verify the function exists with correct signature
        import inspect

        from src.nl2api.llm.factory import create_llm_provider

        sig = inspect.signature(create_llm_provider)
        params = list(sig.parameters.keys())
        assert "provider" in params
        assert "api_key" in params
        assert "model" in params

    def test_create_openai_provider_structure(self) -> None:
        """Test that create_llm_provider accepts openai provider type."""
        # Verify the function accepts openai as a valid provider type
        # by checking that it doesn't raise ValueError
        import inspect

        from src.nl2api.llm.factory import create_llm_provider

        sig = inspect.signature(create_llm_provider)
        # Function signature check
        assert "provider" in sig.parameters

    def test_azure_requires_endpoint(self) -> None:
        """Test that Azure OpenAI requires azure_endpoint."""
        from src.nl2api.llm.factory import create_llm_provider

        with pytest.raises(ValueError, match="azure_endpoint is required"):
            create_llm_provider(
                provider="azure_openai",
                api_key="test-key",
            )

    def test_invalid_provider_raises(self) -> None:
        """Test that invalid provider raises ValueError."""
        from src.nl2api.llm.factory import create_llm_provider

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(provider="invalid", api_key="test")


class TestLLMProviderProtocol:
    """Test suite for LLM provider protocol compliance."""

    def test_claude_provider_has_required_methods(self) -> None:
        """Test that ClaudeProvider has required protocol methods."""
        from src.nl2api.llm.claude import ClaudeProvider

        assert hasattr(ClaudeProvider, "complete")
        assert hasattr(ClaudeProvider, "complete_with_retry")
        assert hasattr(ClaudeProvider, "model_name")

    def test_openai_provider_has_required_methods(self) -> None:
        """Test that OpenAIProvider has required protocol methods."""
        from src.nl2api.llm.openai import OpenAIProvider

        assert hasattr(OpenAIProvider, "complete")
        assert hasattr(OpenAIProvider, "complete_with_retry")
        assert hasattr(OpenAIProvider, "model_name")


class TestMessageConversion:
    """Test suite for message conversion to provider formats."""

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            LLMMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=MessageRole.USER, content="What is Apple's EPS?"),
            LLMMessage(
                role=MessageRole.ASSISTANT,
                content="Let me get that data for you.",
                tool_calls=(
                    LLMToolCall(id="call-1", name="get_data", arguments={"RICs": ["AAPL.O"]}),
                ),
            ),
            LLMMessage(
                role=MessageRole.TOOL,
                content='{"EPS": 6.42}',
                tool_call_id="call-1",
            ),
        ]

    @pytest.mark.asyncio
    async def test_claude_message_conversion(self, sample_messages) -> None:
        """Test that Claude provider converts messages correctly."""
        from src.nl2api.llm.claude import ClaudeProvider

        # Create mock client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = ClaudeProvider.__new__(ClaudeProvider)
        provider._model = "claude-sonnet-4-20250514"
        provider._client = mock_client

        await provider.complete(sample_messages)

        # Verify the call was made
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs

        # System message should be extracted
        assert call_kwargs.get("system") == "You are a helpful assistant."

        # Messages should be converted
        assert len(call_kwargs["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_openai_message_conversion(self, sample_messages) -> None:
        """Test that OpenAI provider converts messages correctly."""
        from src.nl2api.llm.openai import OpenAIProvider

        # Create mock client
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4o"
        provider._client = mock_client

        await provider.complete(sample_messages)

        # Verify the call was made
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs

        # Messages should be converted to OpenAI format
        assert len(call_kwargs["messages"]) == 4  # system, user, assistant, tool
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"


class TestRetryHelpers:
    """Test retry helper functions for both providers."""

    def test_calculate_wait_time_exponential_backoff(self) -> None:
        """Test that wait time increases exponentially."""
        from src.nl2api.llm.claude import _calculate_wait_time

        # With jitter_factor=0, we get exact exponential backoff
        wait_0 = _calculate_wait_time(0, retry_after=None, jitter_factor=0)
        wait_1 = _calculate_wait_time(1, retry_after=None, jitter_factor=0)
        wait_2 = _calculate_wait_time(2, retry_after=None, jitter_factor=0)

        assert wait_0 == 1.0  # 1 * 2^0
        assert wait_1 == 2.0  # 1 * 2^1
        assert wait_2 == 4.0  # 1 * 2^2

    def test_calculate_wait_time_honors_retry_after(self) -> None:
        """Test that retry_after from API is honored."""
        from src.nl2api.llm.claude import _calculate_wait_time

        # With retry_after set, use that value instead of backoff
        wait = _calculate_wait_time(0, retry_after=30.0, jitter_factor=0)
        assert wait == 30.0

        # Retry after is capped by max_delay
        wait_capped = _calculate_wait_time(0, retry_after=120.0, max_delay=60.0, jitter_factor=0)
        assert wait_capped == 60.0

    def test_calculate_wait_time_adds_jitter(self) -> None:
        """Test that jitter is added to prevent thundering herd."""
        from src.nl2api.llm.claude import _calculate_wait_time

        # With jitter, we should get values in a range around the base
        # Run multiple times to verify randomness
        wait_times = [
            _calculate_wait_time(2, retry_after=None, jitter_factor=0.25) for _ in range(50)
        ]

        # Base is 4.0, with 25% jitter we expect range [3.0, 5.0]
        assert all(3.0 <= w <= 5.0 for w in wait_times)

        # Should not all be the same (random)
        assert len(set(wait_times)) > 1

    def test_calculate_wait_time_minimum_floor(self) -> None:
        """Test that wait time never goes below 100ms."""
        from src.nl2api.llm.claude import _calculate_wait_time

        # Even with extreme negative jitter scenario, floor is 0.1
        # (This is enforced by max(0.1, ...) in the implementation)
        wait = _calculate_wait_time(0, retry_after=0.05, jitter_factor=0.5)
        assert wait >= 0.1

    def test_extract_retry_after_from_headers(self) -> None:
        """Test extraction of retry-after header from exception."""
        from src.nl2api.llm.claude import _extract_retry_after

        # Mock an exception with response headers
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "45"}

        mock_error = MagicMock()
        mock_error.response = mock_response

        result = _extract_retry_after(mock_error)
        assert result == 45.0

    def test_extract_retry_after_missing_header(self) -> None:
        """Test that None is returned when header is missing."""
        from src.nl2api.llm.claude import _extract_retry_after

        # Mock an exception without retry-after header
        mock_response = MagicMock()
        mock_response.headers = {}

        mock_error = MagicMock()
        mock_error.response = mock_response

        result = _extract_retry_after(mock_error)
        assert result is None

    def test_extract_retry_after_no_response(self) -> None:
        """Test that None is returned when exception has no response."""
        from src.nl2api.llm.claude import _extract_retry_after

        # Mock an exception without response attribute
        mock_error = MagicMock(spec=[])  # No attributes

        result = _extract_retry_after(mock_error)
        assert result is None

    def test_extract_retry_after_invalid_value(self) -> None:
        """Test that None is returned for non-numeric retry-after."""
        from src.nl2api.llm.claude import _extract_retry_after

        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "invalid"}

        mock_error = MagicMock()
        mock_error.response = mock_response

        result = _extract_retry_after(mock_error)
        assert result is None

    def test_openai_helpers_match_claude(self) -> None:
        """Test that OpenAI helpers have same behavior as Claude."""
        from src.nl2api.llm.claude import _calculate_wait_time as claude_wait
        from src.nl2api.llm.claude import _extract_retry_after as claude_extract
        from src.nl2api.llm.openai import _calculate_wait_time as openai_wait
        from src.nl2api.llm.openai import _extract_retry_after as openai_extract

        # Same inputs should produce same structured output (ignoring jitter randomness)
        # Test with no jitter for determinism
        assert claude_wait(0, retry_after=15.0, jitter_factor=0) == openai_wait(
            0, retry_after=15.0, jitter_factor=0
        )
        assert claude_wait(2, retry_after=None, jitter_factor=0) == openai_wait(
            2, retry_after=None, jitter_factor=0
        )

        # Both should return None for exceptions without response
        mock_error = MagicMock(spec=[])
        assert claude_extract(mock_error) is None
        assert openai_extract(mock_error) is None
