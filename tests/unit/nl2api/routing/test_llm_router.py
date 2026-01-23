"""
Tests for LLMToolRouter

Tests the FM-first routing implementation including:
- Tool-based domain selection
- Confidence scoring
- Cache integration
- Fallback behavior
"""

from typing import Any

import pytest

from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from src.nl2api.routing.cache import InMemoryRoutingCache
from src.nl2api.routing.llm_router import LLMToolRouter
from src.nl2api.routing.protocols import RouterResult


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = responses or []
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
            return response

        # Default: route to first tool
        if tools:
            return LLMResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="call_1",
                        name=tools[0].name,
                        arguments={"reasoning": "Default routing", "confidence": 0.85},
                    ),
                ),
            )
        return LLMResponse(content="unknown")


class MockToolProvider:
    """Mock tool provider for testing."""

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: tuple[str, ...] = (),
        example_queries: tuple[str, ...] = (),
    ):
        self._name = name
        self._description = description
        self._capabilities = capabilities
        self._example_queries = example_queries

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def provider_description(self) -> str:
        return self._description

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._capabilities

    @property
    def example_queries(self) -> tuple[str, ...]:
        return self._example_queries

    async def list_tools(self) -> list[LLMToolDefinition]:
        return [
            LLMToolDefinition(
                name=f"{self._name}.get_data",
                description=f"Get data from {self._name}",
                parameters={"type": "object", "properties": {}},
            )
        ]

    async def get_tool_description(self, tool_name: str) -> str | None:
        return self._description if tool_name == f"{self._name}.get_data" else None


@pytest.fixture
def datastream_provider() -> MockToolProvider:
    """Create mock datastream provider."""
    return MockToolProvider(
        name="datastream",
        description="Stock prices, market data, historical time series",
        capabilities=("stock prices", "market cap", "trading volume"),
        example_queries=("What is Apple's stock price?",),
    )


@pytest.fixture
def estimates_provider() -> MockToolProvider:
    """Create mock estimates provider."""
    return MockToolProvider(
        name="estimates",
        description="Analyst forecasts, EPS estimates, recommendations",
        capabilities=("EPS estimates", "analyst recommendations"),
        example_queries=("What are the EPS estimates for Tesla?",),
    )


@pytest.fixture
def fundamentals_provider() -> MockToolProvider:
    """Create mock fundamentals provider."""
    return MockToolProvider(
        name="fundamentals",
        description="Income statement, balance sheet, financial ratios",
        capabilities=("revenue", "net income", "financial ratios"),
        example_queries=("What is Microsoft's revenue?",),
    )


class TestLLMToolRouter:
    """Tests for LLMToolRouter."""

    @pytest.mark.asyncio
    async def test_routes_to_correct_domain(
        self,
        datastream_provider: MockToolProvider,
        estimates_provider: MockToolProvider,
    ):
        """Test that router selects correct domain based on LLM tool call."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall(
                            id="call_1",
                            name="route_to_datastream",
                            arguments={"reasoning": "Query about stock price", "confidence": 0.95},
                        ),
                    ),
                )
            ]
        )

        router = LLMToolRouter(
            llm=llm,
            tool_providers=[datastream_provider, estimates_provider],
        )

        result = await router.route("What is Apple's stock price?")

        assert result.domain == "datastream"
        assert result.confidence == 0.95
        assert result.reasoning == "Query about stock price"
        assert not result.cached

    @pytest.mark.asyncio
    async def test_uses_default_confidence_when_not_provided(
        self,
        datastream_provider: MockToolProvider,
    ):
        """Test that default confidence is used when LLM doesn't provide one."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall(
                            id="call_1",
                            name="route_to_datastream",
                            arguments={"reasoning": "Stock price query"},
                        ),
                    ),
                )
            ]
        )

        router = LLMToolRouter(
            llm=llm,
            tool_providers=[datastream_provider],
            default_confidence=0.8,
        )

        result = await router.route("What is Apple's price?")

        assert result.domain == "datastream"
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(
        self,
        datastream_provider: MockToolProvider,
    ):
        """Test that cache hits return cached results without LLM call."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall(
                            id="call_1",
                            name="route_to_datastream",
                            arguments={"reasoning": "Stock price", "confidence": 0.9},
                        ),
                    ),
                )
            ]
        )

        cache = InMemoryRoutingCache()
        router = LLMToolRouter(
            llm=llm,
            tool_providers=[datastream_provider],
            cache=cache,
        )

        # First call - should hit LLM
        result1 = await router.route("What is Apple's stock price?")
        assert not result1.cached
        assert len(llm.calls) == 1

        # Second call - should hit cache
        result2 = await router.route("What is Apple's stock price?")
        assert result2.cached
        assert result2.domain == "datastream"
        assert len(llm.calls) == 1  # No additional LLM call

    @pytest.mark.asyncio
    async def test_returns_unknown_domain_on_llm_error(
        self,
        datastream_provider: MockToolProvider,
    ):
        """Test that LLM errors result in unknown domain with zero confidence."""

        class ErrorLLM:
            @property
            def model_name(self) -> str:
                return "error-model"

            async def complete(self, **kwargs) -> LLMResponse:
                raise RuntimeError("LLM API error")

        router = LLMToolRouter(
            llm=ErrorLLM(),
            tool_providers=[datastream_provider],
        )

        result = await router.route("What is Apple's price?")

        assert result.domain == "unknown"
        assert result.confidence == 0.0
        assert "error" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_fallback_parsing_from_content(
        self,
        datastream_provider: MockToolProvider,
    ):
        """Test fallback to content parsing when no tool call."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content="I think this should go to datastream",
                    tool_calls=(),
                )
            ]
        )

        router = LLMToolRouter(
            llm=llm,
            tool_providers=[datastream_provider],
        )

        result = await router.route("What is Apple's price?")

        assert result.domain == "datastream"
        assert result.confidence == 0.5  # Lower confidence for fallback

    @pytest.mark.asyncio
    async def test_builds_routing_tools_from_providers(
        self,
        datastream_provider: MockToolProvider,
        estimates_provider: MockToolProvider,
        fundamentals_provider: MockToolProvider,
    ):
        """Test that routing tools are correctly built from providers."""
        llm = MockLLMProvider()

        router = LLMToolRouter(
            llm=llm,
            tool_providers=[
                datastream_provider,
                estimates_provider,
                fundamentals_provider,
            ],
        )

        # Route a query to trigger LLM call
        await router.route("Test query")

        # Check tools passed to LLM
        assert len(llm.calls) == 1
        tools = llm.calls[0]["tools"]
        assert len(tools) == 3

        tool_names = [t.name for t in tools]
        assert "route_to_datastream" in tool_names
        assert "route_to_estimates" in tool_names
        assert "route_to_fundamentals" in tool_names

    @pytest.mark.asyncio
    async def test_includes_capabilities_in_tool_description(
        self,
        datastream_provider: MockToolProvider,
    ):
        """Test that capabilities are included in routing tool descriptions."""
        llm = MockLLMProvider()

        router = LLMToolRouter(
            llm=llm,
            tool_providers=[datastream_provider],
        )

        await router.route("Test query")

        tools = llm.calls[0]["tools"]
        assert len(tools) == 1

        # Description should include capabilities
        description = tools[0].description
        assert "stock prices" in description.lower()

    @pytest.mark.asyncio
    async def test_records_latency(
        self,
        datastream_provider: MockToolProvider,
    ):
        """Test that latency is recorded in result."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall(
                            id="call_1",
                            name="route_to_datastream",
                            arguments={"reasoning": "Stock price", "confidence": 0.9},
                        ),
                    ),
                )
            ]
        )

        router = LLMToolRouter(
            llm=llm,
            tool_providers=[datastream_provider],
        )

        result = await router.route("What is Apple's price?")

        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_records_model_used(
        self,
        datastream_provider: MockToolProvider,
    ):
        """Test that model name is recorded in result."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall(
                            id="call_1",
                            name="route_to_datastream",
                            arguments={"reasoning": "Stock price", "confidence": 0.9},
                        ),
                    ),
                )
            ]
        )

        router = LLMToolRouter(
            llm=llm,
            tool_providers=[datastream_provider],
        )

        result = await router.route("What is Apple's price?")

        assert result.model_used == "mock-model"


class TestInMemoryRoutingCache:
    """Tests for InMemoryRoutingCache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = InMemoryRoutingCache()

        result = RouterResult(
            domain="datastream",
            confidence=0.9,
            reasoning="Test routing",
        )

        await cache.set("test query", result)
        cached = await cache.get("test query")

        assert cached is not None
        assert cached.domain == "datastream"
        assert cached.confidence == 0.9
        assert cached.cached is True

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self):
        """Test that get returns None for missing keys."""
        cache = InMemoryRoutingCache()

        cached = await cache.get("nonexistent query")

        assert cached is None

    @pytest.mark.asyncio
    async def test_does_not_cache_unknown_domain(self):
        """Test that unknown domains are not cached."""
        cache = InMemoryRoutingCache()

        result = RouterResult(
            domain="unknown",
            confidence=0.0,
            reasoning="Failed routing",
        )

        await cache.set("test query", result)
        cached = await cache.get("test query")

        assert cached is None

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test cache invalidation."""
        cache = InMemoryRoutingCache()

        result = RouterResult(
            domain="datastream",
            confidence=0.9,
        )

        await cache.set("test query", result)
        await cache.invalidate("test query")
        cached = await cache.get("test query")

        assert cached is None

    @pytest.mark.asyncio
    async def test_clear_all(self):
        """Test clearing all cache entries."""
        cache = InMemoryRoutingCache()

        await cache.set("query1", RouterResult(domain="datastream", confidence=0.9))
        await cache.set("query2", RouterResult(domain="estimates", confidence=0.8))

        await cache.clear_all()

        assert await cache.get("query1") is None
        assert await cache.get("query2") is None
