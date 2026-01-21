"""
Tests for EscalatingLLMRouter

Tests the model escalation functionality including:
- Single tier routing
- Escalation on low confidence
- Maximum escalation limits
- Cache integration
"""

import pytest
from typing import Any

from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from src.nl2api.routing.escalating_router import (
    EscalatingLLMRouter,
    ModelTier,
    create_escalating_router,
)
from src.nl2api.routing.cache import InMemoryRoutingCache
from src.nl2api.routing.protocols import RouterResult


class MockLLMProvider:
    """Mock LLM provider for testing escalation."""

    def __init__(
        self,
        model_name: str,
        confidence: float = 0.85,
        domain: str = "datastream",
    ):
        self._model_name = model_name
        self._confidence = confidence
        self._domain = domain
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content="",
            tool_calls=(
                LLMToolCall(
                    id=f"call_{self.call_count}",
                    name=f"route_to_{self._domain}",
                    arguments={
                        "reasoning": f"Routed by {self._model_name}",
                        "confidence": self._confidence,
                    },
                ),
            ),
        )


class MockToolProvider:
    """Mock tool provider for testing."""

    def __init__(self, name: str):
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def provider_description(self) -> str:
        return f"Mock {self._name} provider"

    async def list_tools(self) -> list[LLMToolDefinition]:
        return []

    async def get_tool_description(self, tool_name: str) -> str | None:
        return None


@pytest.fixture
def providers() -> list[MockToolProvider]:
    """Create mock providers."""
    return [
        MockToolProvider("datastream"),
        MockToolProvider("estimates"),
        MockToolProvider("fundamentals"),
    ]


class TestEscalatingLLMRouter:
    """Tests for EscalatingLLMRouter."""

    @pytest.mark.asyncio
    async def test_uses_first_tier_when_confidence_sufficient(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that first tier is used when confidence is above threshold."""
        tier1 = MockLLMProvider("haiku", confidence=0.9)
        tier2 = MockLLMProvider("sonnet", confidence=0.95)

        router = EscalatingLLMRouter(
            model_tiers=[
                ModelTier(name="haiku", model=tier1),
                ModelTier(name="sonnet", model=tier2),
            ],
            tool_providers=providers,
            escalation_threshold=0.7,
        )

        result = await router.route("What is Apple's stock price?")

        assert result.domain == "datastream"
        assert result.confidence == 0.9
        assert result.model_used == "haiku"
        assert tier1.call_count == 1
        assert tier2.call_count == 0  # Should not escalate

    @pytest.mark.asyncio
    async def test_escalates_on_low_confidence(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that router escalates when confidence is below threshold."""
        tier1 = MockLLMProvider("haiku", confidence=0.5)  # Below threshold
        tier2 = MockLLMProvider("sonnet", confidence=0.9)

        router = EscalatingLLMRouter(
            model_tiers=[
                ModelTier(name="haiku", model=tier1),
                ModelTier(name="sonnet", model=tier2),
            ],
            tool_providers=providers,
            escalation_threshold=0.7,
        )

        result = await router.route("What is Apple's stock price?")

        assert result.confidence == 0.9  # From tier2
        assert result.model_used == "sonnet"
        assert tier1.call_count == 1
        assert tier2.call_count == 1

    @pytest.mark.asyncio
    async def test_respects_max_escalations(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that router respects max_escalations limit."""
        tier1 = MockLLMProvider("haiku", confidence=0.3)
        tier2 = MockLLMProvider("sonnet", confidence=0.4)
        tier3 = MockLLMProvider("opus", confidence=0.5)

        router = EscalatingLLMRouter(
            model_tiers=[
                ModelTier(name="haiku", model=tier1),
                ModelTier(name="sonnet", model=tier2),
                ModelTier(name="opus", model=tier3),
            ],
            tool_providers=providers,
            escalation_threshold=0.7,
            max_escalations=1,  # Only allow 1 escalation
        )

        result = await router.route("What is Apple's stock price?")

        # Should stop at tier2 due to max_escalations=1
        assert tier1.call_count == 1
        assert tier2.call_count == 1
        assert tier3.call_count == 0

    @pytest.mark.asyncio
    async def test_returns_best_result_when_all_tiers_low_confidence(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that best result is returned when all tiers have low confidence."""
        tier1 = MockLLMProvider("haiku", confidence=0.3)
        tier2 = MockLLMProvider("sonnet", confidence=0.5)  # Best

        router = EscalatingLLMRouter(
            model_tiers=[
                ModelTier(name="haiku", model=tier1),
                ModelTier(name="sonnet", model=tier2),
            ],
            tool_providers=providers,
            escalation_threshold=0.7,
        )

        result = await router.route("Ambiguous query")

        assert result.confidence == 0.5  # Best from tier2
        assert result.model_used == "sonnet"

    @pytest.mark.asyncio
    async def test_caches_result(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that results are cached after routing."""
        tier1 = MockLLMProvider("haiku", confidence=0.9)
        cache = InMemoryRoutingCache()

        router = EscalatingLLMRouter(
            model_tiers=[ModelTier(name="haiku", model=tier1)],
            tool_providers=providers,
            cache=cache,
        )

        # First call
        result1 = await router.route("What is Apple's stock price?")
        assert not result1.cached
        assert tier1.call_count == 1

        # Second call - should hit cache
        result2 = await router.route("What is Apple's stock price?")
        assert result2.cached
        assert tier1.call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_single_tier_works(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that single tier configuration works."""
        tier1 = MockLLMProvider("haiku", confidence=0.85)

        router = EscalatingLLMRouter(
            model_tiers=[ModelTier(name="haiku", model=tier1)],
            tool_providers=providers,
        )

        result = await router.route("What is Apple's stock price?")

        assert result.domain == "datastream"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_handles_tier_error_gracefully(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that errors in a tier are handled gracefully."""

        class ErrorLLM:
            @property
            def model_name(self) -> str:
                return "error-model"

            async def complete(self, **kwargs) -> LLMResponse:
                raise RuntimeError("API error")

        tier1 = ErrorLLM()
        tier2 = MockLLMProvider("sonnet", confidence=0.9)

        router = EscalatingLLMRouter(
            model_tiers=[
                ModelTier(name="error", model=tier1),
                ModelTier(name="sonnet", model=tier2),
            ],
            tool_providers=providers,
            escalation_threshold=0.7,
        )

        result = await router.route("What is Apple's stock price?")

        # Should fall through to tier2
        assert result.domain == "datastream"
        assert result.confidence == 0.9

    def test_requires_at_least_one_tier(
        self,
        providers: list[MockToolProvider],
    ):
        """Test that at least one tier is required."""
        with pytest.raises(ValueError, match="At least one model tier is required"):
            EscalatingLLMRouter(
                model_tiers=[],
                tool_providers=providers,
            )


class TestCreateEscalatingRouter:
    """Tests for the create_escalating_router factory function."""

    @pytest.mark.asyncio
    async def test_creates_single_tier_router(
        self,
        providers: list[MockToolProvider],
    ):
        """Test creating router with single tier."""
        tier1 = MockLLMProvider("haiku", confidence=0.9)

        router = create_escalating_router(
            tier1_model=tier1,
            tool_providers=providers,
        )

        result = await router.route("Test query")

        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_creates_two_tier_router(
        self,
        providers: list[MockToolProvider],
    ):
        """Test creating router with two tiers."""
        tier1 = MockLLMProvider("haiku", confidence=0.5)
        tier2 = MockLLMProvider("sonnet", confidence=0.9)

        router = create_escalating_router(
            tier1_model=tier1,
            tier2_model=tier2,
            tool_providers=providers,
            escalation_threshold=0.7,
        )

        result = await router.route("Test query")

        assert result.confidence == 0.9  # From tier2
        assert tier1.call_count == 1
        assert tier2.call_count == 1

    @pytest.mark.asyncio
    async def test_creates_three_tier_router(
        self,
        providers: list[MockToolProvider],
    ):
        """Test creating router with three tiers."""
        tier1 = MockLLMProvider("haiku", confidence=0.3)
        tier2 = MockLLMProvider("sonnet", confidence=0.5)
        tier3 = MockLLMProvider("opus", confidence=0.95)

        router = create_escalating_router(
            tier1_model=tier1,
            tier2_model=tier2,
            tier3_model=tier3,
            tool_providers=providers,
            escalation_threshold=0.7,
        )

        result = await router.route("Complex query")

        assert result.confidence == 0.95  # From tier3
        assert tier1.call_count == 1
        assert tier2.call_count == 1
        assert tier3.call_count == 1
