"""
Escalating LLM Router

Routes queries with automatic model escalation for complex queries.
Starts with a fast/cheap model and escalates to more capable models
when confidence is low.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMProvider,
    LLMToolDefinition,
    MessageRole,
)
from src.nl2api.routing.llm_router import ROUTING_SYSTEM_PROMPT
from src.nl2api.routing.protocols import (
    RouterResult,
    RoutingToolDefinition,
    ToolProvider,
)

if TYPE_CHECKING:
    from src.nl2api.routing.cache import RoutingCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTier:
    """Configuration for a model tier in the escalation chain."""

    name: str
    model: LLMProvider
    max_tokens: int = 150
    temperature: float = 0.0


class EscalatingLLMRouter:
    """
    Routes queries with automatic model escalation.

    Uses a tiered approach:
    - Tier 1: Fast/cheap model (e.g., Claude Haiku, GPT-4o-mini)
    - Tier 2: Balanced model (e.g., Claude Sonnet, GPT-4o)
    - Tier 3: Most capable model (e.g., Claude Opus 4.5, GPT-5.1)

    Escalates to higher tiers when confidence is below threshold.
    """

    def __init__(
        self,
        model_tiers: list[ModelTier],
        tool_providers: list[ToolProvider],
        cache: RoutingCache | None = None,
        escalation_threshold: float = 0.7,
        max_escalations: int = 2,
        system_prompt: str | None = None,
        default_confidence: float = 0.85,
    ):
        """
        Initialize the escalating router.

        Args:
            model_tiers: List of model tiers, ordered by capability (cheapest first)
            tool_providers: List of tool providers (agents or MCP servers)
            cache: Optional routing cache
            escalation_threshold: Confidence threshold for escalation (default 0.7)
            max_escalations: Maximum number of escalation attempts (default 2)
            system_prompt: Optional custom system prompt
            default_confidence: Default confidence when LLM doesn't provide one
        """
        if not model_tiers:
            raise ValueError("At least one model tier is required")

        self._model_tiers = model_tiers
        self._providers = tool_providers
        self._cache = cache
        self._escalation_threshold = escalation_threshold
        self._max_escalations = min(max_escalations, len(model_tiers) - 1)
        self._system_prompt = system_prompt or ROUTING_SYSTEM_PROMPT
        self._default_confidence = default_confidence

        # Build routing tools from providers
        self._routing_tools = self._build_routing_tools()

        tier_names = [t.name for t in model_tiers]
        logger.info(
            f"EscalatingLLMRouter initialized with tiers: {tier_names}, "
            f"escalation_threshold={escalation_threshold}"
        )

    def _build_routing_tools(self) -> list[LLMToolDefinition]:
        """Convert tool providers to routing tools."""
        tools = []

        for provider in self._providers:
            routing_def = RoutingToolDefinition(
                name=f"route_to_{provider.provider_name}",
                description=provider.provider_description,
                capabilities=getattr(provider, "capabilities", ()),
                example_queries=getattr(provider, "example_queries", ()),
            )
            tools.append(routing_def.to_llm_tool())

        return tools

    def _get_provider_names(self) -> list[str]:
        """Get list of provider names."""
        return [p.provider_name for p in self._providers]

    async def route(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RouterResult:
        """
        Route a query with automatic model escalation.

        Tries cheaper models first, escalating to more capable models
        when confidence is below the threshold.

        Args:
            query: Natural language query to route
            context: Optional context

        Returns:
            RouterResult with domain, confidence, and escalation metadata
        """
        start_time = time.perf_counter()

        # Check cache first
        if self._cache:
            cached = await self._cache.get(query)
            if cached:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                logger.debug(f"Cache hit for query routing: {cached.domain}")
                return replace(cached, cached=True, latency_ms=latency_ms)

        # Build messages once
        messages = self._build_messages(query, context)

        # Try each tier until we get sufficient confidence
        best_result: RouterResult | None = None
        escalation_count = 0

        for i, tier in enumerate(self._model_tiers):
            if escalation_count > self._max_escalations:
                break

            tier_start = time.perf_counter()

            result = await self._route_with_tier(tier, messages)

            tier_latency = int((time.perf_counter() - tier_start) * 1000)
            logger.info(
                f"Tier {tier.name}: domain={result.domain}, "
                f"confidence={result.confidence:.2f}, latency={tier_latency}ms"
            )

            # Update best result
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            # Check if confidence is sufficient
            if result.confidence >= self._escalation_threshold:
                logger.info(f"Routing complete at tier {tier.name}")
                break

            # Escalate to next tier
            if i < len(self._model_tiers) - 1:
                next_tier = self._model_tiers[i + 1]
                logger.info(
                    f"Escalating from {tier.name} to {next_tier.name} "
                    f"(confidence {result.confidence:.2f} < {self._escalation_threshold})"
                )
                escalation_count += 1

        # Calculate total latency
        total_latency_ms = int((time.perf_counter() - start_time) * 1000)

        if best_result:
            final_result = replace(
                best_result,
                latency_ms=total_latency_ms,
            )

            # Cache the result
            if self._cache and final_result.domain != "unknown":
                await self._cache.set(query, final_result)

            return final_result

        # Should not reach here, but handle gracefully
        return RouterResult(
            domain="unknown",
            confidence=0.0,
            reasoning="No routing result from any tier",
            latency_ms=total_latency_ms,
        )

    async def _route_with_tier(
        self,
        tier: ModelTier,
        messages: list[LLMMessage],
    ) -> RouterResult:
        """
        Route using a specific model tier.

        Args:
            tier: Model tier to use
            messages: Pre-built messages for the routing request

        Returns:
            RouterResult from this tier
        """
        try:
            response = await tier.model.complete(
                messages=messages,
                tools=self._routing_tools,
                temperature=tier.temperature,
                max_tokens=tier.max_tokens,
            )
        except Exception as e:
            logger.error(f"Tier {tier.name} routing failed: {e}")
            return RouterResult(
                domain="unknown",
                confidence=0.0,
                reasoning=f"Tier {tier.name} error: {e}",
                model_used=tier.name,
            )

        return self._parse_routing_response(response, tier.name)

    def _build_messages(
        self,
        query: str,
        context: dict[str, Any] | None,
    ) -> list[LLMMessage]:
        """Build the messages list for routing."""
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self._system_prompt),
            LLMMessage(role=MessageRole.USER, content=query),
        ]

        if context:
            context_str = self._format_context(context)
            if context_str:
                messages.append(
                    LLMMessage(
                        role=MessageRole.SYSTEM,
                        content=f"Additional context:\n{context_str}",
                    )
                )

        return messages

    def _parse_routing_response(
        self,
        response: Any,
        model_name: str,
    ) -> RouterResult:
        """Parse the LLM response to extract routing decision."""
        if response.has_tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.name

            if tool_name.startswith("route_to_"):
                domain = tool_name[len("route_to_") :]

                if domain in self._get_provider_names():
                    confidence = tool_call.arguments.get("confidence", self._default_confidence)
                    reasoning = tool_call.arguments.get("reasoning")

                    return RouterResult(
                        domain=domain,
                        confidence=float(confidence),
                        reasoning=reasoning,
                        model_used=model_name,
                    )

            logger.warning(f"Unknown routing tool: {tool_name}")

        # Fallback parsing from content
        if response.content:
            content_lower = response.content.lower()
            for provider in self._providers:
                if provider.provider_name.lower() in content_lower:
                    return RouterResult(
                        domain=provider.provider_name,
                        confidence=0.5,
                        reasoning="Parsed from content (no tool call)",
                        model_used=model_name,
                    )

        return RouterResult(
            domain="unknown",
            confidence=0.0,
            reasoning="No routing tool selected",
            model_used=model_name,
        )

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context dict as string."""
        parts = []

        if "resolved_entities" in context and context["resolved_entities"]:
            entities = context["resolved_entities"]
            entity_str = ", ".join(f"{k}={v}" for k, v in entities.items())
            parts.append(f"Resolved entities: {entity_str}")

        if "conversation_history" in context and context["conversation_history"]:
            parts.append("Previous conversation context available")

        return "\n".join(parts)


def create_escalating_router(
    tier1_model: LLMProvider,
    tier2_model: LLMProvider | None = None,
    tier3_model: LLMProvider | None = None,
    tool_providers: list[ToolProvider] | None = None,
    cache: RoutingCache | None = None,
    escalation_threshold: float = 0.7,
) -> EscalatingLLMRouter:
    """
    Factory function to create an escalating router with common configurations.

    Args:
        tier1_model: Primary (fast/cheap) model - required
        tier2_model: Secondary (balanced) model - optional
        tier3_model: Tertiary (most capable) model - optional
        tool_providers: Tool providers (agents)
        cache: Optional routing cache
        escalation_threshold: Confidence threshold for escalation

    Returns:
        Configured EscalatingLLMRouter
    """
    tiers = [ModelTier(name=tier1_model.model_name, model=tier1_model)]

    if tier2_model:
        tiers.append(ModelTier(name=tier2_model.model_name, model=tier2_model))

    if tier3_model:
        tiers.append(ModelTier(name=tier3_model.model_name, model=tier3_model))

    return EscalatingLLMRouter(
        model_tiers=tiers,
        tool_providers=tool_providers or [],
        cache=cache,
        escalation_threshold=escalation_threshold,
    )
