"""
LLM Tool Router

Routes queries using LLM tool selection. Agents are exposed as "tools"
to the LLM, which selects the appropriate one based on the query.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from src.common.telemetry import trace_span
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMProvider,
    LLMToolDefinition,
    MessageRole,
)
from src.nl2api.routing.protocols import (
    RouterResult,
    RoutingToolDefinition,
    ToolProvider,
)

if TYPE_CHECKING:
    from src.nl2api.routing.cache import RoutingCache

logger = logging.getLogger(__name__)

# Default system prompt for routing
ROUTING_SYSTEM_PROMPT = """You are a query router for LSEG financial data APIs.

Your task is to analyze the user's query and select the most appropriate domain API by calling the corresponding routing tool.

## Domain Guidelines

- "datastream": Current stock prices, market data, trading volume, historical price time series, indices
- "estimates": FUTURE/FORECAST data - analyst EPS forecasts, revenue projections, recommendations, price targets, I/B/E/S consensus
- "fundamentals": HISTORICAL/REPORTED data - past financial statements, Worldscope data, reported earnings, balance sheet, ratios
- "officers": Executives, board members, compensation, governance data
- "screening": Stock screening, ranking, filtering criteria, TOP/BOTTOM queries

## Critical: Temporal Context

The distinction between "estimates" and "fundamentals" depends on temporal context:
- "EPS forecast", "expected earnings", "next quarter" → estimates (FUTURE)
- "last year's EPS", "reported earnings", "2023 revenue" → fundamentals (PAST)
- "EPS" or "earnings" alone WITHOUT temporal context → AMBIGUOUS

## Confidence Scoring

Set confidence based on clarity:
- 0.85-1.0: Clear, unambiguous query with explicit temporal context
- 0.6-0.8: Reasonable inference but some ambiguity
- 0.3-0.5: AMBIGUOUS - query lacks temporal context for earnings/EPS, could be estimates OR fundamentals

IMPORTANT: When a query asks for "EPS", "earnings", or similar metrics WITHOUT specifying past/historical vs future/forecast, you MUST set confidence <= 0.5 because clarification is needed."""


class LLMToolRouter:
    """
    Routes queries using LLM tool selection.

    Exposes each domain agent as a "routing tool" to the LLM.
    The LLM's tool selection capability determines the appropriate domain.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_providers: list[ToolProvider],
        cache: RoutingCache | None = None,
        routing_model: str | None = None,
        system_prompt: str | None = None,
        default_confidence: float = 0.85,
    ):
        """
        Initialize the LLM tool router.

        Args:
            llm: LLM provider for routing decisions
            tool_providers: List of tool providers (agents or MCP servers)
            cache: Optional routing cache for performance
            routing_model: Optional model override for routing (e.g., use Haiku for cost)
            system_prompt: Optional custom system prompt
            default_confidence: Default confidence when LLM doesn't provide one
        """
        self._llm = llm
        self._providers = tool_providers
        self._cache = cache
        self._routing_model = routing_model
        self._system_prompt = system_prompt or ROUTING_SYSTEM_PROMPT
        self._default_confidence = default_confidence

        # Build routing tools from providers
        self._routing_tools = self._build_routing_tools()

        logger.info(
            f"LLMToolRouter initialized with {len(self._providers)} providers, "
            f"model={routing_model or 'default'}"
        )

    def _build_routing_tools(self) -> list[LLMToolDefinition]:
        """
        Convert tool providers to routing tools.

        Each provider becomes a routing tool the LLM can select.
        """
        tools = []

        for provider in self._providers:
            # Create routing tool definition
            routing_def = RoutingToolDefinition(
                name=f"route_to_{provider.provider_name}",
                description=provider.provider_description,
                # Get capabilities and examples if provider supports them
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
        Route a query to the appropriate domain.

        Args:
            query: Natural language query to route
            context: Optional context (resolved entities, etc.)

        Returns:
            RouterResult with domain and confidence
        """
        start_time = time.perf_counter()

        with trace_span("router.route", {"query_length": len(query)}) as span:
            # Step 1: Check cache
            with trace_span("router.cache_lookup") as cache_span:
                if self._cache:
                    cached = await self._cache.get(query)
                    cache_span.set_attribute("cache.hit", cached is not None)
                    if cached:
                        latency_ms = int((time.perf_counter() - start_time) * 1000)
                        logger.debug(f"Cache hit for query routing: {cached.domain}")
                        span.set_attribute("routing.cached", True)
                        span.set_attribute("routing.domain", cached.domain)
                        span.set_attribute("routing.confidence", cached.confidence)
                        span.set_attribute("routing.latency_ms", latency_ms)
                        return RouterResult(
                            domain=cached.domain,
                            confidence=cached.confidence,
                            reasoning=cached.reasoning,
                            cached=True,
                            latency_ms=latency_ms,
                        )
                else:
                    cache_span.set_attribute("cache.enabled", False)

            # Step 2: Build routing messages
            messages = [
                LLMMessage(role=MessageRole.SYSTEM, content=self._system_prompt),
                LLMMessage(role=MessageRole.USER, content=query),
            ]

            # Add context if provided
            if context:
                context_str = self._format_context(context)
                if context_str:
                    messages.append(
                        LLMMessage(
                            role=MessageRole.SYSTEM,
                            content=f"Additional context:\n{context_str}",
                        )
                    )

            # Step 3: LLM call with routing tools (with retry for rate limits)
            with trace_span("router.llm_call") as llm_span:
                llm_span.set_attribute("llm.model", self._llm.model_name)
                llm_span.set_attribute("llm.tools_count", len(self._routing_tools))
                try:
                    response = await self._llm.complete_with_retry(
                        messages=messages,
                        tools=self._routing_tools,
                        temperature=0.0,
                        max_tokens=150,
                        max_retries=3,
                    )
                    if hasattr(response, "usage") and response.usage:
                        llm_span.set_attribute(
                            "llm.tokens_total",
                            response.usage.total_tokens
                            if hasattr(response.usage, "total_tokens")
                            else 0,
                        )
                except Exception as e:
                    logger.error(f"LLM routing call failed: {e}")
                    llm_span.set_attribute("llm.error", str(e))
                    # Return unknown domain with zero confidence
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    span.set_attribute("routing.error", True)
                    span.set_attribute("routing.latency_ms", latency_ms)
                    return RouterResult(
                        domain="unknown",
                        confidence=0.0,
                        reasoning=f"Routing error: {e}",
                        latency_ms=latency_ms,
                    )

            # Step 4: Parse tool selection
            result = self._parse_routing_response(response, start_time)
            span.set_attribute("routing.domain", result.domain)
            span.set_attribute("routing.confidence", result.confidence)
            span.set_attribute("routing.cached", False)
            span.set_attribute("routing.latency_ms", result.latency_ms)

            # Step 5: Cache result (if successful)
            if self._cache and result.domain != "unknown":
                with trace_span("router.cache_set") as cache_set_span:
                    await self._cache.set(query, result)
                    cache_set_span.set_attribute("cache.domain", result.domain)

            return result

    def _parse_routing_response(
        self,
        response: Any,
        start_time: float,
    ) -> RouterResult:
        """
        Parse the LLM response to extract routing decision.

        Args:
            response: LLM response object
            start_time: Start time for latency calculation

        Returns:
            RouterResult parsed from response
        """
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract token usage from response
        # LLM providers may use different key names: prompt_tokens/completion_tokens
        # or input_tokens/output_tokens
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.get("input_tokens") or response.usage.get(
                "prompt_tokens", 0
            )
            output_tokens = response.usage.get("output_tokens") or response.usage.get(
                "completion_tokens", 0
            )

        if response.has_tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.name

            # Extract domain from tool name (route_to_<domain>)
            if tool_name.startswith("route_to_"):
                domain = tool_name[len("route_to_") :]

                # Validate domain exists
                if domain in self._get_provider_names():
                    confidence = tool_call.arguments.get("confidence", self._default_confidence)
                    reasoning = tool_call.arguments.get("reasoning")

                    return RouterResult(
                        domain=domain,
                        confidence=float(confidence),
                        reasoning=reasoning,
                        latency_ms=latency_ms,
                        model_used=self._llm.model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

            logger.warning(f"Unknown routing tool selected: {tool_name}")

        # Fallback: try to parse from content (shouldn't normally happen)
        if response.content:
            content_lower = response.content.lower()
            for provider in self._providers:
                if provider.provider_name.lower() in content_lower:
                    logger.warning(
                        f"Routing via content parsing (tool call expected): "
                        f"{provider.provider_name}"
                    )
                    return RouterResult(
                        domain=provider.provider_name,
                        confidence=0.5,  # Lower confidence for fallback
                        reasoning="Parsed from response content (no tool call)",
                        latency_ms=latency_ms,
                        model_used=self._llm.model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

        # No valid routing found
        logger.warning("No valid routing decision from LLM")
        return RouterResult(
            domain="unknown",
            confidence=0.0,
            reasoning="LLM did not select a routing tool",
            latency_ms=latency_ms,
            model_used=self._llm.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context dict as string for the prompt."""
        parts = []

        if "resolved_entities" in context and context["resolved_entities"]:
            entities = context["resolved_entities"]
            entity_str = ", ".join(f"{k}={v}" for k, v in entities.items())
            parts.append(f"Resolved entities: {entity_str}")

        if "conversation_history" in context and context["conversation_history"]:
            parts.append("Previous conversation context available")

        return "\n".join(parts)
