"""
Request metrics collection for NL2API.

Provides comprehensive metrics for each request to enable:
- Accuracy measurement and regression detection
- Performance analysis and optimization
- A/B testing comparisons
- Debugging and troubleshooting
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid4())


@dataclass
class RequestMetrics:
    """
    Comprehensive metrics collected for each NL2API request.

    Categories:
    - Identification: request_id, session_id, timestamp
    - Input: query, query_length, query_type
    - Entity Resolution: entities found, resolution method
    - Routing: domain, confidence, cache status, model used
    - Context: retrieval mode, counts, latency
    - Agent Processing: rule vs LLM, tokens, latency
    - Output: tool calls generated
    - Accuracy Signals: for offline analysis

    Usage:
        metrics = RequestMetrics(query="What is Apple's EPS?")
        # ... populate fields during processing ...
        await emit_metrics(metrics)
    """

    # === Identification ===
    request_id: str = field(default_factory=_generate_request_id)
    session_id: str | None = None
    timestamp: str = field(default_factory=_utc_now)

    # === Input ===
    query: str = ""
    query_length: int = 0
    query_expanded: bool = False
    query_expansion_reason: str | None = None

    # === Entity Resolution ===
    entities_extracted: list[str] = field(default_factory=list)
    entities_resolved: dict[str, str] = field(default_factory=dict)
    entities_unresolved: list[str] = field(default_factory=list)
    entity_resolution_method: str = "none"  # none, cache, static, fuzzy, api
    entity_resolution_latency_ms: int = 0

    # === Routing ===
    routing_domain: str = ""
    routing_confidence: float = 0.0
    routing_cached: bool = False
    routing_cache_type: str | None = None  # exact, semantic
    routing_model: str | None = None
    routing_latency_ms: int = 0
    routing_alternatives: list[dict[str, Any]] = field(default_factory=list)

    # === Context Retrieval ===
    context_mode: str = "local"  # local, mcp, hybrid
    context_field_codes_count: int = 0
    context_examples_count: int = 0
    context_latency_ms: int = 0

    # === Agent Processing ===
    agent_domain: str = ""
    agent_used_llm: bool = False
    agent_rule_matched: str | None = None
    agent_llm_model: str | None = None
    agent_tokens_prompt: int = 0
    agent_tokens_completion: int = 0
    agent_latency_ms: int = 0

    # === Output ===
    tool_calls_count: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)

    # === Clarification ===
    needs_clarification: bool = False
    clarification_type: str | None = None  # domain, entity, ambiguity, agent

    # === Performance ===
    total_latency_ms: int = 0
    total_tokens: int = 0

    # === Status ===
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None

    # === Accuracy Signals (populated by offline evaluation) ===
    accuracy_routing_correct: bool | None = None
    accuracy_tools_match: bool | None = None
    accuracy_score: float | None = None

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        if self.query and not self.query_length:
            self.query_length = len(self.query)

        # Compute total tokens
        self.total_tokens = self.agent_tokens_prompt + self.agent_tokens_completion

        # Compute unresolved entities
        if self.entities_extracted and self.entities_resolved:
            resolved_names = set(self.entities_resolved.keys())
            self.entities_unresolved = [
                e for e in self.entities_extracted if e not in resolved_names
            ]

    def finalize(self, total_latency_ms: int) -> None:
        """
        Finalize metrics at end of request processing.

        Args:
            total_latency_ms: Total request latency in milliseconds
        """
        self.total_latency_ms = total_latency_ms
        self.total_tokens = self.agent_tokens_prompt + self.agent_tokens_completion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for structured logging."""
        return json.dumps(self.to_dict(), default=str)

    def to_log_summary(self) -> str:
        """Generate concise log summary for quick debugging."""
        parts = [
            f"domain={self.routing_domain}",
            f"conf={self.routing_confidence:.2f}",
            f"cached={self.routing_cached}",
            f"llm={self.agent_used_llm}",
            f"tools={self.tool_calls_count}",
            f"latency={self.total_latency_ms}ms",
        ]
        if self.total_tokens > 0:
            parts.append(f"tokens={self.total_tokens}")
        if not self.success:
            parts.append(f"error={self.error_type}")
        return " ".join(parts)

    def set_routing_result(
        self,
        domain: str,
        confidence: float,
        cached: bool = False,
        cache_type: str | None = None,
        model: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        """Set routing-related metrics."""
        self.routing_domain = domain
        self.routing_confidence = confidence
        self.routing_cached = cached
        self.routing_cache_type = cache_type
        self.routing_model = model
        self.routing_latency_ms = latency_ms

    def set_entity_resolution(
        self,
        extracted: list[str],
        resolved: dict[str, str],
        method: str = "static",
        latency_ms: int = 0,
    ) -> None:
        """Set entity resolution metrics."""
        self.entities_extracted = extracted
        self.entities_resolved = resolved
        self.entity_resolution_method = method
        self.entity_resolution_latency_ms = latency_ms
        # Compute unresolved
        resolved_names = set(resolved.keys())
        self.entities_unresolved = [e for e in extracted if e not in resolved_names]

    def set_context_retrieval(
        self,
        mode: str,
        field_codes_count: int,
        examples_count: int,
        latency_ms: int = 0,
    ) -> None:
        """Set context retrieval metrics."""
        self.context_mode = mode
        self.context_field_codes_count = field_codes_count
        self.context_examples_count = examples_count
        self.context_latency_ms = latency_ms

    def set_agent_result(
        self,
        domain: str,
        used_llm: bool,
        rule_matched: str | None = None,
        llm_model: str | None = None,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: int = 0,
    ) -> None:
        """Set agent processing metrics."""
        self.agent_domain = domain
        self.agent_used_llm = used_llm
        self.agent_rule_matched = rule_matched
        self.agent_llm_model = llm_model
        self.agent_tokens_prompt = tokens_prompt
        self.agent_tokens_completion = tokens_completion
        self.agent_latency_ms = latency_ms
        self.total_tokens = tokens_prompt + tokens_completion

    def set_tool_calls(self, tool_calls: list[Any]) -> None:
        """
        Set output tool calls.

        Args:
            tool_calls: List of ToolCall objects or dicts
        """
        self.tool_calls_count = len(tool_calls)
        self.tool_names = []
        self.tool_calls = []

        for tc in tool_calls:
            if hasattr(tc, "tool_name"):
                # ToolCall object
                self.tool_names.append(tc.tool_name)
                self.tool_calls.append(
                    {
                        "tool_name": tc.tool_name,
                        "arguments": dict(tc.arguments)
                        if hasattr(tc.arguments, "items")
                        else tc.arguments,
                    }
                )
            elif isinstance(tc, dict):
                self.tool_names.append(tc.get("tool_name", "unknown"))
                self.tool_calls.append(tc)

    def set_error(self, error: Exception) -> None:
        """Set error information."""
        self.success = False
        self.error_type = type(error).__name__
        self.error_message = str(error)

    def set_clarification(self, clarification_type: str) -> None:
        """Set clarification request."""
        self.needs_clarification = True
        self.clarification_type = clarification_type
