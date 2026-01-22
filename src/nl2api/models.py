"""
NL2API Response Models

Defines the response models for the NL2API system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from CONTRACTS import ToolCall


@dataclass(frozen=True)
class ClarificationQuestion:
    """
    A question to clarify an ambiguous query.

    Used when the system cannot determine the user's intent.
    """

    question: str
    options: tuple[str, ...] = ()  # Optional multiple-choice options
    category: str = ""  # e.g., "entity", "time_period", "metric"


@dataclass(frozen=True)
class NL2APIResponse:
    """
    Response from the NL2API system.

    Can be either a successful API call translation or a clarification request.
    """

    # Success case
    tool_calls: tuple[ToolCall, ...] = ()
    confidence: float = 0.0
    reasoning: str = ""

    # Clarification case
    needs_clarification: bool = False
    clarification_questions: tuple[ClarificationQuestion, ...] = ()

    # Metadata
    domain: str | None = None
    resolved_entities: dict[str, str] = field(default_factory=dict)
    session_id: str | None = None
    turn_number: int = 1

    # Debug info
    raw_llm_response: str = ""
    tokens_used: int = 0
    processing_time_ms: int = 0

    # Token tracking (for cost calculation)
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "confidence": self.confidence,
            "domain": self.domain,
            "needs_clarification": self.needs_clarification,
        }

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "tool_name": tc.tool_name,
                    "arguments": dict(tc.arguments),
                }
                for tc in self.tool_calls
            ]

        if self.needs_clarification:
            result["clarification_questions"] = [
                {
                    "question": q.question,
                    "options": list(q.options),
                    "category": q.category,
                }
                for q in self.clarification_questions
            ]

        if self.reasoning:
            result["reasoning"] = self.reasoning

        if self.resolved_entities:
            result["resolved_entities"] = self.resolved_entities

        return result


@dataclass(frozen=True)
class ConversationTurn:
    """
    A single turn in a multi-turn conversation.

    Stored for context in subsequent turns.
    """

    session_id: str
    turn_number: int
    user_query: str
    expanded_query: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    resolved_entities: dict[str, str] = field(default_factory=dict)
    domain: str | None = None
    created_at: str = ""  # ISO format timestamp
