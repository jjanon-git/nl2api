"""
Conversation Models

Data models for multi-turn conversation tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from CONTRACTS import ToolCall


@dataclass(frozen=True)
class ConversationTurn:
    """
    A single turn in a conversation.

    Represents one user query and the system's response.
    """

    turn_number: int
    user_query: str

    # Expanded query after context resolution
    expanded_query: str | None = None

    # Response
    tool_calls: tuple[ToolCall, ...] = ()
    resolved_entities: dict[str, str] = field(default_factory=dict)
    domain: str | None = None
    confidence: float = 0.0

    # Clarification
    needs_clarification: bool = False
    clarification_questions: tuple[str, ...] = ()
    clarification_response: str | None = None

    # Metadata
    processing_time_ms: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_entities(self) -> dict[str, str]:
        """Get resolved entities from this turn."""
        return dict(self.resolved_entities)

    def get_fields(self) -> list[str]:
        """Extract field codes from tool calls."""
        fields = []
        for tc in self.tool_calls:
            if isinstance(tc.arguments, dict):
                fields.extend(tc.arguments.get("fields", []))
        return fields


@dataclass
class ConversationSession:
    """
    A conversation session containing multiple turns.

    Tracks the full conversation history and provides
    context for query expansion.
    """

    id: UUID = field(default_factory=uuid4)
    user_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    turns: list[ConversationTurn] = field(default_factory=list)
    context_summary: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to the conversation."""
        self.turns.append(turn)
        self.last_activity_at = datetime.now(timezone.utc)

    def get_last_turn(self) -> ConversationTurn | None:
        """Get the most recent turn."""
        return self.turns[-1] if self.turns else None

    def get_recent_turns(self, limit: int = 5) -> list[ConversationTurn]:
        """Get the most recent N turns."""
        return self.turns[-limit:] if self.turns else []

    def get_all_entities(self) -> dict[str, str]:
        """Get all resolved entities from the conversation."""
        entities = {}
        for turn in self.turns:
            entities.update(turn.resolved_entities)
        return entities

    def get_last_domain(self) -> str | None:
        """Get the domain from the last turn."""
        last = self.get_last_turn()
        return last.domain if last else None

    @property
    def total_turns(self) -> int:
        """Get total number of turns."""
        return len(self.turns)


@dataclass(frozen=True)
class ConversationContext:
    """
    Context extracted from conversation history.

    Used by QueryExpander to understand follow-up queries.
    """

    # Previous entities mentioned
    entities: dict[str, str] = field(default_factory=dict)

    # Previous metrics/fields requested
    fields: list[str] = field(default_factory=list)

    # Previous domain used
    domain: str | None = None

    # Previous time periods
    periods: list[str] = field(default_factory=list)

    # Raw history for LLM context
    history_text: str = ""

    # Number of turns in context
    turn_count: int = 0

    @classmethod
    def from_session(
        cls,
        session: ConversationSession,
        max_turns: int = 5,
    ) -> ConversationContext:
        """
        Build context from a conversation session.

        Args:
            session: The conversation session
            max_turns: Maximum turns to include in context

        Returns:
            ConversationContext with extracted information
        """
        import re

        recent = session.get_recent_turns(max_turns)

        if not recent:
            return cls()

        # Collect entities from all turns
        entities = {}
        for turn in recent:
            entities.update(turn.resolved_entities)

        # Fallback: extract entity names from query text if no resolved entities
        # This handles cases where entity resolution failed but we still know the entity name
        if not entities:
            for turn in recent:
                query = turn.expanded_query or turn.user_query
                # Pattern for capitalized words that look like company names
                # Matches: "Apple", "Microsoft", "JP Morgan"
                cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
                matches = re.findall(cap_pattern, query)
                # Filter out common words and keep potential entity names
                common_words = {"What", "How", "Show", "Get", "The", "For", "And", "Which", "When", "Where"}
                for match in matches:
                    if match not in common_words:
                        # Store with placeholder RIC - the key is the entity name
                        entities[match] = match

        # Collect fields from all turns
        fields = []
        for turn in recent:
            fields.extend(turn.get_fields())

        # Get last domain
        domain = recent[-1].domain if recent else None

        # Extract periods from fields
        periods = []
        for f in fields:
            if "Period=" in f:
                period = f.split("Period=")[1].split(")")[0]
                if period not in periods:
                    periods.append(period)

        # Build history text for LLM
        history_lines = []
        for turn in recent:
            query = turn.expanded_query or turn.user_query
            history_lines.append(f"User: {query}")
            if turn.tool_calls:
                fields_str = ", ".join(turn.get_fields()[:3])
                entities_str = ", ".join(turn.resolved_entities.keys())
                history_lines.append(
                    f"System: Called get_data for {entities_str} with fields [{fields_str}]"
                )
            elif turn.needs_clarification:
                history_lines.append("System: Asked for clarification")

        return cls(
            entities=entities,
            fields=fields,
            domain=domain,
            periods=periods,
            history_text="\n".join(history_lines),
            turn_count=len(recent),
        )
