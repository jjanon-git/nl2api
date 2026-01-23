"""Tests for conversation data models."""

import pytest

from CONTRACTS import ToolCall
from src.nl2api.conversation.models import (
    ConversationContext,
    ConversationSession,
    ConversationTurn,
)


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    def test_turn_creation_minimal(self) -> None:
        """Test creating a turn with minimal fields."""
        turn = ConversationTurn(turn_number=1, user_query="What is Apple's price?")

        assert turn.turn_number == 1
        assert turn.user_query == "What is Apple's price?"
        assert turn.expanded_query is None
        assert turn.tool_calls == ()
        assert turn.needs_clarification is False

    def test_turn_creation_with_tool_calls(self) -> None:
        """Test creating a turn with tool calls."""
        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"ric": "AAPL.O", "fields": ["P", "VO"]},
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's price?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
            domain="datastream",
            confidence=0.95,
        )

        assert len(turn.tool_calls) == 1
        assert turn.resolved_entities["Apple"] == "AAPL.O"
        assert turn.domain == "datastream"

    def test_turn_is_frozen(self) -> None:
        """Test that turn is immutable."""
        turn = ConversationTurn(turn_number=1, user_query="test")

        with pytest.raises(AttributeError):
            turn.user_query = "modified"  # type: ignore

    def test_get_entities(self) -> None:
        """Test getting entities from turn."""
        turn = ConversationTurn(
            turn_number=1,
            user_query="test",
            resolved_entities={"Apple": "AAPL.O", "Microsoft": "MSFT.O"},
        )

        entities = turn.get_entities()
        assert entities == {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}
        # Should return a copy
        entities["Google"] = "GOOGL.O"
        assert "Google" not in turn.resolved_entities

    def test_get_fields(self) -> None:
        """Test extracting fields from tool calls."""
        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"ric": "AAPL.O", "fields": ["P", "VO", "HI"]},
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="test",
            tool_calls=(tool_call,),
        )

        fields = turn.get_fields()
        assert fields == ["P", "VO", "HI"]

    def test_get_fields_multiple_tool_calls(self) -> None:
        """Test extracting fields from multiple tool calls."""
        tool1 = ToolCall(tool_name="get_data", arguments={"fields": ["P"]})
        tool2 = ToolCall(tool_name="get_data", arguments={"fields": ["VO", "HI"]})
        turn = ConversationTurn(
            turn_number=1,
            user_query="test",
            tool_calls=(tool1, tool2),
        )

        fields = turn.get_fields()
        assert fields == ["P", "VO", "HI"]

    def test_get_fields_no_tool_calls(self) -> None:
        """Test get_fields with no tool calls."""
        turn = ConversationTurn(turn_number=1, user_query="test")
        assert turn.get_fields() == []

    def test_turn_with_clarification(self) -> None:
        """Test turn that needs clarification."""
        turn = ConversationTurn(
            turn_number=1,
            user_query="What's the price?",
            needs_clarification=True,
            clarification_questions=("Which company?", "What time period?"),
        )

        assert turn.needs_clarification is True
        assert len(turn.clarification_questions) == 2


class TestConversationSession:
    """Tests for ConversationSession model."""

    def test_session_creation_defaults(self) -> None:
        """Test session creation with defaults."""
        session = ConversationSession()

        assert session.id is not None
        assert session.user_id is None
        assert session.is_active is True
        assert session.turns == []
        assert session.total_turns == 0

    def test_session_creation_with_user_id(self) -> None:
        """Test session creation with user ID."""
        session = ConversationSession(user_id="user-123")

        assert session.user_id == "user-123"

    def test_add_turn(self) -> None:
        """Test adding turns to session."""
        session = ConversationSession()
        turn = ConversationTurn(turn_number=1, user_query="test")

        session.add_turn(turn)

        assert session.total_turns == 1
        assert session.turns[0] == turn

    def test_add_turn_updates_last_activity(self) -> None:
        """Test that adding turn updates last_activity_at."""
        session = ConversationSession()
        original_activity = session.last_activity_at

        # Small delay to ensure time difference
        import time

        time.sleep(0.01)

        turn = ConversationTurn(turn_number=1, user_query="test")
        session.add_turn(turn)

        assert session.last_activity_at >= original_activity

    def test_get_last_turn(self) -> None:
        """Test getting the last turn."""
        session = ConversationSession()

        assert session.get_last_turn() is None

        turn1 = ConversationTurn(turn_number=1, user_query="first")
        turn2 = ConversationTurn(turn_number=2, user_query="second")
        session.add_turn(turn1)
        session.add_turn(turn2)

        assert session.get_last_turn() == turn2

    def test_get_recent_turns(self) -> None:
        """Test getting recent turns with limit."""
        session = ConversationSession()

        for i in range(10):
            session.add_turn(ConversationTurn(turn_number=i + 1, user_query=f"query {i}"))

        recent = session.get_recent_turns(3)
        assert len(recent) == 3
        assert recent[0].user_query == "query 7"
        assert recent[2].user_query == "query 9"

    def test_get_recent_turns_empty_session(self) -> None:
        """Test get_recent_turns on empty session."""
        session = ConversationSession()
        assert session.get_recent_turns(5) == []

    def test_get_all_entities(self) -> None:
        """Test getting all entities from conversation."""
        session = ConversationSession()

        turn1 = ConversationTurn(
            turn_number=1,
            user_query="Apple price",
            resolved_entities={"Apple": "AAPL.O"},
        )
        turn2 = ConversationTurn(
            turn_number=2,
            user_query="Microsoft price",
            resolved_entities={"Microsoft": "MSFT.O"},
        )
        session.add_turn(turn1)
        session.add_turn(turn2)

        all_entities = session.get_all_entities()
        assert all_entities == {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}

    def test_get_last_domain(self) -> None:
        """Test getting last domain."""
        session = ConversationSession()

        assert session.get_last_domain() is None

        session.add_turn(
            ConversationTurn(
                turn_number=1,
                user_query="test",
                domain="datastream",
            )
        )
        session.add_turn(
            ConversationTurn(
                turn_number=2,
                user_query="test 2",
                domain="estimates",
            )
        )

        assert session.get_last_domain() == "estimates"


class TestConversationContext:
    """Tests for ConversationContext model."""

    def test_empty_context(self) -> None:
        """Test creating empty context."""
        context = ConversationContext()

        assert context.entities == {}
        assert context.fields == []
        assert context.domain is None
        assert context.turn_count == 0

    def test_from_session_empty(self) -> None:
        """Test creating context from empty session."""
        session = ConversationSession()
        context = ConversationContext.from_session(session)

        assert context.entities == {}
        assert context.turn_count == 0

    def test_from_session_with_turns(self) -> None:
        """Test creating context from session with turns."""
        session = ConversationSession()

        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"fields": ["TR.EPSMean"]},
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's EPS?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
            domain="estimates",
        )
        session.add_turn(turn)

        context = ConversationContext.from_session(session)

        assert "Apple" in context.entities
        assert context.entities["Apple"] == "AAPL.O"
        assert "TR.EPSMean" in context.fields
        assert context.domain == "estimates"
        assert context.turn_count == 1

    def test_from_session_extracts_entity_names_from_query(self) -> None:
        """Test that entity names are extracted from query when not resolved."""
        session = ConversationSession()

        # Turn without resolved entities but with entity name in query
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple stock price?",
            # No resolved_entities - simulates resolution failure
        )
        session.add_turn(turn)

        context = ConversationContext.from_session(session)

        # Should extract "Apple" from the query text
        assert "Apple" in context.entities

    def test_from_session_respects_max_turns(self) -> None:
        """Test that max_turns is respected."""
        session = ConversationSession()

        for i in range(10):
            session.add_turn(
                ConversationTurn(
                    turn_number=i + 1,
                    user_query=f"query {i}",
                    resolved_entities={f"Entity{i}": f"RIC{i}"},
                )
            )

        context = ConversationContext.from_session(session, max_turns=3)

        # Should only include last 3 turns' entities
        assert context.turn_count == 3
        assert "Entity7" in context.entities
        assert "Entity8" in context.entities
        assert "Entity9" in context.entities
        assert "Entity0" not in context.entities

    def test_from_session_builds_history_text(self) -> None:
        """Test that history text is built correctly."""
        session = ConversationSession()

        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"fields": ["P"]},
        )
        session.add_turn(
            ConversationTurn(
                turn_number=1,
                user_query="What is Apple's price?",
                tool_calls=(tool_call,),
                resolved_entities={"Apple": "AAPL.O"},
            )
        )

        context = ConversationContext.from_session(session)

        assert "User:" in context.history_text
        assert "Apple" in context.history_text
