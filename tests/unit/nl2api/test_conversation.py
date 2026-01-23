"""Tests for conversation management module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from CONTRACTS import ToolCall
from src.nl2api.conversation.expander import QueryExpander
from src.nl2api.conversation.manager import (
    ConversationManager,
    InMemoryConversationStorage,
)
from src.nl2api.conversation.models import (
    ConversationContext,
    ConversationSession,
    ConversationTurn,
)


class TestConversationTurn:
    """Test suite for ConversationTurn."""

    def test_basic_turn_creation(self) -> None:
        """Test creating a basic conversation turn."""
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's EPS estimate?",
        )

        assert turn.turn_number == 1
        assert turn.user_query == "What is Apple's EPS estimate?"
        assert turn.expanded_query is None
        assert turn.tool_calls == ()
        assert turn.resolved_entities == {}
        assert turn.domain is None

    def test_turn_with_tool_calls(self) -> None:
        """Test turn with tool calls."""
        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"RICs": ["AAPL.O"], "fields": ["TR.EPSMean"]},
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's EPS?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
            domain="estimates",
            confidence=0.9,
        )

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].tool_name == "get_data"
        assert turn.resolved_entities == {"Apple": "AAPL.O"}
        assert turn.domain == "estimates"
        assert turn.confidence == 0.9

    def test_turn_get_entities(self) -> None:
        """Test getting entities from a turn."""
        turn = ConversationTurn(
            turn_number=1,
            user_query="Compare Apple and Microsoft EPS",
            resolved_entities={"Apple": "AAPL.O", "Microsoft": "MSFT.O"},
        )

        entities = turn.get_entities()
        assert entities == {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}

    def test_turn_get_fields(self) -> None:
        """Test extracting fields from tool calls."""
        tool_call = ToolCall(
            tool_name="get_data",
            arguments={
                "RICs": ["AAPL.O"],
                "fields": ["TR.EPSMean(Period=FY1)", "TR.RevenueMean(Period=FY1)"],
            },
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="Get Apple's EPS and revenue",
            tool_calls=(tool_call,),
        )

        fields = turn.get_fields()
        assert fields == ["TR.EPSMean(Period=FY1)", "TR.RevenueMean(Period=FY1)"]

    def test_turn_is_frozen(self) -> None:
        """Test that turns are immutable."""
        turn = ConversationTurn(turn_number=1, user_query="Test")

        with pytest.raises(AttributeError):
            turn.user_query = "Changed"  # type: ignore

    def test_turn_with_clarification(self) -> None:
        """Test turn that needed clarification."""
        turn = ConversationTurn(
            turn_number=1,
            user_query="What's the EPS?",
            needs_clarification=True,
            clarification_questions=("Which company?", "Which time period?"),
            clarification_response="Apple for next year",
        )

        assert turn.needs_clarification is True
        assert len(turn.clarification_questions) == 2
        assert turn.clarification_response == "Apple for next year"


class TestConversationSession:
    """Test suite for ConversationSession."""

    def test_session_creation(self) -> None:
        """Test creating a new session."""
        session = ConversationSession()

        assert isinstance(session.id, UUID)
        assert session.user_id is None
        assert session.is_active is True
        assert session.turns == []
        assert session.total_turns == 0

    def test_session_with_user_id(self) -> None:
        """Test session with user ID."""
        session = ConversationSession(user_id="user-123")
        assert session.user_id == "user-123"

    def test_add_turn(self) -> None:
        """Test adding turns to session."""
        session = ConversationSession()
        turn = ConversationTurn(turn_number=1, user_query="Test query")

        session.add_turn(turn)

        assert session.total_turns == 1
        assert session.turns[0] == turn

    def test_get_last_turn(self) -> None:
        """Test getting the last turn."""
        session = ConversationSession()

        # Empty session
        assert session.get_last_turn() is None

        # Add turns
        turn1 = ConversationTurn(turn_number=1, user_query="First")
        turn2 = ConversationTurn(turn_number=2, user_query="Second")
        session.add_turn(turn1)
        session.add_turn(turn2)

        assert session.get_last_turn() == turn2

    def test_get_recent_turns(self) -> None:
        """Test getting recent turns."""
        session = ConversationSession()

        # Add 10 turns
        for i in range(10):
            turn = ConversationTurn(turn_number=i + 1, user_query=f"Query {i + 1}")
            session.add_turn(turn)

        # Get last 5
        recent = session.get_recent_turns(5)
        assert len(recent) == 5
        assert recent[0].turn_number == 6
        assert recent[-1].turn_number == 10

    def test_get_all_entities(self) -> None:
        """Test collecting entities from all turns."""
        session = ConversationSession()

        turn1 = ConversationTurn(
            turn_number=1,
            user_query="Apple's EPS",
            resolved_entities={"Apple": "AAPL.O"},
        )
        turn2 = ConversationTurn(
            turn_number=2,
            user_query="Compare to Microsoft",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        session.add_turn(turn1)
        session.add_turn(turn2)

        entities = session.get_all_entities()
        assert entities == {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}

    def test_get_last_domain(self) -> None:
        """Test getting the last domain used."""
        session = ConversationSession()

        assert session.get_last_domain() is None

        turn = ConversationTurn(turn_number=1, user_query="Test", domain="estimates")
        session.add_turn(turn)

        assert session.get_last_domain() == "estimates"

    def test_add_turn_updates_activity(self) -> None:
        """Test that adding turn updates last_activity_at."""
        session = ConversationSession()
        original_time = session.last_activity_at

        # Add turn after a small delay
        import time

        time.sleep(0.01)

        turn = ConversationTurn(turn_number=1, user_query="Test")
        session.add_turn(turn)

        assert session.last_activity_at >= original_time


class TestConversationContext:
    """Test suite for ConversationContext."""

    def test_empty_context(self) -> None:
        """Test creating empty context."""
        context = ConversationContext()

        assert context.entities == {}
        assert context.fields == []
        assert context.domain is None
        assert context.periods == []
        assert context.history_text == ""
        assert context.turn_count == 0

    def test_from_empty_session(self) -> None:
        """Test building context from empty session."""
        session = ConversationSession()
        context = ConversationContext.from_session(session)

        assert context.turn_count == 0
        assert context.entities == {}

    def test_from_session_with_turns(self) -> None:
        """Test building context from session with turns."""
        session = ConversationSession()

        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"RICs": ["AAPL.O"], "fields": ["TR.EPSMean(Period=FY1)"]},
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="Apple's EPS",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
            domain="estimates",
        )
        session.add_turn(turn)

        context = ConversationContext.from_session(session)

        assert context.turn_count == 1
        assert context.entities == {"Apple": "AAPL.O"}
        assert "TR.EPSMean(Period=FY1)" in context.fields
        assert context.domain == "estimates"

    def test_from_session_respects_max_turns(self) -> None:
        """Test that max_turns limit is respected."""
        session = ConversationSession()

        # Add 10 turns
        for i in range(10):
            turn = ConversationTurn(
                turn_number=i + 1,
                user_query=f"Query {i + 1}",
                resolved_entities={f"Company{i}": f"RIC{i}"},
            )
            session.add_turn(turn)

        # Get context with max 3 turns
        context = ConversationContext.from_session(session, max_turns=3)

        assert context.turn_count == 3
        # Should only have entities from last 3 turns
        assert "Company7" in context.entities
        assert "Company8" in context.entities
        assert "Company9" in context.entities
        assert "Company0" not in context.entities

    def test_history_text_format(self) -> None:
        """Test that history text is properly formatted."""
        session = ConversationSession()

        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"RICs": ["AAPL.O"], "fields": ["TR.EPSMean"]},
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="Apple's EPS",
            expanded_query="What is Apple's EPS estimate?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
        )
        session.add_turn(turn)

        context = ConversationContext.from_session(session)

        # Should use expanded query if available
        assert "What is Apple's EPS estimate?" in context.history_text
        assert "System:" in context.history_text

    def test_from_session_extracts_entities_from_query_text(self) -> None:
        """Test fallback entity extraction from query text when resolved_entities is empty.

        This tests the fix for the multi-turn context bug where entity resolution
        might fail (e.g., external API rate limit) but we still want to preserve
        the entity name from the query text for pronoun expansion.
        """
        session = ConversationSession()

        # Simulate a turn where entity resolution failed (resolved_entities is empty)
        # but the query clearly mentions a company name
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Microsoft's stock price?",
            domain="datastream",
            resolved_entities={},  # Entity resolution failed
        )
        session.add_turn(turn)

        context = ConversationContext.from_session(session)

        # Should extract "Microsoft" from the query text as a fallback
        assert context.turn_count == 1
        assert "Microsoft" in context.entities

    def test_from_session_prefers_resolved_entities(self) -> None:
        """Test that resolved entities take precedence over text extraction."""
        session = ConversationSession()

        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's stock price?",
            domain="datastream",
            resolved_entities={"Apple": "AAPL.O"},  # Proper resolution
        )
        session.add_turn(turn)

        context = ConversationContext.from_session(session)

        # Should use resolved entity with proper RIC
        assert context.entities == {"Apple": "AAPL.O"}


class TestQueryExpander:
    """Test suite for QueryExpander."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_no_expansion_without_context(self) -> None:
        """Test that no expansion happens without context."""
        context = ConversationContext()  # Empty context

        result = self.expander.expand("What is Apple's EPS?", context)

        assert result.was_expanded is False
        assert result.original_query == result.expanded_query

    def test_pronoun_expansion_their(self) -> None:
        """Test expansion of 'their' pronoun."""
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = self.expander.expand("What is their revenue?", context)

        assert result.was_expanded is True
        assert "Apple" in result.expanded_query
        assert result.expansion_type == "entity"

    def test_pronoun_expansion_its(self) -> None:
        """Test expansion of 'its' pronoun."""
        context = ConversationContext(
            entities={"Microsoft": "MSFT.O"},
            turn_count=1,
        )

        result = self.expander.expand("What is its EPS?", context)

        assert result.was_expanded is True
        assert "Microsoft's" in result.expanded_query

    def test_comparison_expansion(self) -> None:
        """Test expansion of comparison queries."""
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean(Period=FY1)"],
            turn_count=1,
        )

        result = self.expander.expand("What about Microsoft?", context)

        assert result.was_expanded is True
        assert "Microsoft" in result.expanded_query
        assert result.expansion_type == "comparison"

    def test_period_expansion(self) -> None:
        """Test expansion of period change queries."""
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean(Period=FY1)"],
            turn_count=1,
        )

        result = self.expander.expand("now quarterly", context)

        assert result.was_expanded is True
        assert "quarterly" in result.expanded_query.lower()
        assert result.expansion_type == "period"

    def test_metric_expansion(self) -> None:
        """Test expansion of metric change queries."""
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = self.expander.expand("what about revenue?", context)

        assert result.was_expanded is True
        assert "Apple" in result.expanded_query
        assert "revenue" in result.expanded_query.lower()
        assert result.expansion_type == "metric"

    def test_no_expansion_for_complete_query(self) -> None:
        """Test no expansion for already complete queries."""
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = self.expander.expand(
            "What is Microsoft's revenue estimate for 2024?",
            context,
        )

        # This is a complete query, should not be expanded
        assert result.was_expanded is False

    def test_expansion_confidence(self) -> None:
        """Test that expansion results have confidence scores."""
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = self.expander.expand("What is their EPS?", context)

        assert 0 < result.confidence <= 1.0


class TestInMemoryConversationStorage:
    """Test suite for InMemoryConversationStorage."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.storage = InMemoryConversationStorage()

    @pytest.mark.asyncio
    async def test_save_and_get_session(self) -> None:
        """Test saving and retrieving a session."""
        session = ConversationSession(user_id="test-user")

        await self.storage.save_session(session)
        retrieved = await self.storage.get_session(session.id)

        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self) -> None:
        """Test getting a session that doesn't exist."""
        result = await self.storage.get_session(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_save_turn(self) -> None:
        """Test saving a turn to a session.

        Note: InMemoryStorage.save_turn is a no-op because the session
        object stored is the same object that the manager manipulates.
        The turn is added by the manager, not the storage.
        """
        session = ConversationSession()
        await self.storage.save_session(session)

        # Add turn to session (as manager would do)
        turn = ConversationTurn(turn_number=1, user_query="Test query")
        session.add_turn(turn)

        # save_turn is a no-op for in-memory storage, but should not error
        await self.storage.save_turn(session.id, turn)

        # Get session and verify turn is there
        retrieved = await self.storage.get_session(session.id)
        assert retrieved is not None
        assert len(retrieved.turns) == 1
        assert retrieved.turns[0].user_query == "Test query"

    @pytest.mark.asyncio
    async def test_get_recent_turns(self) -> None:
        """Test getting recent turns."""
        session = ConversationSession()
        await self.storage.save_session(session)

        # Add 5 turns to the session (as manager would do)
        for i in range(5):
            turn = ConversationTurn(turn_number=i + 1, user_query=f"Query {i + 1}")
            session.add_turn(turn)
            await self.storage.save_turn(session.id, turn)

        recent = await self.storage.get_recent_turns(session.id, limit=3)

        assert len(recent) == 3
        assert recent[0].turn_number == 3
        assert recent[-1].turn_number == 5

    @pytest.mark.asyncio
    async def test_expire_inactive_sessions(self) -> None:
        """Test expiring inactive sessions."""
        # Create an active session
        session = ConversationSession()
        # Manually set old activity time
        old_time = datetime.now(UTC) - timedelta(hours=2)
        session.last_activity_at = old_time

        await self.storage.save_session(session)

        # Expire sessions inactive for 1 hour
        expired_count = await self.storage.expire_inactive_sessions(timedelta(hours=1))

        assert expired_count == 1

        # Verify session is inactive
        retrieved = await self.storage.get_session(session.id)
        assert retrieved is not None
        assert retrieved.is_active is False


class TestConversationManager:
    """Test suite for ConversationManager."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = ConversationManager(
            history_limit=5,
            session_ttl_minutes=30,
        )

    @pytest.mark.asyncio
    async def test_create_session(self) -> None:
        """Test creating a new session."""
        session = await self.manager.create_session(user_id="test-user")

        assert isinstance(session.id, UUID)
        assert session.user_id == "test-user"
        assert session.is_active is True

    @pytest.mark.asyncio
    async def test_get_session(self) -> None:
        """Test getting an existing session."""
        session = await self.manager.create_session()
        retrieved = await self.manager.get_session(session.id)

        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self) -> None:
        """Test getting a session that doesn't exist."""
        result = await self.manager.get_session(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self) -> None:
        """Test get_or_create creates new session when none exists."""
        session = await self.manager.get_or_create_session(
            session_id=None,
            user_id="new-user",
        )

        assert session is not None
        assert session.user_id == "new-user"

    @pytest.mark.asyncio
    async def test_get_or_create_session_existing(self) -> None:
        """Test get_or_create returns existing session."""
        original = await self.manager.create_session()
        retrieved = await self.manager.get_or_create_session(
            session_id=original.id,
        )

        assert retrieved.id == original.id

    @pytest.mark.asyncio
    async def test_add_turn(self) -> None:
        """Test adding a turn to a session."""
        session = await self.manager.create_session()
        turn = ConversationTurn(turn_number=1, user_query="Test")

        await self.manager.add_turn(session, turn)

        assert session.total_turns == 1

    def test_get_context(self) -> None:
        """Test getting conversation context."""
        session = ConversationSession()
        turn = ConversationTurn(
            turn_number=1,
            user_query="Apple's EPS",
            resolved_entities={"Apple": "AAPL.O"},
        )
        session.add_turn(turn)

        context = self.manager.get_context(session)

        assert context.turn_count == 1
        assert "Apple" in context.entities

    def test_expand_query(self) -> None:
        """Test query expansion using manager."""
        session = ConversationSession()
        turn = ConversationTurn(
            turn_number=1,
            user_query="Apple's EPS",
            resolved_entities={"Apple": "AAPL.O"},
        )
        session.add_turn(turn)

        result = self.manager.expand_query("What is their revenue?", session)

        assert result.was_expanded is True
        assert "Apple" in result.expanded_query

    @pytest.mark.asyncio
    async def test_end_session(self) -> None:
        """Test ending a session."""
        session = await self.manager.create_session()
        await self.manager.end_session(session.id)

        retrieved = await self.manager.get_session(session.id)
        assert retrieved is None  # Inactive sessions return None

    def test_build_history_prompt(self) -> None:
        """Test building history prompt."""
        session = ConversationSession()

        # Empty session
        prompt = self.manager.build_history_prompt(session)
        assert prompt == ""

        # Add turns
        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"RICs": ["AAPL.O"], "fields": ["TR.EPSMean"]},
        )
        turn = ConversationTurn(
            turn_number=1,
            user_query="Apple's EPS",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
        )
        session.add_turn(turn)

        prompt = self.manager.build_history_prompt(session)

        assert "Previous conversation:" in prompt
        assert "User:" in prompt
        assert "Apple" in prompt

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self) -> None:
        """Test cleaning up expired sessions."""
        # Create a session with old activity time
        session = await self.manager.create_session()
        session.last_activity_at = datetime.now(UTC) - timedelta(hours=2)

        # Create manager with short TTL
        manager = ConversationManager(session_ttl_minutes=1)
        manager._active_sessions[session.id] = session

        await manager.cleanup_expired_sessions()

        # At least the cached session should be expired
        assert session.id not in manager._active_sessions
