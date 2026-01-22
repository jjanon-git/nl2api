"""Tests for ConversationManager."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from CONTRACTS import ToolCall
from src.nl2api.conversation.manager import (
    ConversationManager,
    ConversationStorage,
    InMemoryConversationStorage,
)
from src.nl2api.conversation.models import (
    ConversationSession,
    ConversationTurn,
)


class TestInMemoryConversationStorage:
    """Tests for InMemoryConversationStorage."""

    @pytest.mark.asyncio
    async def test_save_and_get_session(self) -> None:
        """Test saving and retrieving a session."""
        storage = InMemoryConversationStorage()
        session = ConversationSession(user_id="test-user")

        await storage.save_session(session)
        retrieved = await storage.get_session(session.id)

        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self) -> None:
        """Test getting a session that doesn't exist."""
        storage = InMemoryConversationStorage()
        result = await storage.get_session(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent_turns(self) -> None:
        """Test getting recent turns."""
        storage = InMemoryConversationStorage()
        session = ConversationSession()

        for i in range(5):
            session.add_turn(ConversationTurn(
                turn_number=i + 1,
                user_query=f"query {i}",
            ))

        await storage.save_session(session)

        turns = await storage.get_recent_turns(session.id, limit=3)
        assert len(turns) == 3
        assert turns[0].user_query == "query 2"

    @pytest.mark.asyncio
    async def test_expire_inactive_sessions(self) -> None:
        """Test expiring inactive sessions."""
        storage = InMemoryConversationStorage()

        # Create an old session
        old_session = ConversationSession()
        old_session.last_activity_at = datetime.now(UTC) - timedelta(hours=2)
        await storage.save_session(old_session)

        # Create a recent session
        new_session = ConversationSession()
        await storage.save_session(new_session)

        # Expire sessions older than 1 hour
        expired_count = await storage.expire_inactive_sessions(timedelta(hours=1))

        assert expired_count == 1
        assert old_session.is_active is False
        assert new_session.is_active is True


class TestConversationManager:
    """Tests for ConversationManager."""

    @pytest.mark.asyncio
    async def test_create_session(self) -> None:
        """Test creating a new session."""
        manager = ConversationManager()

        session = await manager.create_session(user_id="test-user")

        assert session is not None
        assert session.user_id == "test-user"
        assert session.is_active is True

    @pytest.mark.asyncio
    async def test_create_session_with_config_overrides(self) -> None:
        """Test creating session with config overrides."""
        manager = ConversationManager()

        config = {"model": "claude-3-opus", "temperature": 0.7}
        session = await manager.create_session(config_overrides=config)

        assert session.config_overrides == config

    @pytest.mark.asyncio
    async def test_get_session_from_cache(self) -> None:
        """Test getting session from cache."""
        manager = ConversationManager()

        session = await manager.create_session()
        retrieved = await manager.get_session(session.id)

        assert retrieved is session  # Same object from cache

    @pytest.mark.asyncio
    async def test_get_session_from_storage(self) -> None:
        """Test getting session from storage when not in cache."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage)

        session = await manager.create_session()
        session_id = session.id

        # Clear cache
        manager._active_sessions.clear()

        # Should fetch from storage
        retrieved = await manager.get_session(session_id)
        assert retrieved is not None
        assert retrieved.id == session_id

    @pytest.mark.asyncio
    async def test_get_expired_session_returns_none(self) -> None:
        """Test that expired sessions return None."""
        manager = ConversationManager(session_ttl_minutes=1)

        session = await manager.create_session()
        # Make session old
        session.last_activity_at = datetime.now(UTC) - timedelta(hours=1)

        retrieved = await manager.get_session(session.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_or_create_session_existing(self) -> None:
        """Test get_or_create with existing session."""
        manager = ConversationManager()

        original = await manager.create_session()
        retrieved = await manager.get_or_create_session(session_id=original.id)

        assert retrieved.id == original.id

    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self) -> None:
        """Test get_or_create creates new session when not found."""
        manager = ConversationManager()

        # Try to get non-existent session
        session = await manager.get_or_create_session(
            session_id=uuid4(),
            user_id="new-user",
        )

        # Should create new session
        assert session is not None
        assert session.user_id == "new-user"

    def test_get_context(self) -> None:
        """Test getting context from session."""
        manager = ConversationManager()
        session = ConversationSession()

        session.add_turn(ConversationTurn(
            turn_number=1,
            user_query="Apple's EPS?",
            resolved_entities={"Apple": "AAPL.O"},
            domain="estimates",
        ))

        context = manager.get_context(session)

        assert "Apple" in context.entities
        assert context.domain == "estimates"

    def test_expand_query_with_context(self) -> None:
        """Test query expansion using session context."""
        manager = ConversationManager()
        session = ConversationSession()

        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"fields": ["TR.EPSMean"]},
        )
        session.add_turn(ConversationTurn(
            turn_number=1,
            user_query="What is Apple's EPS?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
        ))

        # Follow-up with pronoun reference
        result = manager.expand_query("What about their revenue?", session)

        assert result.was_expanded is True
        assert "Apple" in result.expanded_query

    def test_expand_query_no_context(self) -> None:
        """Test query expansion with empty session."""
        manager = ConversationManager()
        session = ConversationSession()

        result = manager.expand_query("What is Apple's price?", session)

        assert result.was_expanded is False
        assert result.expanded_query == "What is Apple's price?"

    @pytest.mark.asyncio
    async def test_add_turn(self) -> None:
        """Test adding a turn to session."""
        manager = ConversationManager()
        session = await manager.create_session()

        turn = ConversationTurn(
            turn_number=1,
            user_query="test query",
        )
        await manager.add_turn(session, turn)

        assert session.total_turns == 1
        assert session.turns[0] == turn

    @pytest.mark.asyncio
    async def test_end_session(self) -> None:
        """Test ending a session."""
        manager = ConversationManager()
        session = await manager.create_session()
        session_id = session.id

        await manager.end_session(session_id)

        # Session should be inactive
        retrieved = await manager.get_session(session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self) -> None:
        """Test cleaning up expired sessions."""
        manager = ConversationManager(session_ttl_minutes=1)

        # Create sessions
        _active = await manager.create_session()  # noqa: F841
        expired = await manager.create_session()

        # Make one expired
        expired.last_activity_at = datetime.now(UTC) - timedelta(hours=1)

        count = await manager.cleanup_expired_sessions()

        assert count >= 1
        assert expired.is_active is False
        assert expired.id not in manager._active_sessions

    def test_build_history_prompt_empty(self) -> None:
        """Test building history prompt for empty session."""
        manager = ConversationManager()
        session = ConversationSession()

        prompt = manager.build_history_prompt(session)
        assert prompt == ""

    def test_build_history_prompt_with_turns(self) -> None:
        """Test building history prompt with turns."""
        manager = ConversationManager()
        session = ConversationSession()

        tool_call = ToolCall(
            tool_name="get_data",
            arguments={"fields": ["P"]},
        )
        session.add_turn(ConversationTurn(
            turn_number=1,
            user_query="Apple's price?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
        ))
        session.add_turn(ConversationTurn(
            turn_number=2,
            user_query="What about volume?",
            needs_clarification=True,
        ))

        prompt = manager.build_history_prompt(session)

        assert "Previous conversation:" in prompt
        assert "Apple's price?" in prompt
        assert "clarification" in prompt.lower()

    def test_build_history_prompt_respects_max_turns(self) -> None:
        """Test that history prompt respects max_turns."""
        manager = ConversationManager(history_limit=3)
        session = ConversationSession()

        for i in range(10):
            session.add_turn(ConversationTurn(
                turn_number=i + 1,
                user_query=f"query {i}",
            ))

        prompt = manager.build_history_prompt(session, max_turns=2)

        # Should only include last 2 turns
        assert "query 8" in prompt
        assert "query 9" in prompt
        assert "query 0" not in prompt


class TestConversationStorageProtocol:
    """Tests to verify protocol compliance."""

    def test_in_memory_storage_is_protocol_compliant(self) -> None:
        """Verify InMemoryConversationStorage implements protocol."""
        storage = InMemoryConversationStorage()
        assert isinstance(storage, ConversationStorage)
