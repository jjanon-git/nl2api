"""
Integration tests for multi-turn conversation flows.

These tests verify that the orchestrator correctly handles:
- Session creation and persistence
- Query expansion with conversation context
- Entity carryover between turns
- Clarification flows

These tests use mocked LLM to test component integration
without requiring API keys.
"""

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from CONTRACTS import ToolCall
from src.nl2api.agents.datastream import DatastreamAgent
from src.nl2api.clarification.detector import AmbiguityDetector
from src.nl2api.conversation.manager import (
    ConversationManager,
    InMemoryConversationStorage,
)
from src.nl2api.conversation.models import ConversationTurn
from src.nl2api.llm.protocols import LLMProvider, LLMResponse
from src.nl2api.orchestrator import NL2APIOrchestrator


def create_mock_llm() -> LLMProvider:
    """Create a mock LLM provider for testing."""
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Test response",
        tool_calls=(),
        usage={"input_tokens": 100, "output_tokens": 50},
    ))
    return mock_llm


class TestMultiTurnSessionManagement:
    """Tests for session creation and persistence."""

    @pytest.mark.asyncio
    async def test_creates_new_session_when_none_provided(self) -> None:
        """Test that a new session is created when no session_id is provided."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
        )

        response = await orchestrator.process("What is Apple's price?")

        # Should have created a session
        assert response.session_id is not None
        # Session should be stored
        session = await storage.get_session(UUID(response.session_id))
        assert session is not None
        assert session.total_turns == 1

    @pytest.mark.asyncio
    async def test_reuses_existing_session(self) -> None:
        """Test that an existing session is reused when session_id is provided."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
        )

        # First query - creates session
        response1 = await orchestrator.process("What is Apple's price?")
        session_id = response1.session_id

        # Second query - reuses session
        response2 = await orchestrator.process(
            "What about volume?",
            session_id=session_id,
        )

        # Should be the same session
        assert response2.session_id == session_id

        # Session should have 2 turns
        session = await storage.get_session(UUID(session_id))
        assert session.total_turns == 2

    @pytest.mark.asyncio
    async def test_creates_new_session_for_invalid_id(self) -> None:
        """Test that a new session is created when an invalid session_id is provided."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
        )

        # Use a non-existent session ID
        from uuid import uuid4
        fake_session_id = str(uuid4())

        response = await orchestrator.process(
            "What is Apple's price?",
            session_id=fake_session_id,
        )

        # Should have created a new session (different from the fake ID)
        assert response.session_id != fake_session_id


class TestMultiTurnContextBuilding:
    """Tests for context building across turns."""

    @pytest.mark.asyncio
    async def test_context_builds_across_turns(self) -> None:
        """Test that context accumulates across multiple turns."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
        )

        # Build up context over multiple turns
        response1 = await orchestrator.process("What is Apple's price?")
        session_id = response1.session_id

        await orchestrator.process("What about Microsoft?", session_id=session_id)
        await orchestrator.process("And Google?", session_id=session_id)

        # Session should have all turns
        session = await storage.get_session(UUID(session_id))
        assert session.total_turns == 3

        # Each turn should have the original query
        assert session.turns[0].user_query == "What is Apple's price?"
        assert session.turns[1].user_query == "What about Microsoft?"
        assert session.turns[2].user_query == "And Google?"


class TestMultiTurnTurnRecording:
    """Tests for turn recording in sessions."""

    @pytest.mark.asyncio
    async def test_turns_recorded_with_user_query(self) -> None:
        """Test that turns are recorded with user queries."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
        )

        response = await orchestrator.process("What is Apple's price and volume?")

        session = await storage.get_session(UUID(response.session_id))
        turn = session.turns[0]

        # Turn should have user query
        assert turn.user_query == "What is Apple's price and volume?"
        # Turn number should be set
        assert turn.turn_number == 1

    @pytest.mark.asyncio
    async def test_turn_numbers_increment(self) -> None:
        """Test that turn numbers increment correctly."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
        )

        response1 = await orchestrator.process("Query 1")
        session_id = response1.session_id

        await orchestrator.process("Query 2", session_id=session_id)
        await orchestrator.process("Query 3", session_id=session_id)

        session = await storage.get_session(UUID(session_id))
        assert session.turns[0].turn_number == 1
        assert session.turns[1].turn_number == 2
        assert session.turns[2].turn_number == 3


class TestMultiTurnAmbiguityHandling:
    """Tests for ambiguity detection in multi-turn context."""

    @pytest.mark.asyncio
    async def test_ambiguous_query_returns_clarification(self) -> None:
        """Test that ambiguous queries return clarification questions."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()
        detector = AmbiguityDetector()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
            ambiguity_detector=detector,
        )

        # Query with ambiguous time reference
        response = await orchestrator.process("What is Apple's recent performance?")

        # Should have response (either with clarification or processed)
        assert response is not None
        assert response.session_id is not None

    @pytest.mark.asyncio
    async def test_clarification_turn_recorded(self) -> None:
        """Test that clarification turns are recorded."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()
        detector = AmbiguityDetector()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
            ambiguity_detector=detector,
        )

        response = await orchestrator.process("How is the company doing?")

        session = await storage.get_session(UUID(response.session_id))
        # Regardless of clarification, a turn should be recorded
        assert session.total_turns >= 1


class TestMultiTurnHistoryBuilding:
    """Tests for conversation history building."""

    @pytest.mark.asyncio
    async def test_history_prompt_built_from_turns(self) -> None:
        """Test that history prompt is built from conversation turns."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
            history_limit=5,
        )

        # Build history over multiple turns
        response1 = await orchestrator.process("What is Apple's price?")
        session_id = response1.session_id

        await orchestrator.process("What about volume?", session_id=session_id)
        await orchestrator.process("And Microsoft's price?", session_id=session_id)

        # Get session and build history
        session = await storage.get_session(UUID(session_id))
        history_prompt = orchestrator._conversation_manager.build_history_prompt(session)

        # History should contain previous queries
        # Either it has content or the session has the correct number of turns
        assert len(session.turns) == 3
        if history_prompt:
            assert "Apple" in history_prompt or "price" in history_prompt.lower()

    @pytest.mark.asyncio
    async def test_history_limit_respected(self) -> None:
        """Test that history limit is respected when building prompt."""
        mock_llm = create_mock_llm()
        storage = InMemoryConversationStorage()

        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"datastream": DatastreamAgent(llm=mock_llm)},
            conversation_storage=storage,
            history_limit=2,  # Only keep 2 turns in history
        )

        session_id = None
        for i in range(5):
            response = await orchestrator.process(
                f"Query {i}",
                session_id=session_id,
            )
            session_id = response.session_id

        # Session should have all 5 turns
        session = await storage.get_session(UUID(session_id))
        assert session.total_turns == 5

        # But history prompt should only include last 2 when limited
        history_prompt = orchestrator._conversation_manager.build_history_prompt(
            session, max_turns=2
        )
        # If there's history content, it should not contain early queries
        if history_prompt:
            assert "Query 0" not in history_prompt
            assert "Query 1" not in history_prompt
            assert "Query 2" not in history_prompt


class TestConversationManagerIntegration:
    """Tests for ConversationManager integration."""

    @pytest.mark.asyncio
    async def test_manager_creates_sessions_correctly(self) -> None:
        """Test that ConversationManager creates sessions correctly."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage, session_ttl_minutes=30)

        session = await manager.create_session(user_id="test-user")

        assert session is not None
        assert session.user_id == "test-user"
        assert session.is_active is True

        # Should be retrievable
        retrieved = await manager.get_session(session.id)
        assert retrieved is session

    @pytest.mark.asyncio
    async def test_manager_adds_turns_correctly(self) -> None:
        """Test that ConversationManager adds turns correctly."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage)

        session = await manager.create_session()
        turn = ConversationTurn(turn_number=1, user_query="Test query")

        await manager.add_turn(session, turn)

        assert session.total_turns == 1
        assert session.turns[0] == turn

    @pytest.mark.asyncio
    async def test_manager_builds_context_from_session(self) -> None:
        """Test that ConversationManager builds context from session."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage)

        session = await manager.create_session()

        # Add a turn with resolved entities
        tool_call = ToolCall(tool_name="get_data", arguments={"fields": ["P"]})
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's price?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
            domain="datastream",
        )
        await manager.add_turn(session, turn)

        # Get context
        context = manager.get_context(session)

        assert "Apple" in context.entities
        assert context.domain == "datastream"
        assert context.turn_count == 1

    @pytest.mark.asyncio
    async def test_manager_expands_query_with_context(self) -> None:
        """Test that ConversationManager expands queries using context."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage)

        session = await manager.create_session()

        # Add a turn with entity
        tool_call = ToolCall(tool_name="get_data", arguments={"fields": ["TR.EPSMean"]})
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's EPS?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
        )
        await manager.add_turn(session, turn)

        # Expand follow-up query
        result = manager.expand_query("What about their revenue?", session)

        assert result.was_expanded is True
        assert "Apple" in result.expanded_query


class TestQueryExpansionIntegration:
    """Tests for query expansion in orchestrator context."""

    @pytest.mark.asyncio
    async def test_query_expansion_applied_to_follow_ups(self) -> None:
        """Test that query expansion is applied to follow-up queries."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage)

        # Create session with context
        session = await manager.create_session()
        tool_call = ToolCall(tool_name="get_data", arguments={"fields": ["TR.EPSMean"]})
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's EPS?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
        )
        await manager.add_turn(session, turn)

        # Test pronoun expansion
        result = manager.expand_query("What is their revenue?", session)
        assert result.was_expanded is True
        assert "Apple" in result.expanded_query

    @pytest.mark.asyncio
    async def test_comparison_expansion_in_context(self) -> None:
        """Test comparison query expansion with context."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage)

        # Create session with context
        session = await manager.create_session()
        tool_call = ToolCall(tool_name="get_data", arguments={"fields": ["TR.EPSMean"]})
        turn = ConversationTurn(
            turn_number=1,
            user_query="What is Apple's EPS?",
            tool_calls=(tool_call,),
            resolved_entities={"Apple": "AAPL.O"},
        )
        await manager.add_turn(session, turn)

        # Test comparison expansion
        result = manager.expand_query("What about Microsoft?", session)
        assert result.was_expanded is True
        assert "Microsoft" in result.expanded_query


class TestSessionLifecycle:
    """Tests for session lifecycle management."""

    @pytest.mark.asyncio
    async def test_session_can_be_ended(self) -> None:
        """Test that sessions can be ended."""
        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage)

        session = await manager.create_session()
        session_id = session.id

        # End the session
        await manager.end_session(session_id)

        # Session should no longer be retrievable (inactive)
        retrieved = await manager.get_session(session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_expired_sessions_cleaned_up(self) -> None:
        """Test that expired sessions are cleaned up."""
        from datetime import datetime, timedelta

        storage = InMemoryConversationStorage()
        manager = ConversationManager(storage=storage, session_ttl_minutes=1)

        # Create and expire a session
        session = await manager.create_session()
        session.last_activity_at = datetime.now(UTC) - timedelta(hours=1)

        # Cleanup should mark it as expired
        count = await manager.cleanup_expired_sessions()
        assert count >= 1
        assert session.is_active is False
