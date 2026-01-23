"""
Conversation Manager

Manages multi-turn conversations including:
- Session lifecycle
- Context tracking
- Query expansion
- History storage
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from src.nl2api.conversation.expander import ExpansionResult, QueryExpander
from src.nl2api.conversation.models import (
    ConversationContext,
    ConversationSession,
    ConversationTurn,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ConversationStorage(Protocol):
    """Protocol for conversation persistence."""

    async def get_session(self, session_id: UUID) -> ConversationSession | None:
        """Get a session by ID."""
        ...

    async def save_session(self, session: ConversationSession) -> None:
        """Save a session."""
        ...

    async def save_turn(self, session_id: UUID, turn: ConversationTurn) -> None:
        """Save a turn to a session."""
        ...

    async def get_recent_turns(
        self,
        session_id: UUID,
        limit: int = 5,
    ) -> list[ConversationTurn]:
        """Get recent turns for a session."""
        ...

    async def expire_inactive_sessions(
        self,
        inactive_threshold: timedelta,
    ) -> int:
        """Mark inactive sessions as expired. Returns count."""
        ...


class InMemoryConversationStorage:
    """In-memory conversation storage for testing."""

    def __init__(self):
        self._sessions: dict[UUID, ConversationSession] = {}

    async def get_session(self, session_id: UUID) -> ConversationSession | None:
        return self._sessions.get(session_id)

    async def save_session(self, session: ConversationSession) -> None:
        self._sessions[session.id] = session

    async def save_turn(self, session_id: UUID, turn: ConversationTurn) -> None:
        # In-memory storage doesn't need to do anything here since
        # the turn is already added to the session object by ConversationManager.
        # This method exists for API compatibility with PostgresConversationStorage.
        pass

    async def get_recent_turns(
        self,
        session_id: UUID,
        limit: int = 5,
    ) -> list[ConversationTurn]:
        session = self._sessions.get(session_id)
        if session:
            return session.get_recent_turns(limit)
        return []

    async def expire_inactive_sessions(
        self,
        inactive_threshold: timedelta,
    ) -> int:
        cutoff = datetime.now(UTC) - inactive_threshold
        expired = 0
        for session in self._sessions.values():
            if session.is_active and session.last_activity_at < cutoff:
                session.is_active = False
                expired += 1
        return expired


class ConversationManager:
    """
    Manages multi-turn conversation sessions.

    Responsibilities:
    - Create and track conversation sessions
    - Maintain conversation context
    - Expand follow-up queries
    - Persist conversation history
    """

    def __init__(
        self,
        storage: ConversationStorage | None = None,
        history_limit: int = 5,
        session_ttl_minutes: int = 30,
    ):
        """
        Initialize the conversation manager.

        Args:
            storage: Optional storage backend (uses in-memory if not provided)
            history_limit: Maximum turns to keep in context
            session_ttl_minutes: Session timeout in minutes
        """
        self._storage = storage or InMemoryConversationStorage()
        self._history_limit = history_limit
        self._session_ttl = timedelta(minutes=session_ttl_minutes)
        self._expander = QueryExpander()

        # Active sessions cache
        self._active_sessions: dict[UUID, ConversationSession] = {}

    async def create_session(
        self,
        user_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> ConversationSession:
        """
        Create a new conversation session.

        Args:
            user_id: Optional user identifier
            config_overrides: Optional configuration overrides

        Returns:
            New ConversationSession
        """
        session = ConversationSession(
            id=uuid4(),
            user_id=user_id,
            config_overrides=config_overrides or {},
        )

        await self._storage.save_session(session)
        self._active_sessions[session.id] = session

        logger.info(f"Created new session: {session.id}")
        return session

    async def get_session(self, session_id: UUID) -> ConversationSession | None:
        """
        Get a session by ID.

        Args:
            session_id: The session UUID

        Returns:
            ConversationSession if found and active, None otherwise
        """
        # Check cache first
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            if self._is_session_expired(session):
                session.is_active = False
                del self._active_sessions[session_id]
                return None
            return session

        # Try storage
        session = await self._storage.get_session(session_id)
        if session and session.is_active:
            if self._is_session_expired(session):
                session.is_active = False
                await self._storage.save_session(session)
                return None

            self._active_sessions[session_id] = session
            return session

        return None

    async def get_or_create_session(
        self,
        session_id: UUID | None = None,
        user_id: str | None = None,
    ) -> ConversationSession:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional existing session ID
            user_id: Optional user identifier for new sessions

        Returns:
            ConversationSession (existing or new)
        """
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session

        return await self.create_session(user_id=user_id)

    def get_context(self, session: ConversationSession) -> ConversationContext:
        """
        Get conversation context for a session.

        Args:
            session: The conversation session

        Returns:
            ConversationContext with extracted information
        """
        return ConversationContext.from_session(
            session,
            max_turns=self._history_limit,
        )

    def expand_query(
        self,
        query: str,
        session: ConversationSession,
    ) -> ExpansionResult:
        """
        Expand a follow-up query using session context.

        Args:
            query: The user's query
            session: The conversation session

        Returns:
            ExpansionResult with expanded query
        """
        context = self.get_context(session)
        return self._expander.expand(query, context)

    async def add_turn(
        self,
        session: ConversationSession,
        turn: ConversationTurn,
    ) -> None:
        """
        Add a turn to a session.

        Args:
            session: The conversation session
            turn: The turn to add
        """
        session.add_turn(turn)
        await self._storage.save_turn(session.id, turn)

        logger.debug(
            f"Added turn {turn.turn_number} to session {session.id}: {turn.user_query[:50]}..."
        )

    async def end_session(self, session_id: UUID) -> None:
        """
        End a conversation session.

        Args:
            session_id: The session to end
        """
        session = await self.get_session(session_id)
        if session:
            session.is_active = False
            await self._storage.save_session(session)

            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

            logger.info(f"Ended session: {session_id}")

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions expired
        """
        # Clean cache
        expired_ids = []
        for session_id, session in self._active_sessions.items():
            if self._is_session_expired(session):
                session.is_active = False
                expired_ids.append(session_id)

        for session_id in expired_ids:
            del self._active_sessions[session_id]

        # Clean storage
        storage_expired = await self._storage.expire_inactive_sessions(self._session_ttl)

        total = len(expired_ids) + storage_expired
        if total > 0:
            logger.info(f"Expired {total} inactive sessions")

        return total

    def _is_session_expired(self, session: ConversationSession) -> bool:
        """Check if a session has expired."""
        if not session.is_active:
            return True

        age = datetime.now(UTC) - session.last_activity_at
        return age > self._session_ttl

    def build_history_prompt(
        self,
        session: ConversationSession,
        max_turns: int | None = None,
    ) -> str:
        """
        Build a history prompt for LLM context.

        Args:
            session: The conversation session
            max_turns: Maximum turns to include (defaults to history_limit)

        Returns:
            Formatted history string for LLM prompt
        """
        limit = max_turns or self._history_limit
        recent = session.get_recent_turns(limit)

        if not recent:
            return ""

        lines = ["Previous conversation:"]
        for turn in recent:
            query = turn.expanded_query or turn.user_query
            lines.append(f"User: {query}")

            if turn.tool_calls:
                # Summarize the response
                entities = list(turn.resolved_entities.keys())
                fields = turn.get_fields()[:3]
                lines.append(f"Assistant: Retrieved {', '.join(fields)} for {', '.join(entities)}")
            elif turn.needs_clarification:
                lines.append("Assistant: Asked for clarification")

        return "\n".join(lines)
