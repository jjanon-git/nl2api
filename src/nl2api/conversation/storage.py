"""
PostgreSQL Conversation Storage

Persistent storage for conversation sessions using PostgreSQL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from CONTRACTS import ToolCall
from src.nl2api.conversation.models import ConversationSession, ConversationTurn

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


class PostgresConversationStorage:
    """
    PostgreSQL-backed conversation storage.

    Uses the conversations and conversation_sessions tables
    defined in migration 003_conversations.sql.
    """

    def __init__(self, pool: "asyncpg.Pool"):
        """
        Initialize with database connection pool.

        Args:
            pool: asyncpg connection pool
        """
        self._pool = pool

    async def get_session(self, session_id: UUID) -> ConversationSession | None:
        """Get a session by ID with its turns."""
        async with self._pool.acquire() as conn:
            # Get session
            session_row = await conn.fetchrow(
                """
                SELECT id, user_id, started_at, last_activity_at,
                       is_active, total_turns, context_summary, config_overrides
                FROM conversation_sessions
                WHERE id = $1
                """,
                session_id,
            )

            if not session_row:
                return None

            # Get turns
            turn_rows = await conn.fetch(
                """
                SELECT turn_number, user_query, expanded_query, tool_calls,
                       resolved_entities, domain, confidence, needs_clarification,
                       clarification_questions, clarification_response,
                       processing_time_ms, created_at
                FROM conversations
                WHERE session_id = $1
                ORDER BY turn_number ASC
                """,
                session_id,
            )

            turns = [self._row_to_turn(row) for row in turn_rows]

            return ConversationSession(
                id=session_row["id"],
                user_id=session_row["user_id"],
                started_at=session_row["started_at"],
                last_activity_at=session_row["last_activity_at"],
                is_active=session_row["is_active"],
                turns=turns,
                context_summary=session_row["context_summary"],
                config_overrides=session_row["config_overrides"] or {},
            )

    async def save_session(self, session: ConversationSession) -> None:
        """Save or update a session."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_sessions
                    (id, user_id, started_at, last_activity_at, is_active,
                     total_turns, context_summary, config_overrides)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    last_activity_at = EXCLUDED.last_activity_at,
                    is_active = EXCLUDED.is_active,
                    total_turns = EXCLUDED.total_turns,
                    context_summary = EXCLUDED.context_summary,
                    config_overrides = EXCLUDED.config_overrides
                """,
                session.id,
                session.user_id,
                session.started_at,
                session.last_activity_at,
                session.is_active,
                session.total_turns,
                session.context_summary,
                json.dumps(session.config_overrides) if session.config_overrides else None,
            )

    async def save_turn(self, session_id: UUID, turn: ConversationTurn) -> None:
        """Save a turn to a session."""
        async with self._pool.acquire() as conn:
            # Convert tool calls to JSON
            tool_calls_json = None
            if turn.tool_calls:
                tool_calls_json = json.dumps([
                    {
                        "tool_name": tc.tool_name,
                        "arguments": dict(tc.arguments) if tc.arguments else {},
                    }
                    for tc in turn.tool_calls
                ])

            await conn.execute(
                """
                INSERT INTO conversations
                    (session_id, turn_number, user_query, expanded_query,
                     tool_calls, resolved_entities, domain, confidence,
                     needs_clarification, clarification_questions,
                     clarification_response, processing_time_ms, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (session_id, turn_number) DO UPDATE SET
                    expanded_query = EXCLUDED.expanded_query,
                    tool_calls = EXCLUDED.tool_calls,
                    resolved_entities = EXCLUDED.resolved_entities,
                    domain = EXCLUDED.domain,
                    confidence = EXCLUDED.confidence,
                    needs_clarification = EXCLUDED.needs_clarification,
                    clarification_questions = EXCLUDED.clarification_questions,
                    clarification_response = EXCLUDED.clarification_response,
                    processing_time_ms = EXCLUDED.processing_time_ms
                """,
                session_id,
                turn.turn_number,
                turn.user_query,
                turn.expanded_query,
                tool_calls_json,
                json.dumps(turn.resolved_entities) if turn.resolved_entities else None,
                turn.domain,
                turn.confidence,
                turn.needs_clarification,
                json.dumps(list(turn.clarification_questions)) if turn.clarification_questions else None,
                turn.clarification_response,
                turn.processing_time_ms,
                turn.created_at,
            )

    async def get_recent_turns(
        self,
        session_id: UUID,
        limit: int = 5,
    ) -> list[ConversationTurn]:
        """Get recent turns for a session."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT turn_number, user_query, expanded_query, tool_calls,
                       resolved_entities, domain, confidence, needs_clarification,
                       clarification_questions, clarification_response,
                       processing_time_ms, created_at
                FROM conversations
                WHERE session_id = $1
                ORDER BY turn_number DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )

            # Reverse to get chronological order
            turns = [self._row_to_turn(row) for row in reversed(rows)]
            return turns

    async def expire_inactive_sessions(
        self,
        inactive_threshold: timedelta,
    ) -> int:
        """Mark inactive sessions as expired."""
        cutoff = datetime.utcnow() - inactive_threshold

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversation_sessions
                SET is_active = FALSE
                WHERE is_active = TRUE
                  AND last_activity_at < $1
                """,
                cutoff,
            )

            # Parse "UPDATE N" to get count
            try:
                count = int(result.split()[-1])
                return count
            except (ValueError, IndexError):
                return 0

    def _row_to_turn(self, row: Any) -> ConversationTurn:
        """Convert a database row to ConversationTurn."""
        # Parse tool calls
        tool_calls = ()
        if row["tool_calls"]:
            tc_data = row["tool_calls"]
            if isinstance(tc_data, str):
                tc_data = json.loads(tc_data)
            tool_calls = tuple(
                ToolCall(
                    tool_name=tc["tool_name"],
                    arguments=tc.get("arguments", {}),
                )
                for tc in tc_data
            )

        # Parse resolved entities
        entities = {}
        if row["resolved_entities"]:
            entities = row["resolved_entities"]
            if isinstance(entities, str):
                entities = json.loads(entities)

        # Parse clarification questions
        clarification_questions = ()
        if row["clarification_questions"]:
            cq_data = row["clarification_questions"]
            if isinstance(cq_data, str):
                cq_data = json.loads(cq_data)
            clarification_questions = tuple(cq_data)

        return ConversationTurn(
            turn_number=row["turn_number"],
            user_query=row["user_query"],
            expanded_query=row["expanded_query"],
            tool_calls=tool_calls,
            resolved_entities=entities,
            domain=row["domain"],
            confidence=row["confidence"] or 0.0,
            needs_clarification=row["needs_clarification"] or False,
            clarification_questions=clarification_questions,
            clarification_response=row["clarification_response"],
            processing_time_ms=row["processing_time_ms"] or 0,
            created_at=row["created_at"],
        )
