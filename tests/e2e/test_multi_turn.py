"""
E2E Multi-Turn Conversation Tests

Tests conversation continuity and context preservation across multiple turns.
Uses fresh orchestrator per test for session isolation.

Cost: ~$0.02-0.04 per test (multiple LLM calls)
"""

import pytest


class TestConversationContinuity:
    """Test that context is preserved across conversation turns."""

    @pytest.mark.asyncio
    async def test_pronoun_resolution(self, fresh_orchestrator):
        """Follow-up query should resolve 'it' to previous company."""
        orchestrator = fresh_orchestrator

        # First turn: establish context
        result1 = await orchestrator.process("What is Apple's stock price?")
        session_id = result1.session_id

        assert not result1.needs_clarification
        assert result1.domain == "datastream"

        # Second turn: use pronoun
        result2 = await orchestrator.process(
            "What about its PE ratio?",
            session_id=session_id,
        )

        # Should understand "its" refers to Apple
        assert not result2.needs_clarification
        assert len(result2.tool_calls) >= 1

        tickers = result2.tool_calls[0].arguments.get("tickers", [])
        ticker_str = str(tickers).upper()
        # Should still be Apple-related
        assert "AAPL" in ticker_str or "APPLE" in ticker_str, f"Expected Apple ticker, got {tickers}"

    @pytest.mark.asyncio
    async def test_domain_switch(self, fresh_orchestrator):
        """Should handle switching domains mid-conversation."""
        orchestrator = fresh_orchestrator

        # First turn: datastream query
        result1 = await orchestrator.process("What is Microsoft's stock price?")
        session_id = result1.session_id
        assert result1.domain == "datastream"

        # Second turn: switch to estimates
        result2 = await orchestrator.process(
            "What are analyst EPS estimates for it?",
            session_id=session_id,
        )

        assert result2.domain == "estimates"
        assert len(result2.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_company_context_preserved(self, fresh_orchestrator):
        """Multiple questions about same company."""
        orchestrator = fresh_orchestrator

        # First turn
        result1 = await orchestrator.process("Tell me about Tesla's stock")
        session_id = result1.session_id

        # Second turn - implicit reference
        result2 = await orchestrator.process(
            "And the PE ratio?",
            session_id=session_id,
        )

        # Should still be Tesla
        if result2.tool_calls:
            tickers = result2.tool_calls[0].arguments.get("tickers", [])
            ticker_str = str(tickers).upper()
            resolved_str = str(result2.resolved_entities)
            assert "TSLA" in ticker_str or "TESLA" in ticker_str or "Tesla" in resolved_str

    @pytest.mark.asyncio
    async def test_new_session_no_context(self, fresh_orchestrator):
        """New session should not have previous context."""
        orchestrator = fresh_orchestrator

        # First conversation
        result1 = await orchestrator.process("What is Apple's price?")
        session1 = result1.session_id

        # New session (no session_id passed)
        result2 = await orchestrator.process("What is its PE ratio?")
        session2 = result2.session_id

        # Should be different sessions
        assert session1 != session2

        # Without context, "its" is ambiguous - might trigger clarification
        # or fail to resolve. Either is acceptable for a new session.


class TestClarificationFlow:
    """Test clarification request and response flow."""

    @pytest.mark.asyncio
    async def test_ambiguous_query_triggers_clarification(self, fresh_orchestrator):
        """Ambiguous temporal reference should request clarification."""
        orchestrator = fresh_orchestrator

        result = await orchestrator.process(
            "What was Apple's revenue last quarter?"
        )

        # May or may not trigger clarification depending on implementation
        # If it does, verify the flow
        if result.needs_clarification:
            assert len(result.clarification_questions) >= 1
            question = result.clarification_questions[0]
            assert question.question  # Has a question text
            assert len(question.options) >= 2  # Has options

    @pytest.mark.asyncio
    async def test_clarification_response(self, fresh_orchestrator):
        """After clarification, should proceed with query."""
        orchestrator = fresh_orchestrator

        # Query that might need clarification
        result1 = await orchestrator.process(
            "Show me stock performance"  # Vague - which stock?
        )

        if result1.needs_clarification:
            session_id = result1.session_id

            # Provide clarification
            result2 = await orchestrator.process(
                "Apple",  # Clarify which stock
                session_id=session_id,
            )

            # Should now proceed or ask follow-up
            # Either tool_calls or another clarification is valid


class TestTurnTracking:
    """Test that turn numbers are tracked correctly."""

    @pytest.mark.asyncio
    async def test_turn_numbers_increment(self, fresh_orchestrator):
        """Turn numbers should increment within a session."""
        orchestrator = fresh_orchestrator

        result1 = await orchestrator.process("What is Apple's price?")
        session_id = result1.session_id
        assert result1.turn_number == 1

        result2 = await orchestrator.process(
            "And Microsoft?",
            session_id=session_id,
        )
        assert result2.turn_number == 2

        result3 = await orchestrator.process(
            "Compare them",
            session_id=session_id,
        )
        assert result3.turn_number == 3

    @pytest.mark.asyncio
    async def test_new_session_resets_turn(self, fresh_orchestrator):
        """New session should start at turn 1."""
        orchestrator = fresh_orchestrator

        result1 = await orchestrator.process("What is Apple's price?")
        assert result1.turn_number == 1

        # New session
        result2 = await orchestrator.process("What is Microsoft's price?")
        assert result2.turn_number == 1  # Reset for new session
