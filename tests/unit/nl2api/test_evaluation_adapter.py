"""Tests for NL2API evaluation adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from CONTRACTS import SystemResponse, ToolCall
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.evaluation.adapter import NL2APITargetAdapter, NL2APIBatchAdapter
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from src.nl2api.models import ClarificationQuestion, NL2APIResponse
from src.nl2api.orchestrator import NL2APIOrchestrator


@dataclass
class MockLLMProvider:
    """Mock LLM provider for testing."""

    model_name: str = "mock-model"
    response: LLMResponse = field(default_factory=lambda: LLMResponse(content="estimates"))

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        return self.response

    async def complete_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> LLMResponse:
        return self.response


@dataclass
class MockAgent:
    """Mock domain agent for testing."""

    domain_name: str = "estimates"
    domain_description: str = "Test estimates agent"
    result: AgentResult = field(default_factory=lambda: AgentResult(
        tool_calls=(
            ToolCall(tool_name="get_data", arguments={"RICs": ["AAPL.O"], "fields": ["TR.EPSMean"]}),
        ),
        confidence=0.9,
        reasoning="Test reasoning",
        domain="estimates",
    ))
    can_handle_score: float = 0.9

    async def process(self, context: AgentContext) -> AgentResult:
        return self.result

    async def can_handle(self, query: str) -> float:
        return self.can_handle_score


class TestNL2APITargetAdapter:
    """Test suite for NL2APITargetAdapter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_agent = MockAgent()
        self.orchestrator = NL2APIOrchestrator(
            llm=self.mock_llm,
            agents={"estimates": self.mock_agent},
        )
        self.adapter = NL2APITargetAdapter(self.orchestrator)

    @pytest.mark.asyncio
    async def test_invoke_returns_system_response(self) -> None:
        """Test that invoke returns a SystemResponse."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        assert isinstance(response, SystemResponse)
        assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_successful_invocation_has_tool_calls(self) -> None:
        """Test that successful invocation includes tool calls in raw_output."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        # Parse raw_output as JSON
        tool_calls = json.loads(response.raw_output)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "get_data"
        assert tool_calls[0]["arguments"]["RICs"] == ["AAPL.O"]

    @pytest.mark.asyncio
    async def test_parsed_tool_calls_are_set(self) -> None:
        """Test that parsed_tool_calls are set on success."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        assert response.parsed_tool_calls is not None
        assert len(response.parsed_tool_calls) == 1
        assert response.parsed_tool_calls[0].tool_name == "get_data"

    @pytest.mark.asyncio
    async def test_execution_data_includes_metadata(self) -> None:
        """Test that execution_data includes NL2API metadata."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        assert response.execution_data is not None
        assert response.execution_data["domain"] == "estimates"
        assert response.execution_data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_reasoning_in_nl_response(self) -> None:
        """Test that reasoning is included in nl_response."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        assert response.nl_response == "Test reasoning"

    @pytest.mark.asyncio
    async def test_clarification_returns_empty_tool_calls(self) -> None:
        """Test that clarification responses return empty tool calls."""
        # Configure agent to return clarification
        self.mock_agent.result = AgentResult(
            needs_clarification=True,
            clarification_questions=("Which company did you mean?",),
            domain="estimates",
        )

        response = await self.adapter.invoke("What is the EPS?")

        assert response.raw_output == "[]"
        assert response.parsed_tool_calls is None
        assert response.error is not None
        assert "clarification" in response.error.lower()

    @pytest.mark.asyncio
    async def test_error_returns_empty_response(self) -> None:
        """Test that errors return empty response with error message."""
        # Configure agent to raise exception
        async def raise_error(context):
            raise ValueError("Test error")

        self.mock_agent.process = raise_error

        response = await self.adapter.invoke("What is Apple's EPS?")

        # Error during agent processing triggers orchestrator error handling
        # which returns a clarification response
        assert response.raw_output == "[]"
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_session_tracking(self) -> None:
        """Test that session tracking works for multi-turn."""
        adapter = NL2APITargetAdapter(self.orchestrator, session_id="test-session")

        # First turn
        await adapter.invoke("What is Apple's EPS estimate?")

        # History should be updated
        assert len(adapter._conversation_history) == 1
        assert adapter._conversation_history[0]["query"] == "What is Apple's EPS estimate?"

    @pytest.mark.asyncio
    async def test_reset_conversation_clears_history(self) -> None:
        """Test that reset_conversation clears history."""
        adapter = NL2APITargetAdapter(self.orchestrator, session_id="test-session")

        await adapter.invoke("Test query")
        assert len(adapter._conversation_history) == 1

        adapter.reset_conversation()
        assert len(adapter._conversation_history) == 0

    @pytest.mark.asyncio
    async def test_set_session_id_clears_history_when_none(self) -> None:
        """Test that setting session_id to None clears history."""
        adapter = NL2APITargetAdapter(self.orchestrator, session_id="test-session")

        await adapter.invoke("Test query")
        adapter.set_session_id(None)

        assert adapter._session_id is None
        assert len(adapter._conversation_history) == 0


class TestNL2APIBatchAdapter:
    """Test suite for NL2APIBatchAdapter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_agent = MockAgent()
        self.orchestrator_created = False

        async def create_orchestrator():
            self.orchestrator_created = True
            return NL2APIOrchestrator(
                llm=self.mock_llm,
                agents={"estimates": self.mock_agent},
            )

        self.factory = create_orchestrator
        self.adapter = NL2APIBatchAdapter(self.factory)

    @pytest.mark.asyncio
    async def test_lazy_orchestrator_creation(self) -> None:
        """Test that orchestrator is created lazily on first invoke."""
        assert not self.orchestrator_created
        assert self.adapter._orchestrator is None

        await self.adapter.invoke("Test query")

        assert self.orchestrator_created
        assert self.adapter._orchestrator is not None

    @pytest.mark.asyncio
    async def test_orchestrator_reused(self) -> None:
        """Test that orchestrator is reused across invocations."""
        await self.adapter.invoke("Query 1")
        first_orchestrator = self.adapter._orchestrator

        await self.adapter.invoke("Query 2")
        second_orchestrator = self.adapter._orchestrator

        assert first_orchestrator is second_orchestrator

    @pytest.mark.asyncio
    async def test_reset_clears_orchestrator(self) -> None:
        """Test that reset clears the orchestrator."""
        await self.adapter.invoke("Test query")
        assert self.adapter._orchestrator is not None

        await self.adapter.reset()
        assert self.adapter._orchestrator is None

    @pytest.mark.asyncio
    async def test_invoke_returns_system_response(self) -> None:
        """Test that invoke returns proper SystemResponse."""
        response = await self.adapter.invoke("What is Apple's EPS?")

        assert isinstance(response, SystemResponse)
        tool_calls = json.loads(response.raw_output)
        assert len(tool_calls) == 1


class TestTargetSystemProtocol:
    """Test that adapters satisfy the TargetSystem protocol."""

    def test_nl2api_adapter_is_target_system(self) -> None:
        """Test that NL2APITargetAdapter satisfies TargetSystem protocol."""
        from src.nl2api.evaluation.adapter import TargetSystem

        mock_llm = MockLLMProvider()
        mock_agent = MockAgent()
        orchestrator = NL2APIOrchestrator(llm=mock_llm, agents={"estimates": mock_agent})
        adapter = NL2APITargetAdapter(orchestrator)

        assert isinstance(adapter, TargetSystem)
