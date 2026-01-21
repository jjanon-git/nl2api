"""Tests for NL2API orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from CONTRACTS import ToolCall
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
)
from src.nl2api.models import NL2APIResponse
from src.nl2api.orchestrator import NL2APIOrchestrator


@dataclass
class MockLLMProvider:
    """Mock LLM provider."""

    model_name: str = "mock"
    response_content: str = "estimates"

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return LLMResponse(content=self.response_content)

    async def complete_with_retry(self, messages, tools=None, temperature=0.0, max_tokens=4096, max_retries=3):
        return LLMResponse(content=self.response_content)


@dataclass
class MockAgent:
    """Mock domain agent."""

    domain_name: str = "estimates"
    domain_description: str = "Test agent"
    can_handle_score: float = 0.9
    result: AgentResult = field(default_factory=lambda: AgentResult(
        tool_calls=(ToolCall(tool_name="get_data", arguments={"RICs": ["AAPL.O"]}),),
        confidence=0.9,
        domain="estimates",
    ))

    async def can_handle(self, query: str) -> float:
        return self.can_handle_score

    async def process(self, context: AgentContext) -> AgentResult:
        return self.result


@dataclass
class MockEntityResolver:
    """Mock entity resolver."""

    entities: dict[str, str] = field(default_factory=lambda: {"Apple": "AAPL.O"})

    async def resolve(self, query: str) -> dict[str, str]:
        result = {}
        for name, ric in self.entities.items():
            if name.lower() in query.lower():
                result[name] = ric
        return result


class TestNL2APIOrchestrator:
    """Test suite for NL2APIOrchestrator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.llm = MockLLMProvider()
        self.agent = MockAgent()
        self.resolver = MockEntityResolver()
        self.orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            entity_resolver=self.resolver,
        )

    @pytest.mark.asyncio
    async def test_process_returns_nl2api_response(self) -> None:
        """Test that process returns NL2APIResponse."""
        result = await self.orchestrator.process("What is Apple's EPS?")

        assert isinstance(result, NL2APIResponse)

    @pytest.mark.asyncio
    async def test_process_includes_tool_calls(self) -> None:
        """Test that successful processing includes tool calls."""
        result = await self.orchestrator.process("What is Apple's EPS?")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_data"

    @pytest.mark.asyncio
    async def test_process_resolves_entities(self) -> None:
        """Test that entities are resolved."""
        result = await self.orchestrator.process("What is Apple's EPS?")

        assert "Apple" in result.resolved_entities
        assert result.resolved_entities["Apple"] == "AAPL.O"

    @pytest.mark.asyncio
    async def test_process_sets_domain(self) -> None:
        """Test that domain is set on response."""
        result = await self.orchestrator.process("What is Apple's EPS?")

        assert result.domain == "estimates"

    @pytest.mark.asyncio
    async def test_process_tracks_timing(self) -> None:
        """Test that processing time is tracked."""
        result = await self.orchestrator.process("What is Apple's EPS?")

        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_classify_uses_agent_scores(self) -> None:
        """Test that classification uses agent can_handle scores."""
        # Agent returns high score, should be selected
        self.agent.can_handle_score = 0.95

        result = await self.orchestrator.process("What is Apple's EPS?")

        assert result.domain == "estimates"

    @pytest.mark.asyncio
    async def test_classify_falls_back_to_llm(self) -> None:
        """Test that classification falls back to LLM when scores are low."""
        self.agent.can_handle_score = 0.3  # Low score
        self.llm.response_content = "estimates"

        result = await self.orchestrator.process("What is Apple's EPS?")

        # Should still route to estimates based on LLM classification
        assert result.domain == "estimates"

    @pytest.mark.asyncio
    async def test_unknown_domain_returns_clarification(self) -> None:
        """Test that unknown domain returns clarification request."""
        self.agent.can_handle_score = 0.0
        self.llm.response_content = "unknown"

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={},  # No agents
        )

        result = await orchestrator.process("Random query")

        assert result.needs_clarification

    @pytest.mark.asyncio
    async def test_agent_clarification_propagated(self) -> None:
        """Test that agent clarification requests are propagated."""
        self.agent.result = AgentResult(
            needs_clarification=True,
            clarification_questions=("Which company?",),
            domain="estimates",
        )

        result = await self.orchestrator.process("What is the EPS?")

        assert result.needs_clarification
        assert len(result.clarification_questions) == 1

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test error handling during processing."""
        async def raise_error(context):
            raise RuntimeError("Test error")

        self.agent.process = raise_error

        result = await self.orchestrator.process("What is Apple's EPS?")

        # Should return clarification with error message
        assert result.needs_clarification

    @pytest.mark.asyncio
    async def test_register_agent(self) -> None:
        """Test registering a new agent."""
        new_agent = MockAgent(domain_name="fundamentals", domain_description="Test")

        self.orchestrator.register_agent(new_agent)

        assert "fundamentals" in self.orchestrator.get_domains()

    def test_get_domains(self) -> None:
        """Test getting available domains."""
        domains = self.orchestrator.get_domains()

        assert "estimates" in domains

    @pytest.mark.asyncio
    async def test_process_without_entity_resolver(self) -> None:
        """Test processing without entity resolver."""
        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": self.agent},
            entity_resolver=None,
        )

        result = await orchestrator.process("What is Apple's EPS?")

        assert result.resolved_entities == {}

    @pytest.mark.asyncio
    async def test_multiple_agents_routing(self) -> None:
        """Test routing between multiple agents."""
        estimates_agent = MockAgent(domain_name="estimates", can_handle_score=0.9)
        fundamentals_agent = MockAgent(domain_name="fundamentals", can_handle_score=0.3)

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={
                "estimates": estimates_agent,
                "fundamentals": fundamentals_agent,
            },
        )

        result = await orchestrator.process("EPS forecast")

        # Should route to estimates (higher score)
        assert result.domain == "estimates"


class TestOrchestratorClassification:
    """Test suite for query classification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.llm = MockLLMProvider()

    @pytest.mark.asyncio
    async def test_keyword_classification_high_confidence(self) -> None:
        """Test keyword-based classification with high confidence."""
        agent = MockAgent(can_handle_score=0.85)
        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": agent},
        )

        # Should use keyword classification, not LLM
        result = await orchestrator.process("EPS estimate")

        assert result.domain == "estimates"

    @pytest.mark.asyncio
    async def test_llm_classification_for_ambiguous_query(self) -> None:
        """Test LLM classification for ambiguous queries."""
        agent = MockAgent(can_handle_score=0.4)  # Low confidence
        self.llm.response_content = "estimates"

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": agent},
        )

        result = await orchestrator.process("What about the company data?")

        # Should still route correctly
        assert result.domain == "estimates"

    @pytest.mark.asyncio
    async def test_fuzzy_domain_matching(self) -> None:
        """Test fuzzy matching of domain names from LLM."""
        agent = MockAgent(can_handle_score=0.3)
        self.llm.response_content = "I think this is about estimates"

        orchestrator = NL2APIOrchestrator(
            llm=self.llm,
            agents={"estimates": agent},
        )

        result = await orchestrator.process("Query")

        # Should fuzzy match "estimates" from response
        assert result.domain == "estimates"


class TestOrchestratorWithRAG:
    """Test suite for orchestrator with RAG integration."""

    @dataclass
    class MockRAGRetriever:
        """Mock RAG retriever."""

        field_codes: list = field(default_factory=list)
        examples: list = field(default_factory=list)

        async def retrieve_field_codes(self, query, domain=None, limit=10):
            return self.field_codes

        async def retrieve_examples(self, query, domain=None, limit=5):
            return self.examples

        async def retrieve(self, query, domain=None, doc_type=None, limit=10):
            return self.field_codes + self.examples

    @pytest.mark.asyncio
    async def test_rag_context_passed_to_agent(self) -> None:
        """Test that RAG context is passed to the agent."""
        from src.nl2api.rag.protocols import DocumentType, RetrievalResult

        llm = MockLLMProvider()
        captured_context = None

        async def capture_context(context):
            nonlocal captured_context
            captured_context = context
            return AgentResult(
                tool_calls=(ToolCall(tool_name="get_data", arguments={}),),
                confidence=0.9,
                domain="estimates",
            )

        agent = MockAgent()
        agent.process = capture_context

        rag = self.MockRAGRetriever(
            field_codes=[
                RetrievalResult(
                    id="fc-001",
                    document_type=DocumentType.FIELD_CODE,
                    content="Mean EPS",
                    score=0.9,
                    field_code="TR.EPSMean",
                )
            ],
            examples=[
                RetrievalResult(
                    id="qe-001",
                    document_type=DocumentType.QUERY_EXAMPLE,
                    content="Q: EPS?",
                    score=0.8,
                    example_query="EPS?",
                    example_api_call="get_data(...)",
                )
            ],
        )

        orchestrator = NL2APIOrchestrator(
            llm=llm,
            agents={"estimates": agent},
            rag=rag,
        )

        await orchestrator.process("What is Apple's EPS?")

        assert captured_context is not None
        assert len(captured_context.field_codes) == 1
        assert len(captured_context.query_examples) == 1
