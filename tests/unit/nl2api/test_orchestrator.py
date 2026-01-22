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
    """Mock domain agent that implements the DomainAgent protocol."""

    domain_name: str = "estimates"
    domain_description: str = "Test agent"
    capabilities: tuple[str, ...] = ("EPS estimates", "revenue forecasts")
    example_queries: tuple[str, ...] = ("What is Apple's EPS?",)
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
    """Mock entity resolver that implements the EntityResolver protocol."""

    entities: dict[str, str] = field(default_factory=lambda: {"Apple": "AAPL.O"})

    async def resolve(self, query: str) -> dict[str, str]:
        result = {}
        for name, ric in self.entities.items():
            if name.lower() in query.lower():
                result[name] = ric
        return result

    async def resolve_single(self, entity: str, entity_type: str | None = None) -> Any:
        """Resolve a single entity."""
        from src.nl2api.resolution.protocols import ResolvedEntity

        for name, ric in self.entities.items():
            if name.lower() == entity.lower():
                return ResolvedEntity(
                    original=entity,
                    identifier=ric,
                    entity_type=entity_type or "company",
                )
        return None

    async def resolve_batch(self, entities: list[str]) -> list[Any]:
        """Resolve multiple entities in batch."""
        results = []
        for entity in entities:
            result = await self.resolve_single(entity)
            if result:
                results.append(result)
        return results


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


class TestOrchestratorDualModeContext:
    """Test suite for orchestrator dual-mode context retrieval."""

    @dataclass
    class MockContextProvider:
        """Mock context provider for testing."""

        field_codes: list = field(default_factory=list)
        examples: list = field(default_factory=list)
        calls: list = field(default_factory=list)

        async def get_field_codes(self, query, domain, limit=5):
            self.calls.append(("field_codes", query, domain, limit))
            return self.field_codes

        async def get_query_examples(self, query, domain, limit=3):
            self.calls.append(("examples", query, domain, limit))
            return self.examples

    @pytest.mark.asyncio
    async def test_orchestrator_uses_context_retriever(self) -> None:
        """Test that orchestrator uses the context retriever when provided."""
        llm = MockLLMProvider()
        agent = MockAgent()
        captured_context = None

        async def capture_context(context):
            nonlocal captured_context
            captured_context = context
            return AgentResult(
                tool_calls=(ToolCall(tool_name="get_data", arguments={}),),
                confidence=0.9,
                domain="estimates",
            )

        agent.process = capture_context

        context_provider = self.MockContextProvider(
            field_codes=[{"code": "TR.EPSMean", "description": "EPS Mean", "source": "test"}],
            examples=[{"query": "What is EPS?", "api_call": "get_data()", "source": "test"}],
        )

        orchestrator = NL2APIOrchestrator(
            llm=llm,
            agents={"estimates": agent},
            context_retriever=context_provider,
        )

        await orchestrator.process("What is Apple's EPS?")

        # Context provider should have been called
        assert len(context_provider.calls) == 2
        assert context_provider.calls[0][0] == "field_codes"
        assert context_provider.calls[1][0] == "examples"

        # Context should have been passed to agent
        assert captured_context is not None
        assert len(captured_context.field_codes) == 1
        assert captured_context.field_codes[0]["code"] == "TR.EPSMean"

    @pytest.mark.asyncio
    async def test_orchestrator_creates_dual_mode_retriever_from_rag(self) -> None:
        """Test that orchestrator creates DualModeContextRetriever from RAG."""
        from src.nl2api.rag.protocols import DocumentType, RetrievalResult

        llm = MockLLMProvider()
        agent = MockAgent()
        captured_context = None

        async def capture_context(context):
            nonlocal captured_context
            captured_context = context
            return AgentResult(
                tool_calls=(ToolCall(tool_name="get_data", arguments={}),),
                confidence=0.9,
                domain="estimates",
            )

        agent.process = capture_context

        @dataclass
        class MockRAG:
            async def retrieve(
                self,
                query: str,
                domain: str | None = None,
                document_types: list | None = None,
                limit: int = 10,
                threshold: float = 0.5,
            ) -> list:
                return []

            async def retrieve_field_codes(self, query, domain=None, limit=10):
                return [
                    RetrievalResult(
                        id="fc-001",
                        document_type=DocumentType.FIELD_CODE,
                        content="Mean EPS",
                        score=0.9,
                        field_code="TR.EPSMean",
                    )
                ]

            async def retrieve_examples(self, query, domain=None, limit=5):
                return [
                    RetrievalResult(
                        id="qe-001",
                        document_type=DocumentType.QUERY_EXAMPLE,
                        content="Q: EPS?",
                        score=0.8,
                        example_query="EPS?",
                        example_api_call="get_data(...)",
                    )
                ]

        orchestrator = NL2APIOrchestrator(
            llm=llm,
            agents={"estimates": agent},
            rag=MockRAG(),
            context_mode="local",  # Use RAG mode
        )

        await orchestrator.process("What is Apple's EPS?")

        # Context should have been retrieved from RAG
        assert captured_context is not None
        assert len(captured_context.field_codes) == 1
        assert captured_context.field_codes[0]["code"] == "TR.EPSMean"

    @pytest.mark.asyncio
    async def test_orchestrator_context_mode_hybrid(self) -> None:
        """Test orchestrator in hybrid context mode."""
        llm = MockLLMProvider()
        agent = MockAgent()
        captured_context = None

        async def capture_context(context):
            nonlocal captured_context
            captured_context = context
            return AgentResult(
                tool_calls=(ToolCall(tool_name="get_data", arguments={}),),
                confidence=0.9,
                domain="estimates",
            )

        agent.process = capture_context

        # Create mock MCP retriever
        @dataclass
        class MockMCPRetriever:
            async def get_field_codes(self, query, domain, limit=5):
                return [{"code": "MCP_CODE", "description": "From MCP", "source": "mcp"}]

            async def get_query_examples(self, query, domain, limit=3):
                return [{"query": "MCP query", "api_call": "mcp_call()", "source": "mcp"}]

        orchestrator = NL2APIOrchestrator(
            llm=llm,
            agents={"estimates": agent},
            rag=None,  # No RAG
            mcp_retriever=MockMCPRetriever(),
            context_mode="mcp",  # Use MCP mode
        )

        await orchestrator.process("What is Apple's EPS?")

        # Context should have been retrieved from MCP
        assert captured_context is not None
        assert len(captured_context.field_codes) == 1
        assert captured_context.field_codes[0]["code"] == "MCP_CODE"
        assert captured_context.field_codes[0]["source"] == "mcp"

    @pytest.mark.asyncio
    async def test_orchestrator_context_retrieval_handles_errors(self) -> None:
        """Test that context retrieval errors are handled gracefully."""
        llm = MockLLMProvider()
        agent = MockAgent()
        captured_context = None

        async def capture_context(context):
            nonlocal captured_context
            captured_context = context
            return AgentResult(
                tool_calls=(ToolCall(tool_name="get_data", arguments={}),),
                confidence=0.9,
                domain="estimates",
            )

        agent.process = capture_context

        # Create error-raising context provider
        @dataclass
        class ErrorContextProvider:
            async def get_field_codes(self, query, domain, limit=5):
                raise Exception("Context retrieval failed")

            async def get_query_examples(self, query, domain, limit=3):
                raise Exception("Context retrieval failed")

        orchestrator = NL2APIOrchestrator(
            llm=llm,
            agents={"estimates": agent},
            context_retriever=ErrorContextProvider(),
        )

        # Should not raise, should continue with empty context
        result = await orchestrator.process("What is Apple's EPS?")

        assert result.tool_calls is not None
        assert captured_context is not None
        # Context should be empty due to error
        assert len(captured_context.field_codes) == 0

    @pytest.mark.asyncio
    async def test_orchestrator_without_context_retriever(self) -> None:
        """Test orchestrator works without any context retriever."""
        llm = MockLLMProvider()
        agent = MockAgent()
        captured_context = None

        async def capture_context(context):
            nonlocal captured_context
            captured_context = context
            return AgentResult(
                tool_calls=(ToolCall(tool_name="get_data", arguments={}),),
                confidence=0.9,
                domain="estimates",
            )

        agent.process = capture_context

        orchestrator = NL2APIOrchestrator(
            llm=llm,
            agents={"estimates": agent},
            rag=None,
            context_retriever=None,
        )

        await orchestrator.process("What is Apple's EPS?")

        # Should work with empty context
        assert captured_context is not None
        assert len(captured_context.field_codes) == 0
        assert len(captured_context.query_examples) == 0


class TestOrchestratorRoutingModel:
    """Test suite for routing model configuration."""

    def test_config_defaults_to_haiku_for_routing(self) -> None:
        """Config should default to Haiku for routing (cost optimization)."""
        from src.nl2api.config import NL2APIConfig

        cfg = NL2APIConfig()
        assert cfg.routing_model == "claude-3-5-haiku-20241022"
        assert cfg.routing_model != cfg.llm_model  # Should differ from main model

    def test_orchestrator_uses_injected_llm_for_default_router(self) -> None:
        """Orchestrator should use injected LLM for default router (no hidden config dependency).

        This is a regression test for a bug where _create_default_router() created a new
        NL2APIConfig() which looked for NL2API_ANTHROPIC_API_KEY from environment,
        causing failures when only ANTHROPIC_API_KEY was set.

        The fix: orchestrator always uses the injected LLM for routing. If you need
        a different model for routing, create the router externally and pass it in.
        """
        main_llm = MockLLMProvider(model_name="sonnet")
        agent = MockAgent()

        # Create orchestrator without router - it should create default router
        # using the SAME injected LLM (not a new one from config)
        orchestrator = NL2APIOrchestrator(
            llm=main_llm,
            agents={"estimates": agent},
        )

        # Verify the router uses the same LLM instance
        assert orchestrator._router is not None
        assert orchestrator._router._llm is main_llm

    def test_orchestrator_accepts_custom_router(self) -> None:
        """Orchestrator should accept pre-configured router for custom routing model.

        If you want a different model for routing (e.g., Haiku for cost savings),
        create the router externally with that model and pass it to orchestrator.
        """
        from src.nl2api.routing.llm_router import LLMToolRouter
        from src.nl2api.routing.providers import AgentToolProvider

        main_llm = MockLLMProvider(model_name="sonnet")
        routing_llm = MockLLMProvider(model_name="haiku")
        agent = MockAgent()

        # Create router with different LLM
        custom_router = LLMToolRouter(
            llm=routing_llm,
            tool_providers=[AgentToolProvider(agent)],
        )

        # Pass custom router to orchestrator
        orchestrator = NL2APIOrchestrator(
            llm=main_llm,
            agents={"estimates": agent},
            router=custom_router,
        )

        # Verify custom router is used
        assert orchestrator._router is custom_router
        assert orchestrator._router._llm is routing_llm
        # Main LLM is still used for other operations
        assert orchestrator._llm is main_llm
