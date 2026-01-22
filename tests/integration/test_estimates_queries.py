"""
Integration tests for EstimatesAgent with sample EPS/revenue queries.

These tests demonstrate the NL2API system working end-to-end with
the EstimatesAgent for common financial estimate queries.

Note: These tests use mock LLM providers and don't require actual API keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from CONTRACTS import ToolCall
from src.nl2api import NL2APIOrchestrator, NL2APITargetAdapter
from src.nl2api.agents.estimates import EstimatesAgent
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from src.nl2api.resolution.protocols import ResolvedEntity


@dataclass
class MockLLMProvider:
    """Mock LLM provider that returns domain classification."""

    model_name: str = "mock-model"

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # Return "estimates" for classification
        return LLMResponse(content="estimates")

    async def complete_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> LLMResponse:
        # If tools are provided, return a tool call
        if tools:
            # Extract info from messages to generate appropriate response
            user_message = next(
                (m.content for m in messages if m.role.value == "user"),
                ""
            )

            # Generate tool call based on query
            if "eps" in user_message.lower() or "earnings" in user_message.lower():
                return LLMResponse(
                    tool_calls=(
                        LLMToolCall(
                            id="call-1",
                            name="get_data",
                            arguments={
                                "tickers": ["AAPL.O"],
                                "fields": ["TR.EPSMean(Period=FY1)"],
                            },
                        ),
                    ),
                )
            elif "revenue" in user_message.lower():
                return LLMResponse(
                    tool_calls=(
                        LLMToolCall(
                            id="call-1",
                            name="get_data",
                            arguments={
                                "tickers": ["AAPL.O"],
                                "fields": ["TR.RevenueMean(Period=FY1)"],
                            },
                        ),
                    ),
                )

        return LLMResponse(content="estimates")


@dataclass
class MockEntityResolver:
    """Mock entity resolver for company to RIC mapping."""

    mappings: dict[str, str] = field(default_factory=lambda: {
        "Apple": "AAPL.O",
        "AAPL": "AAPL.O",
        "Microsoft": "MSFT.O",
        "MSFT": "MSFT.O",
        "Tesla": "TSLA.O",
        "Amazon": "AMZN.O",
        "Google": "GOOGL.O",
        "Alphabet": "GOOGL.O",
    })

    async def resolve(self, query: str) -> dict[str, str]:
        """Resolve entities in the query."""
        result = {}
        for name, ric in self.mappings.items():
            if name.lower() in query.lower():
                result[name] = ric
        return result

    async def resolve_single(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> ResolvedEntity | None:
        """Resolve a single entity."""
        ric = self.mappings.get(entity)
        if ric:
            return ResolvedEntity(
                original=entity,
                identifier=ric,
                entity_type="company",
            )
        return None

    async def resolve_batch(
        self,
        entities: list[str],
    ) -> list[ResolvedEntity]:
        """Resolve multiple entities in batch."""
        results = []
        for entity in entities:
            resolved = await self.resolve_single(entity)
            if resolved:
                results.append(resolved)
        return results


class TestSampleEPSQueries:
    """Test EPS-related queries through the full NL2API stack."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_resolver = MockEntityResolver()
        self.agent = EstimatesAgent(llm=self.mock_llm)
        self.orchestrator = NL2APIOrchestrator(
            llm=self.mock_llm,
            agents={"estimates": self.agent},
            entity_resolver=self.mock_resolver,
        )

    @pytest.mark.asyncio
    async def test_simple_eps_query(self) -> None:
        """Test: 'What is Apple's EPS estimate?'"""
        result = await self.orchestrator.process(
            "What is Apple's EPS estimate?"
        )

        assert not result.needs_clarification
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_data"
        assert "AAPL.O" in result.tool_calls[0].arguments["tickers"]
        assert any("EPSMean" in f for f in result.tool_calls[0].arguments["fields"])
        assert result.domain == "estimates"

    @pytest.mark.asyncio
    async def test_earnings_estimate_query(self) -> None:
        """Test: 'Get Microsoft's earnings forecast'"""
        result = await self.orchestrator.process(
            "Get Microsoft's earnings forecast"
        )

        assert not result.needs_clarification
        assert len(result.tool_calls) == 1
        assert "MSFT.O" in result.tool_calls[0].arguments["tickers"]
        # earnings maps to EPS
        assert any("EPSMean" in f for f in result.tool_calls[0].arguments["fields"])

    @pytest.mark.asyncio
    async def test_quarterly_eps_query(self) -> None:
        """Test: 'What is Tesla's EPS estimate for next quarter?'

        Note: "next quarter" is ambiguous - the system asks for clarification
        to get a specific time period like Q1 2024 or FQ1.
        """
        result = await self.orchestrator.process(
            "What is Tesla's EPS estimate for next quarter?"
        )

        # Ambiguous temporal reference triggers clarification
        assert result.needs_clarification
        assert result.domain == "estimates"
        assert "Tesla" in result.resolved_entities
        # The clarification should be about time period
        assert any(q.category == "time_period" for q in result.clarification_questions)

    @pytest.mark.asyncio
    async def test_eps_with_explicit_period(self) -> None:
        """Test: 'Get Apple's EPS estimate for FY2'"""
        # Note: Rule-based extraction should detect "two years"
        result = await self.orchestrator.process(
            "Get Apple's EPS estimate for two years ahead"
        )

        assert not result.needs_clarification
        fields = result.tool_calls[0].arguments["fields"]
        assert any("FY2" in f for f in fields)


class TestSampleRevenueQueries:
    """Test revenue-related queries through the full NL2API stack."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_resolver = MockEntityResolver()
        self.agent = EstimatesAgent(llm=self.mock_llm)
        self.orchestrator = NL2APIOrchestrator(
            llm=self.mock_llm,
            agents={"estimates": self.agent},
            entity_resolver=self.mock_resolver,
        )

    @pytest.mark.asyncio
    async def test_simple_revenue_query(self) -> None:
        """Test: 'What is Apple's revenue estimate?'"""
        result = await self.orchestrator.process(
            "What is Apple's revenue estimate?"
        )

        assert not result.needs_clarification
        assert len(result.tool_calls) == 1
        assert "AAPL.O" in result.tool_calls[0].arguments["tickers"]
        assert any("RevenueMean" in f for f in result.tool_calls[0].arguments["fields"])

    @pytest.mark.asyncio
    async def test_sales_forecast_query(self) -> None:
        """Test: 'Get Microsoft's sales forecast'"""
        result = await self.orchestrator.process(
            "Get Microsoft's sales forecast"
        )

        assert not result.needs_clarification
        # "sales" should map to revenue
        assert any("RevenueMean" in f for f in result.tool_calls[0].arguments["fields"])

    @pytest.mark.asyncio
    async def test_quarterly_revenue_query(self) -> None:
        """Test: 'What is Amazon's revenue forecast for next quarter?'

        Note: "next quarter" is ambiguous - the system asks for clarification
        to get a specific time period.
        """
        result = await self.orchestrator.process(
            "What is Amazon's revenue forecast for next quarter?"
        )

        # Ambiguous temporal reference triggers clarification
        assert result.needs_clarification
        assert result.domain == "estimates"
        assert "Amazon" in result.resolved_entities
        assert any(q.category == "time_period" for q in result.clarification_questions)


class TestSampleAnalystQueries:
    """Test analyst rating/recommendation queries."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_resolver = MockEntityResolver()
        self.agent = EstimatesAgent(llm=self.mock_llm)
        self.orchestrator = NL2APIOrchestrator(
            llm=self.mock_llm,
            agents={"estimates": self.agent},
            entity_resolver=self.mock_resolver,
        )

    @pytest.mark.asyncio
    async def test_analyst_rating_query(self) -> None:
        """Test: 'What is Tesla's analyst rating?'"""
        result = await self.orchestrator.process(
            "What is Tesla's analyst rating?"
        )

        assert not result.needs_clarification
        fields = result.tool_calls[0].arguments["fields"]
        # Should include recommendation fields
        assert any("RecMean" in f for f in fields)

    @pytest.mark.asyncio
    async def test_price_target_query(self) -> None:
        """Test: 'What is Apple's price target?'"""
        result = await self.orchestrator.process(
            "What is Apple's price target?"
        )

        assert not result.needs_clarification
        fields = result.tool_calls[0].arguments["fields"]
        assert any("PriceTarget" in f for f in fields)


class TestSampleSurpriseQueries:
    """Test earnings surprise queries."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_resolver = MockEntityResolver()
        self.agent = EstimatesAgent(llm=self.mock_llm)
        self.orchestrator = NL2APIOrchestrator(
            llm=self.mock_llm,
            agents={"estimates": self.agent},
            entity_resolver=self.mock_resolver,
        )

    @pytest.mark.asyncio
    async def test_earnings_beat_query(self) -> None:
        """Test: 'Did Amazon beat earnings last quarter?'

        Note: "last quarter" is ambiguous - the system asks for clarification
        to get a specific time period like Q4 2023.
        """
        result = await self.orchestrator.process(
            "Did Amazon beat earnings last quarter?"
        )

        # Ambiguous temporal reference triggers clarification
        assert result.needs_clarification
        assert result.domain == "estimates"
        assert "Amazon" in result.resolved_entities
        assert any(q.category == "time_period" for q in result.clarification_questions)

    @pytest.mark.asyncio
    async def test_eps_surprise_query(self) -> None:
        """Test: 'What was Tesla's EPS surprise?'"""
        result = await self.orchestrator.process(
            "What was Tesla's EPS surprise?"
        )

        assert not result.needs_clarification
        fields = result.tool_calls[0].arguments["fields"]
        assert any("SurprisePct" in f for f in fields)


class TestEvaluationAdapterIntegration:
    """Test the evaluation adapter with sample queries."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_resolver = MockEntityResolver()
        self.agent = EstimatesAgent(llm=self.mock_llm)
        self.orchestrator = NL2APIOrchestrator(
            llm=self.mock_llm,
            agents={"estimates": self.agent},
            entity_resolver=self.mock_resolver,
        )
        self.adapter = NL2APITargetAdapter(self.orchestrator)

    @pytest.mark.asyncio
    async def test_adapter_eps_query(self) -> None:
        """Test adapter with EPS query."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        assert response.raw_output != "[]"
        tool_calls = json.loads(response.raw_output)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "get_data"

    @pytest.mark.asyncio
    async def test_adapter_metadata(self) -> None:
        """Test adapter includes proper metadata."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        assert response.execution_data is not None
        assert response.execution_data["domain"] == "estimates"
        assert response.execution_data["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_adapter_latency_tracked(self) -> None:
        """Test adapter tracks latency."""
        response = await self.adapter.invoke("What is Apple's EPS estimate?")

        assert response.latency_ms >= 0


class TestMultipleCompanies:
    """Test queries involving multiple companies."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.mock_resolver = MockEntityResolver()
        self.agent = EstimatesAgent(llm=self.mock_llm)
        self.orchestrator = NL2APIOrchestrator(
            llm=self.mock_llm,
            agents={"estimates": self.agent},
            entity_resolver=self.mock_resolver,
        )

    @pytest.mark.asyncio
    async def test_two_company_comparison(self) -> None:
        """Test: 'Compare EPS estimates for Apple and Microsoft'"""
        result = await self.orchestrator.process(
            "Compare EPS estimates for Apple and Microsoft"
        )

        assert not result.needs_clarification
        rics = result.tool_calls[0].arguments["tickers"]
        assert "AAPL.O" in rics
        assert "MSFT.O" in rics
