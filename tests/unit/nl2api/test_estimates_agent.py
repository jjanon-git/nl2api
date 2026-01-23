"""Tests for EstimatesAgent."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.nl2api.agents.estimates import EstimatesAgent
from src.nl2api.agents.protocols import AgentContext
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from src.nl2api.rag.indexer import (
    FieldCodeDocument,
    QueryExampleDocument,
    parse_estimates_reference,
    parse_query_examples,
)


@dataclass
class MockLLMProvider:
    """Mock LLM provider for testing."""

    model_name: str = "mock-model"
    response: LLMResponse = field(default_factory=lambda: LLMResponse())

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Return the configured mock response."""
        return self.response

    async def complete_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Return the configured mock response."""
        return self.response


class TestEstimatesAgentCanHandle:
    """Test suite for EstimatesAgent.can_handle()."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = EstimatesAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_high_confidence_with_multiple_keywords(self) -> None:
        """Test high confidence when multiple estimate keywords present."""
        query = "What is the EPS estimate forecast for Apple?"
        confidence = await self.agent.can_handle(query)
        assert confidence >= 0.7

    @pytest.mark.asyncio
    async def test_very_high_confidence_with_many_keywords(self) -> None:
        """Test very high confidence with 3+ keywords."""
        query = "Get analyst rating recommendation and price target for Tesla"
        confidence = await self.agent.can_handle(query)
        assert confidence >= 0.9

    @pytest.mark.asyncio
    async def test_medium_confidence_with_single_keyword(self) -> None:
        """Test medium confidence with single keyword."""
        query = "Get the EPS for Apple"
        confidence = await self.agent.can_handle(query)
        assert 0.4 <= confidence <= 0.6

    @pytest.mark.asyncio
    async def test_zero_confidence_for_unrelated_query(self) -> None:
        """Test zero confidence for unrelated query."""
        query = "What is the current stock price of AAPL?"
        confidence = await self.agent.can_handle(query)
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_handles_surprise_keywords(self) -> None:
        """Test detection of earnings surprise keywords."""
        query = "Did Amazon beat earnings last quarter?"
        confidence = await self.agent.can_handle(query)
        assert confidence > 0.0

    @pytest.mark.asyncio
    async def test_handles_recommendation_keywords(self) -> None:
        """Test detection of analyst recommendation keywords."""
        query = "What is the analyst buy rating for Microsoft?"
        confidence = await self.agent.can_handle(query)
        assert confidence >= 0.7


class TestEstimatesAgentPeriodDetection:
    """Test suite for EstimatesAgent._detect_period()."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = EstimatesAgent(llm=self.mock_llm)

    def test_next_quarter_detection(self) -> None:
        """Test detection of next quarter period."""
        assert self.agent._detect_period("next quarter earnings") == "FQ1"
        assert self.agent._detect_period("quarterly forecast") == "FQ1"

    def test_last_quarter_detection(self) -> None:
        """Test detection of last quarter period."""
        assert self.agent._detect_period("last quarter results") == "FQ0"
        assert self.agent._detect_period("previous quarter earnings") == "FQ0"

    def test_fy2_detection(self) -> None:
        """Test detection of FY2 period."""
        assert self.agent._detect_period("next year estimates") == "FY2"
        assert self.agent._detect_period("two years ahead") == "FY2"

    def test_fy3_detection(self) -> None:
        """Test detection of FY3 period."""
        assert self.agent._detect_period("three years estimate") == "FY3"

    def test_default_to_fy1(self) -> None:
        """Test default to FY1 when no period specified."""
        assert self.agent._detect_period("what is apple's eps?") == "FY1"
        assert self.agent._detect_period("get revenue estimate") == "FY1"


class TestEstimatesAgentFieldDetection:
    """Test suite for EstimatesAgent._detect_fields()."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = EstimatesAgent(llm=self.mock_llm)

    def test_eps_detection(self) -> None:
        """Test EPS field detection."""
        fields = self.agent._detect_fields("what is the eps estimate", "FY1")
        assert "TR.EPSMean(Period=FY1)" in fields

    def test_revenue_detection(self) -> None:
        """Test revenue field detection."""
        fields = self.agent._detect_fields("get revenue forecast", "FY1")
        assert "TR.RevenueMean(Period=FY1)" in fields

    def test_ebitda_detection(self) -> None:
        """Test EBITDA field detection."""
        fields = self.agent._detect_fields("what is the ebitda estimate", "FY1")
        assert "TR.EBITDAMean(Period=FY1)" in fields

    def test_earnings_maps_to_eps(self) -> None:
        """Test 'earnings' keyword maps to EPS."""
        fields = self.agent._detect_fields("earnings estimate", "FY1")
        assert "TR.EPSMean(Period=FY1)" in fields

    def test_sales_maps_to_revenue(self) -> None:
        """Test 'sales' keyword maps to revenue."""
        fields = self.agent._detect_fields("sales forecast", "FY1")
        assert "TR.RevenueMean(Period=FY1)" in fields

    def test_surprise_detection(self) -> None:
        """Test surprise field detection."""
        fields = self.agent._detect_fields("did it beat earnings", "FY1")
        assert "TR.EPSSurprisePct(Period=FQ0)" in fields

    def test_recommendation_package(self) -> None:
        """Test full recommendation package detection."""
        fields = self.agent._detect_fields("analyst rating", "FY1")
        assert "TR.RecMean" in fields
        assert "TR.NumBuys" in fields
        assert "TR.NumHolds" in fields
        assert "TR.NumSells" in fields

    def test_price_target_fields(self) -> None:
        """Test price target field detection."""
        fields = self.agent._detect_fields("price target", "FY1")
        assert "TR.PriceTargetMean" in fields
        assert "TR.PriceTargetHigh" in fields
        assert "TR.PriceTargetLow" in fields

    def test_forward_pe_detection(self) -> None:
        """Test forward P/E detection."""
        fields = self.agent._detect_fields("forward pe ratio", "FY1")
        assert "TR.PtoEPSMeanEst(Period=FY1)" in fields

    def test_peg_ratio_detection(self) -> None:
        """Test PEG ratio detection."""
        fields = self.agent._detect_fields("what is the peg ratio", "FY1")
        assert "TR.PEGRatio" in fields

    def test_ltg_detection(self) -> None:
        """Test long-term growth detection."""
        fields = self.agent._detect_fields("long-term growth estimate", "FY1")
        assert "TR.LTGMean" in fields

    def test_no_duplicates(self) -> None:
        """Test that fields are deduplicated."""
        fields = self.agent._detect_fields("eps earnings estimate", "FY1")
        eps_count = sum(1 for f in fields if "EPSMean" in f)
        assert eps_count == 1


class TestEstimatesAgentRuleBasedExtraction:
    """Test suite for EstimatesAgent._try_rule_based_extraction()."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = EstimatesAgent(llm=self.mock_llm)

    def test_simple_eps_query(self) -> None:
        """Test rule-based extraction for simple EPS query."""
        context = AgentContext(
            query="What is Apple's EPS estimate?",
            resolved_entities={"Apple": "AAPL.O"},
        )
        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        assert result.confidence >= 0.8
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_data"
        assert result.tool_calls[0].arguments["tickers"] == ["AAPL.O"]
        assert "TR.EPSMean(Period=FY1)" in result.tool_calls[0].arguments["fields"]

    def test_quarterly_revenue_query(self) -> None:
        """Test rule-based extraction for quarterly revenue."""
        context = AgentContext(
            query="Get Microsoft's revenue forecast for next quarter",
            resolved_entities={"Microsoft": "MSFT.O"},
        )
        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        assert "TR.RevenueMean(Period=FQ1)" in result.tool_calls[0].arguments["fields"]

    def test_multiple_rics(self) -> None:
        """Test rule-based extraction with multiple RICs."""
        context = AgentContext(
            query="Compare EPS estimates for Apple and Microsoft",
            resolved_entities={"Apple": "AAPL.O", "Microsoft": "MSFT.O"},
        )
        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        rics = result.tool_calls[0].arguments["tickers"]
        assert "AAPL.O" in rics
        assert "MSFT.O" in rics

    def test_no_entities_returns_none(self) -> None:
        """Test that missing entities returns None."""
        context = AgentContext(
            query="What is Apple's EPS estimate?",
            resolved_entities={},  # No entities
        )
        result = self.agent._try_rule_based_extraction(context)

        assert result is None

    def test_no_fields_detected_returns_none(self) -> None:
        """Test that query without detectable fields returns None."""
        context = AgentContext(
            query="What about the company?",
            resolved_entities={"Apple": "AAPL.O"},
        )
        result = self.agent._try_rule_based_extraction(context)

        assert result is None

    def test_analyst_rating_query(self) -> None:
        """Test rule-based extraction for analyst ratings."""
        context = AgentContext(
            query="What is the analyst rating for Tesla?",
            resolved_entities={"Tesla": "TSLA.O"},
        )
        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        fields = result.tool_calls[0].arguments["fields"]
        assert "TR.RecMean" in fields
        assert "TR.NumBuys" in fields

    def test_earnings_surprise_query(self) -> None:
        """Test rule-based extraction for earnings surprise."""
        context = AgentContext(
            query="Did Amazon beat earnings last quarter?",
            resolved_entities={"Amazon": "AMZN.O"},
        )
        result = self.agent._try_rule_based_extraction(context)

        assert result is not None
        fields = result.tool_calls[0].arguments["fields"]
        assert any("SurprisePct" in f for f in fields)


class TestEstimatesAgentProcess:
    """Test suite for EstimatesAgent.process()."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Configure mock to return a tool call
        self.mock_response = LLMResponse(
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="get_data",
                    arguments={
                        "RICs": ["AAPL.O"],
                        "fields": ["TR.EPSMean(Period=FY1)"],
                    },
                ),
            ),
            usage={"total_tokens": 100},
        )
        self.mock_llm = MockLLMProvider(response=self.mock_response)
        self.agent = EstimatesAgent(llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_rule_based_takes_precedence(self) -> None:
        """Test that rule-based extraction is preferred when confident."""
        context = AgentContext(
            query="What is Apple's EPS estimate?",
            resolved_entities={"Apple": "AAPL.O"},
        )
        result = await self.agent.process(context)

        # Should use rule-based, not LLM
        assert result.confidence >= 0.8
        assert "Rule-based" in result.reasoning

    @pytest.mark.asyncio
    async def test_llm_fallback_for_complex_queries(self) -> None:
        """Test that LLM is used for complex queries."""
        context = AgentContext(
            query="Show me historical EPS estimates and revisions for the past year",
            resolved_entities={"Apple": "AAPL.O"},
        )
        result = await self.agent.process(context)

        # This query doesn't match simple rules, should fall back to LLM
        assert len(result.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_domain_name(self) -> None:
        """Test that domain name is set correctly."""
        context = AgentContext(
            query="What is Apple's EPS estimate?",
            resolved_entities={"Apple": "AAPL.O"},
        )
        result = await self.agent.process(context)

        assert result.domain == "estimates"


class TestEstimatesAgentProperties:
    """Test suite for EstimatesAgent properties."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.agent = EstimatesAgent(llm=self.mock_llm)

    def test_domain_name(self) -> None:
        """Test domain_name property."""
        assert self.agent.domain_name == "estimates"

    def test_domain_description(self) -> None:
        """Test domain_description property."""
        desc = self.agent.domain_description
        assert "I/B/E/S" in desc
        assert "Estimates" in desc

    def test_system_prompt(self) -> None:
        """Test get_system_prompt returns useful content."""
        prompt = self.agent.get_system_prompt()
        assert "TR.EPSMean" in prompt
        assert "Period=FY1" in prompt
        assert "get_data" in prompt

    def test_tools_definition(self) -> None:
        """Test get_tools returns proper tool definition."""
        tools = self.agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "get_data"  # Canonical format
        assert "tickers" in tools[0].parameters["properties"]  # Canonical argument key
        assert "fields" in tools[0].parameters["properties"]


class TestParseEstimatesReference:
    """Test suite for parse_estimates_reference()."""

    def test_parses_table_format(self) -> None:
        """Test parsing markdown table format."""
        content = """
## Field Codes

| Natural Language | TR Code | Description |
|------------------|---------|-------------|
| EPS, earnings per share | `TR.EPSMean` | Mean EPS estimate |
| Revenue, sales | `TR.RevenueMean` | Mean revenue estimate |
"""
        docs = parse_estimates_reference(content)

        assert len(docs) == 2
        assert docs[0].field_code == "TR.EPSMean"
        assert docs[0].description == "Mean EPS estimate"
        assert "EPS" in docs[0].natural_language_hints
        assert docs[0].domain == "estimates"

    def test_skips_header_rows(self) -> None:
        """Test that header rows are skipped."""
        content = """
| Natural Language | TR Code | Description |
|------------------|---------|-------------|
| metric | `header` | should skip |
| EPS | `TR.EPSMean` | Mean EPS estimate |
"""
        docs = parse_estimates_reference(content)

        # Should only have 1 doc, not the header row
        assert len(docs) == 1
        assert docs[0].field_code == "TR.EPSMean"

    def test_extracts_keywords(self) -> None:
        """Test keyword extraction from natural language column."""
        content = """
| Natural Language | TR Code | Description |
|------------------|---------|-------------|
| eps, earnings, profit estimate | `TR.EPSMean` | Mean EPS |
"""
        docs = parse_estimates_reference(content)

        assert len(docs[0].natural_language_hints) == 3
        assert "eps" in docs[0].natural_language_hints
        assert "earnings" in docs[0].natural_language_hints

    def test_sets_source_metadata(self) -> None:
        """Test that source metadata is set."""
        content = """
| Natural Language | TR Code | Description |
|------------------|---------|-------------|
| EPS | `TR.EPSMean` | Mean EPS estimate |
"""
        docs = parse_estimates_reference(content)

        assert docs[0].metadata is not None
        assert docs[0].metadata["source"] == "estimates.md"


class TestParseQueryExamples:
    """Test suite for parse_query_examples()."""

    def test_parses_qa_format(self) -> None:
        """Test parsing Q&A format."""
        content = """
## Examples

**Q1:** "What is Apple's EPS estimate?"
```python
get_data(RICs=["AAPL.O"], fields=["TR.EPSMean(Period=FY1)"])
```

**Q2:** "Get Microsoft's revenue forecast"
```python
get_data(RICs=["MSFT.O"], fields=["TR.RevenueMean(Period=FY1)"])
```
"""
        docs = parse_query_examples(content)

        assert len(docs) == 2
        assert docs[0].query == "What is Apple's EPS estimate?"
        assert "get_data" in docs[0].api_call
        assert docs[0].domain == "estimates"

    def test_sets_complexity_level(self) -> None:
        """Test complexity level detection from context."""
        content = """
## Level 2 - Intermediate

**Q1:** "Complex query here"
```python
get_data(RICs=["AAPL.O"], fields=["TR.EPSMean"])
```
"""
        docs = parse_query_examples(content)

        assert len(docs) == 1
        assert docs[0].complexity_level == 2

    def test_default_complexity(self) -> None:
        """Test default complexity level."""
        content = """
**Q1:** "Simple query"
```python
get_data(RICs=["AAPL.O"], fields=["TR.EPSMean"])
```
"""
        docs = parse_query_examples(content)

        assert docs[0].complexity_level == 1

    def test_sets_source_metadata(self) -> None:
        """Test that source metadata is set."""
        content = """
**Q1:** "Test query"
```python
get_data(RICs=["AAPL.O"], fields=["TR.EPSMean"])
```
"""
        docs = parse_query_examples(content)

        assert docs[0].metadata is not None
        assert docs[0].metadata["source"] == "estimates.md"


class TestFieldCodeDocument:
    """Test suite for FieldCodeDocument dataclass."""

    def test_creation(self) -> None:
        """Test FieldCodeDocument creation."""
        doc = FieldCodeDocument(
            field_code="TR.EPSMean",
            description="Mean EPS estimate",
            domain="estimates",
            natural_language_hints=["eps", "earnings"],
        )

        assert doc.field_code == "TR.EPSMean"
        assert doc.description == "Mean EPS estimate"
        assert doc.domain == "estimates"
        assert len(doc.natural_language_hints) == 2

    def test_optional_metadata(self) -> None:
        """Test optional metadata field."""
        doc = FieldCodeDocument(
            field_code="TR.EPSMean",
            description="Mean EPS",
            domain="estimates",
            natural_language_hints=[],
            metadata={"source": "test"},
        )

        assert doc.metadata == {"source": "test"}

    def test_default_metadata(self) -> None:
        """Test default metadata is None."""
        doc = FieldCodeDocument(
            field_code="TR.EPSMean",
            description="Mean EPS",
            domain="estimates",
            natural_language_hints=[],
        )

        assert doc.metadata is None


class TestQueryExampleDocument:
    """Test suite for QueryExampleDocument dataclass."""

    def test_creation(self) -> None:
        """Test QueryExampleDocument creation."""
        doc = QueryExampleDocument(
            query="What is Apple's EPS?",
            api_call="get_data(RICs=['AAPL.O'])",
            domain="estimates",
        )

        assert doc.query == "What is Apple's EPS?"
        assert doc.api_call == "get_data(RICs=['AAPL.O'])"
        assert doc.domain == "estimates"
        assert doc.complexity_level == 1  # Default

    def test_with_complexity(self) -> None:
        """Test with custom complexity level."""
        doc = QueryExampleDocument(
            query="Complex query",
            api_call="get_data(...)",
            domain="estimates",
            complexity_level=3,
        )

        assert doc.complexity_level == 3

    def test_with_metadata(self) -> None:
        """Test with metadata."""
        doc = QueryExampleDocument(
            query="Test",
            api_call="get_data(...)",
            domain="estimates",
            metadata={"tags": ["test"]},
        )

        assert doc.metadata == {"tags": ["test"]}
