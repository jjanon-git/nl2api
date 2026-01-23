"""
E2E Query Flow Tests

Tests complete user query flows through the NL2API system with real LLM.
Each test verifies: Query → Routing → Agent → Tool Calls

Cost: ~$0.01-0.02 per test run (using Haiku)
"""

import pytest


class TestDatastreamQueries:
    """E2E tests for Datastream agent queries."""

    @pytest.mark.asyncio
    async def test_simple_price_query(self, orchestrator):
        """Query: 'What is Apple's stock price?'"""
        result = await orchestrator.process("What is Apple's stock price?")

        assert not result.needs_clarification, (
            f"Unexpected clarification: {result.clarification_questions}"
        )
        assert result.domain == "datastream"
        assert len(result.tool_calls) >= 1

        tool_call = result.tool_calls[0]
        assert tool_call.tool_name == "get_data"
        assert "tickers" in tool_call.arguments
        assert "fields" in tool_call.arguments
        # Should resolve Apple to an Apple-related ticker
        tickers = tool_call.arguments["tickers"]
        ticker_str = str(tickers).upper()
        assert "AAPL" in ticker_str or "APPLE" in ticker_str, (
            f"Expected Apple ticker, got {tickers}"
        )

    @pytest.mark.asyncio
    async def test_multi_field_query(self, orchestrator):
        """Query: 'Get Microsoft's price, volume, and market cap'"""
        result = await orchestrator.process("Get Microsoft's price, volume, and market cap")

        assert not result.needs_clarification
        assert result.domain == "datastream"
        assert len(result.tool_calls) >= 1

        tool_call = result.tool_calls[0]
        fields = tool_call.arguments.get("fields", [])
        # Should have multiple fields
        assert len(fields) >= 2, f"Expected multiple fields, got {fields}"

    @pytest.mark.asyncio
    async def test_historical_query(self, orchestrator):
        """Query: 'Apple's closing prices for the last month'

        Note: 'last month' is ambiguous and may trigger clarification.
        This is expected behavior.
        """
        result = await orchestrator.process("Apple's closing prices for the last month")

        assert result.domain == "datastream"

        # Ambiguous time reference may trigger clarification - that's OK
        if result.needs_clarification:
            assert any(q.category == "time_period" for q in result.clarification_questions)
        else:
            assert len(result.tool_calls) >= 1
            tool_call = result.tool_calls[0]
            assert "tickers" in tool_call.arguments


class TestEstimatesQueries:
    """E2E tests for Estimates agent queries."""

    @pytest.mark.asyncio
    async def test_eps_estimate_query(self, orchestrator):
        """Query: 'What are analyst EPS estimates for Apple?'"""
        result = await orchestrator.process("What are analyst EPS estimates for Apple?")

        assert not result.needs_clarification
        assert result.domain == "estimates"
        assert len(result.tool_calls) >= 1

        tool_call = result.tool_calls[0]
        assert tool_call.tool_name == "get_data"
        fields = tool_call.arguments.get("fields", [])
        # Should include EPS-related fields
        assert any("EPS" in f.upper() for f in fields), f"Expected EPS field, got {fields}"

    @pytest.mark.asyncio
    async def test_price_target_query(self, orchestrator):
        """Query: 'What is Tesla's analyst price target?'"""
        result = await orchestrator.process("What is Tesla's analyst price target?")

        assert not result.needs_clarification
        assert result.domain == "estimates"
        assert len(result.tool_calls) >= 1

        fields = result.tool_calls[0].arguments.get("fields", [])
        assert any("PriceTarget" in f or "Target" in f for f in fields)

    @pytest.mark.asyncio
    async def test_recommendation_query(self, orchestrator):
        """Query: 'What is the analyst rating for Amazon?'"""
        result = await orchestrator.process("What is the analyst rating for Amazon?")

        assert not result.needs_clarification
        assert result.domain == "estimates"


class TestFundamentalsQueries:
    """E2E tests for Fundamentals agent queries."""

    @pytest.mark.asyncio
    async def test_pe_ratio_query(self, orchestrator):
        """Query: 'What is Apple's PE ratio?'"""
        result = await orchestrator.process("What is Apple's PE ratio?")

        assert not result.needs_clarification
        assert result.domain == "fundamentals"
        assert len(result.tool_calls) >= 1

        fields = result.tool_calls[0].arguments.get("fields", [])
        # Should include PE-related field
        assert any("PE" in f.upper() or "P/E" in f.upper() for f in fields), (
            f"Expected PE field, got {fields}"
        )

    @pytest.mark.asyncio
    async def test_revenue_query(self, orchestrator):
        """Query: 'What was Microsoft's revenue last year?'"""
        result = await orchestrator.process("What was Microsoft's revenue last year?")

        assert not result.needs_clarification
        assert result.domain == "fundamentals"

    @pytest.mark.asyncio
    async def test_balance_sheet_query(self, orchestrator):
        """Query: 'Show me Google's total assets'"""
        result = await orchestrator.process("Show me Google's total assets")

        assert not result.needs_clarification
        assert result.domain == "fundamentals"


class TestScreeningQueries:
    """E2E tests for Screening agent queries."""

    @pytest.mark.asyncio
    async def test_top_n_query(self, orchestrator):
        """Query: 'Top 10 tech stocks by market cap'"""
        result = await orchestrator.process("Top 10 tech stocks by market cap")

        assert not result.needs_clarification
        assert result.domain == "screening"
        assert len(result.tool_calls) >= 1

        tool_call = result.tool_calls[0]
        # Screening returns SCREEN expression in tickers
        tickers = tool_call.arguments.get("tickers", "")
        assert "SCREEN" in str(tickers).upper() or isinstance(tickers, str)

    @pytest.mark.asyncio
    async def test_filter_query(self, orchestrator):
        """Query: 'US stocks with PE ratio below 15'"""
        result = await orchestrator.process("US stocks with PE ratio below 15")

        assert not result.needs_clarification
        assert result.domain == "screening"


class TestOfficersQueries:
    """E2E tests for Officers agent queries."""

    @pytest.mark.asyncio
    async def test_ceo_query(self, orchestrator):
        """Query: 'Who is Apple's CEO?'"""
        result = await orchestrator.process("Who is Apple's CEO?")

        assert not result.needs_clarification
        assert result.domain == "officers"
        assert len(result.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_compensation_query(self, orchestrator):
        """Query: 'What is the CEO compensation at Microsoft?'"""
        result = await orchestrator.process("What is the CEO compensation at Microsoft?")

        assert not result.needs_clarification
        assert result.domain == "officers"


class TestCrossAgentQueries:
    """E2E tests for queries that could route to multiple agents."""

    @pytest.mark.asyncio
    async def test_ambiguous_price_query(self, orchestrator):
        """Query that could be datastream or estimates."""
        result = await orchestrator.process("What is Apple's current stock price?")

        # Should route to datastream for current price
        assert result.domain == "datastream"
        assert len(result.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_multiple_companies(self, orchestrator):
        """Query with multiple companies."""
        result = await orchestrator.process("Compare Apple and Microsoft stock prices")

        assert not result.needs_clarification
        assert len(result.tool_calls) >= 1

        # Should include both companies
        tickers = result.tool_calls[0].arguments.get("tickers", [])
        ticker_str = str(tickers)
        assert "AAPL" in ticker_str or "Apple" in ticker_str
        assert "MSFT" in ticker_str or "Microsoft" in ticker_str


class TestEntityResolution:
    """E2E tests for entity resolution in queries."""

    @pytest.mark.asyncio
    async def test_company_name_resolution(self, orchestrator):
        """Company names should resolve to RICs."""
        result = await orchestrator.process("What is Apple's stock price?")

        assert "Apple" in result.resolved_entities or len(result.tool_calls) > 0
        if result.tool_calls:
            tickers = result.tool_calls[0].arguments.get("tickers", [])
            ticker_str = str(tickers).upper()
            # Should be RIC format or recognized ticker
            assert (
                ".O" in ticker_str
                or ".N" in ticker_str
                or "AAPL" in ticker_str
                or "APPLE" in ticker_str
            )

    @pytest.mark.asyncio
    async def test_ticker_passthrough(self, orchestrator):
        """Ticker symbols should pass through."""
        result = await orchestrator.process("What is AAPL's price?")

        assert len(result.tool_calls) >= 1
        tickers = result.tool_calls[0].arguments.get("tickers", [])
        ticker_str = str(tickers).upper()
        assert "AAPL" in ticker_str or "APPLE" in ticker_str

    @pytest.mark.asyncio
    async def test_index_resolution(self, orchestrator):
        """Index names should resolve correctly."""
        result = await orchestrator.process("What is the S&P 500 level?")

        # Should handle index queries
        assert len(result.tool_calls) >= 1 or result.needs_clarification
