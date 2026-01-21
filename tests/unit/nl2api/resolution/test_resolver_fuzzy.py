"""Unit tests for ExternalEntityResolver fuzzy matching and normalization."""

import pytest
from src.nl2api.resolution.resolver import ExternalEntityResolver


class TestResolverNormalization:
    """Test suite for company name normalization."""

    @pytest.mark.asyncio
    async def test_strips_inc_suffix(self):
        """Test stripping Inc. suffix."""
        resolver = ExternalEntityResolver(use_cache=False)
        result = await resolver.resolve_single("Apple Inc.")
        assert result is not None
        assert result.identifier == "AAPL.O"

    @pytest.mark.asyncio
    async def test_strips_corp_suffix(self):
        """Test stripping Corp suffix."""
        resolver = ExternalEntityResolver(use_cache=False)
        result = await resolver.resolve_single("Microsoft Corp")
        assert result is not None
        assert result.identifier == "MSFT.O"

    @pytest.mark.asyncio
    async def test_strips_and_co_suffix(self):
        """Test stripping & Co suffix."""
        resolver = ExternalEntityResolver(use_cache=False, fuzzy_threshold=80)
        result = await resolver.resolve_single("JP Morgan & Co")
        assert result is not None
        assert result.identifier == "JPM.N"

    @pytest.mark.asyncio
    async def test_strips_and_company_suffix(self):
        """Test stripping & Company suffix."""
        resolver = ExternalEntityResolver(use_cache=False, fuzzy_threshold=80)
        result = await resolver.resolve_single("JP Morgan & Company")
        assert result is not None
        assert result.identifier == "JPM.N"

    @pytest.mark.asyncio
    async def test_strips_combined_suffixes(self):
        """Test stripping combined suffixes like '& Co Inc.'."""
        resolver = ExternalEntityResolver(use_cache=False, fuzzy_threshold=80)
        result = await resolver.resolve_single("JP Morgan Chase & Co Inc")
        assert result is not None
        assert result.identifier == "JPM.N"


class TestResolverIgnoreWords:
    """Test that common words are filtered out."""

    @pytest.mark.asyncio
    async def test_ignores_common_query_words(self):
        """Test that common query words are not resolved."""
        resolver = ExternalEntityResolver(use_cache=False)

        # These should return None, not be treated as company names
        for word in ["forecast", "eps", "revenue", "price", "stock"]:
            result = await resolver.resolve_single(word)
            assert result is None, f"'{word}' should be ignored"

    @pytest.mark.asyncio
    async def test_query_with_only_common_words(self):
        """Test that queries with only common words return empty results."""
        resolver = ExternalEntityResolver(use_cache=False)
        # Use a query without any pattern that could match company names
        result = await resolver.resolve("what is the best forecast for revenue")
        assert result == {}


class TestResolverCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_stats_available(self):
        """Test that circuit breaker stats are accessible."""
        resolver = ExternalEntityResolver()
        stats = resolver.circuit_breaker_stats

        assert "state" in stats
        assert "failure_count" in stats
        assert "total_calls" in stats

    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        resolver = ExternalEntityResolver()
        resolver.reset_circuit_breaker()

        stats = resolver.circuit_breaker_stats
        assert stats["failure_count"] == 0


class TestResolverEntityExtraction:
    """Test entity extraction from queries."""

    @pytest.mark.asyncio
    async def test_extracts_capitalized_names(self):
        """Test extraction of capitalized company names."""
        resolver = ExternalEntityResolver(use_cache=False)
        result = await resolver.resolve("Show me Apple and Microsoft earnings")

        assert "Apple" in result
        assert "Microsoft" in result

    @pytest.mark.asyncio
    async def test_extracts_ticker_symbols(self):
        """Test extraction of ticker symbols from queries."""
        resolver = ExternalEntityResolver(use_cache=False)
        result = await resolver.resolve("What is AAPL trading at?")

        assert "AAPL" in result
        assert result["AAPL"] == "AAPL.O"

    @pytest.mark.asyncio
    async def test_extracts_from_possessive(self):
        """Test extraction handles possessive forms."""
        resolver = ExternalEntityResolver(use_cache=False)
        result = await resolver.resolve("Apple's revenue growth")

        assert "Apple" in result


@pytest.mark.asyncio
async def test_resolver_exact_match():
    """Test exact matches from mappings."""
    resolver = ExternalEntityResolver(use_cache=False)
    
    # Primary name
    result = await resolver.resolve_single("Apple")
    assert result is not None
    assert result.identifier == "AAPL.O"
    
    # Alias
    result = await resolver.resolve_single("Google")
    assert result is not None
    assert result.identifier == "GOOGL.O"


@pytest.mark.asyncio
async def test_resolver_ticker_match():
    """Test ticker matches."""
    resolver = ExternalEntityResolver(use_cache=False)
    
    result = await resolver.resolve_single("AAPL")
    assert result is not None
    assert result.identifier == "AAPL.O"
    assert result.entity_type == "ticker"


@pytest.mark.asyncio
async def test_resolver_fuzzy_match():
    """Test fuzzy matching for near-matches."""
    # Use threshold of 75 to avoid boundary issues with scores exactly at 80
    resolver = ExternalEntityResolver(use_cache=False, fuzzy_threshold=75)

    # Slight misspelling or variation
    result = await resolver.resolve_single("Appel Inc")
    assert result is not None, "Should fuzzy match 'Appel' to 'Apple'"
    assert result.identifier == "AAPL.O"

    result = await resolver.resolve_single("Microsft")
    assert result is not None, "Should fuzzy match 'Microsft' to 'Microsoft'"
    assert result.identifier == "MSFT.O"

    result = await resolver.resolve_single("JP Morgan Chase & Co")
    assert result is not None, "Should fuzzy match 'JP Morgan Chase' to 'jpmorgan chase'"
    assert result.identifier == "JPM.N"


@pytest.mark.asyncio
async def test_resolver_extract_and_resolve():
    """Test end-to-end extraction and resolution."""
    resolver = ExternalEntityResolver(use_cache=False)
    
    query = "What is the EPS forecast for Apple and Microsoft?"
    resolved = await resolver.resolve(query)
    
    assert "Apple" in resolved
    assert resolved["Apple"] == "AAPL.O"
    assert "Microsoft" in resolved
    assert resolved["Microsoft"] == "MSFT.O"
