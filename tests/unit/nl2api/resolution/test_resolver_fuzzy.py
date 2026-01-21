"""Unit tests for ExternalEntityResolver normalization and entity extraction."""

import pytest
from src.nl2api.resolution.resolver import ExternalEntityResolver


class TestResolverNormalization:
    """Test suite for company name normalization."""

    def test_normalizes_company_suffixes(self):
        """Test that suffix normalization patterns work."""
        # The normalization happens internally - we test extraction patterns
        resolver = ExternalEntityResolver(use_cache=False)

        # Test entity extraction patterns work with various suffixes
        entities = resolver._extract_entities("Apple Inc. reported earnings")
        assert any("Apple" in e for e in entities)

        entities = resolver._extract_entities("Microsoft Corp announced")
        assert any("Microsoft" in e for e in entities)


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

    def test_extracts_capitalized_names(self):
        """Test extraction of capitalized company names."""
        resolver = ExternalEntityResolver(use_cache=False)
        entities = resolver._extract_entities("Show me Apple and Microsoft earnings")

        assert any("Apple" in e for e in entities)
        assert any("Microsoft" in e for e in entities)

    def test_extracts_ticker_symbols(self):
        """Test extraction of ticker symbols from queries."""
        resolver = ExternalEntityResolver(use_cache=False)
        entities = resolver._extract_entities("What is AAPL trading at?")

        assert "AAPL" in entities

    def test_extracts_company_with_suffix(self):
        """Test extraction handles company suffixes."""
        resolver = ExternalEntityResolver(use_cache=False)
        entities = resolver._extract_entities("Apple Inc reported strong earnings")

        # Should extract "Apple Inc" or similar
        assert any("Apple" in e for e in entities)

    def test_filters_short_strings(self):
        """Test that very short strings are filtered."""
        resolver = ExternalEntityResolver(use_cache=False)
        entities = resolver._extract_entities("I want A stock")

        # Single letter "I" and "A" should be filtered
        assert "I" not in entities
        assert "A" not in entities

    def test_filters_common_uppercase_words(self):
        """Test that common uppercase words are not treated as tickers."""
        resolver = ExternalEntityResolver(use_cache=False)
        entities = resolver._extract_entities("THE EPS FOR AND PE ratio")

        # These common words should be filtered even though uppercase
        assert "THE" not in entities
        assert "FOR" not in entities
        assert "AND" not in entities
        assert "EPS" not in entities
        assert "PE" not in entities


class TestResolverCaching:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test resolver works with cache disabled."""
        resolver = ExternalEntityResolver(use_cache=False)
        # Should not raise even without cache
        result = await resolver.resolve_single("some_company")
        # Result depends on external APIs, just verify no exception


class TestResolverInit:
    """Test resolver initialization."""

    def test_default_initialization(self):
        """Test resolver initializes with defaults."""
        resolver = ExternalEntityResolver()
        assert resolver._use_cache is True
        assert resolver._timeout_seconds == 5.0

    def test_custom_initialization(self):
        """Test resolver initializes with custom params."""
        resolver = ExternalEntityResolver(
            use_cache=False,
            timeout_seconds=10.0,
            circuit_failure_threshold=3,
        )
        assert resolver._use_cache is False
        assert resolver._timeout_seconds == 10.0
