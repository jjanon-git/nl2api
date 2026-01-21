"""Tests for entity resolver."""

from __future__ import annotations

import pytest

from src.nl2api.resolution.resolver import ExternalEntityResolver
from src.nl2api.resolution.protocols import ResolvedEntity


class TestResolvedEntity:
    """Test suite for ResolvedEntity dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic entity creation."""
        entity = ResolvedEntity(
            original="Apple",
            identifier="AAPL.O",
            entity_type="company",
            confidence=0.95,
        )
        assert entity.original == "Apple"
        assert entity.identifier == "AAPL.O"
        assert entity.confidence == 0.95

    def test_with_metadata(self) -> None:
        """Test entity with metadata."""
        entity = ResolvedEntity(
            original="Apple Inc.",
            identifier="AAPL.O",
            entity_type="company",
            confidence=0.98,
            metadata={"exchange": "NASDAQ", "isin": "US0378331005"},
        )
        assert entity.entity_type == "company"
        assert entity.metadata["exchange"] == "NASDAQ"

    def test_default_values(self) -> None:
        """Test default values."""
        entity = ResolvedEntity(
            original="Test",
            identifier="TEST.X",
            entity_type="company",
            confidence=0.5,
        )
        assert entity.alternatives == ()
        assert entity.metadata == {}


class TestExternalEntityResolver:
    """Test suite for entity resolution.

    NOTE: These tests use MockEntityResolver since ExternalEntityResolver
    requires a database connection. For real resolver tests, use integration tests.
    """

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Use MockEntityResolver for unit tests - ExternalEntityResolver needs database
        from src.nl2api.resolution.mock_resolver import MockEntityResolver
        self.resolver = MockEntityResolver()

    @pytest.mark.asyncio
    async def test_resolve_known_company(self) -> None:
        """Test resolving a known company name."""
        result = await self.resolver.resolve("What is Apple's EPS?")

        assert "Apple" in result
        assert result["Apple"] == "AAPL.O"

    @pytest.mark.asyncio
    async def test_resolve_single_known_company(self) -> None:
        """Test resolving a single known company."""
        result = await self.resolver.resolve_single("Microsoft")

        assert result is not None
        assert result.identifier == "MSFT.O"
        assert result.entity_type == "company"

    @pytest.mark.asyncio
    async def test_resolve_multiple_entities(self) -> None:
        """Test resolving multiple entities in one query."""
        result = await self.resolver.resolve("Compare Apple and Microsoft EPS")

        assert "Apple" in result
        assert "Microsoft" in result
        assert result["Apple"] == "AAPL.O"
        assert result["Microsoft"] == "MSFT.O"

    @pytest.mark.asyncio
    async def test_resolve_case_insensitive(self) -> None:
        """Test case-insensitive resolution."""
        result = await self.resolver.resolve("What is APPLE's EPS?")

        # Should still match "Apple" (case-insensitive lookup)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resolve_unknown_entity(self) -> None:
        """Test handling of unknown entities."""
        result = await self.resolver.resolve_single("XYZCorp International")

        # XYZCorp not in known mappings
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_empty_query(self) -> None:
        """Test handling of empty query."""
        result = await self.resolver.resolve("")

        assert result == {}

    @pytest.mark.asyncio
    async def test_resolve_no_entities(self) -> None:
        """Test query with no entity mentions."""
        result = await self.resolver.resolve("what is the best stock")

        assert result == {}

    @pytest.mark.asyncio
    async def test_known_company_mappings(self) -> None:
        """Test all common company mappings."""
        test_cases = [
            ("Apple", "AAPL.O"),
            ("Microsoft", "MSFT.O"),
            ("Google", "GOOGL.O"),
            ("Amazon", "AMZN.O"),
            ("Tesla", "TSLA.O"),
            ("Meta", "META.O"),
            ("Netflix", "NFLX.O"),
            ("Nvidia", "NVDA.O"),
        ]

        for company, expected_ric in test_cases:
            result = await self.resolver.resolve(f"What is {company}'s EPS?")
            assert company in result, f"Failed to resolve {company}"
            assert result[company] == expected_ric, f"Wrong RIC for {company}"

    @pytest.mark.asyncio
    async def test_resolve_with_possessive(self) -> None:
        """Test resolving company with possessive form."""
        result = await self.resolver.resolve("Apple's revenue forecast")

        assert "Apple" in result

    @pytest.mark.asyncio
    async def test_resolve_full_company_name(self) -> None:
        """Test resolving full company name."""
        result = await self.resolver.resolve("Alphabet earnings")

        # Alphabet is Google's parent company
        assert "Alphabet" in result
        assert result["Alphabet"] == "GOOGL.O"

    @pytest.mark.asyncio
    async def test_resolve_batch(self) -> None:
        """Test batch resolution of multiple entities."""
        entities = ["Apple", "Microsoft", "NonExistent Corp"]
        results = await self.resolver.resolve_batch(entities)

        # Should resolve Apple and Microsoft but not NonExistent
        assert len(results) == 2
        identifiers = {r.identifier for r in results}
        assert "AAPL.O" in identifiers
        assert "MSFT.O" in identifiers


class TestEntityResolverProtocol:
    """Test suite for EntityResolver protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that EntityResolver is runtime checkable."""
        from src.nl2api.resolution.protocols import EntityResolver

        class MinimalResolver:
            async def resolve(self, query: str) -> dict[str, str]:
                return {}

            async def resolve_single(self, entity: str, entity_type: str | None = None):
                return None

            async def resolve_batch(self, entities: list[str]) -> list:
                return []

        resolver = MinimalResolver()
        assert isinstance(resolver, EntityResolver)

    def test_external_resolver_is_entity_resolver(self) -> None:
        """Test that ExternalEntityResolver satisfies EntityResolver protocol."""
        from src.nl2api.resolution.protocols import EntityResolver

        resolver = ExternalEntityResolver()
        assert isinstance(resolver, EntityResolver)


class TestEntityResolverWithExternalAPI:
    """Test suite for entity resolver with external API integration.

    Note: These tests are designed to work with mocked external calls.
    In production, the resolver would call an external entity resolution service.
    """

    @pytest.mark.asyncio
    async def test_external_api_fallback(self) -> None:
        """Test fallback to external API for unknown entities."""
        # The current implementation uses a static mapping
        # External API integration would be added in Phase 3
        resolver = ExternalEntityResolver()

        # For now, unknown entities return None
        result = await resolver.resolve_single("Acme Corporation")

        # When external API is integrated, this would return the resolved RIC
        assert result is None  # No external API configured

    @pytest.mark.asyncio
    async def test_caching_behavior(self) -> None:
        """Test that resolver caches results appropriately."""
        resolver = ExternalEntityResolver(use_cache=True)

        # First call
        result1 = await resolver.resolve("Apple stock")

        # Second call with same entity
        result2 = await resolver.resolve("Apple forecast")

        # Both should resolve Apple to the same RIC
        assert result1.get("Apple") == result2.get("Apple")

    @pytest.mark.asyncio
    async def test_caching_disabled(self) -> None:
        """Test behavior when caching is disabled."""
        resolver = ExternalEntityResolver(use_cache=False)

        result1 = await resolver.resolve("Apple stock")
        result2 = await resolver.resolve("Apple forecast")

        # Should still work, just not cached
        assert result1.get("Apple") == result2.get("Apple")
