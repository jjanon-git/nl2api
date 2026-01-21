"""Tests for Mock Entity Resolver."""

from __future__ import annotations

import pytest

from src.nl2api.resolution import MockEntityResolver, ResolvedEntity


class TestMockEntityResolverSingleResolution:
    """Test suite for single entity resolution."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.resolver = MockEntityResolver()

    @pytest.mark.asyncio
    async def test_resolve_apple(self) -> None:
        """Test resolving Apple to AAPL.O."""
        result = await self.resolver.resolve_single("Apple")
        assert result is not None
        assert result.identifier == "AAPL.O"
        assert result.entity_type == "company"
        assert result.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_resolve_apple_inc(self) -> None:
        """Test resolving Apple Inc. to AAPL.O."""
        result = await self.resolver.resolve_single("Apple Inc.")
        assert result is not None
        assert result.identifier == "AAPL.O"

    @pytest.mark.asyncio
    async def test_resolve_microsoft(self) -> None:
        """Test resolving Microsoft to MSFT.O."""
        result = await self.resolver.resolve_single("Microsoft")
        assert result is not None
        assert result.identifier == "MSFT.O"

    @pytest.mark.asyncio
    async def test_resolve_google(self) -> None:
        """Test resolving Google to GOOGL.O."""
        result = await self.resolver.resolve_single("Google")
        assert result is not None
        assert result.identifier == "GOOGL.O"

    @pytest.mark.asyncio
    async def test_resolve_alphabet(self) -> None:
        """Test resolving Alphabet to GOOGL.O."""
        result = await self.resolver.resolve_single("Alphabet")
        assert result is not None
        assert result.identifier == "GOOGL.O"

    @pytest.mark.asyncio
    async def test_resolve_amazon(self) -> None:
        """Test resolving Amazon to AMZN.O."""
        result = await self.resolver.resolve_single("Amazon")
        assert result is not None
        assert result.identifier == "AMZN.O"

    @pytest.mark.asyncio
    async def test_resolve_tesla(self) -> None:
        """Test resolving Tesla to TSLA.O."""
        result = await self.resolver.resolve_single("Tesla")
        assert result is not None
        assert result.identifier == "TSLA.O"

    @pytest.mark.asyncio
    async def test_resolve_nvidia(self) -> None:
        """Test resolving NVIDIA to NVDA.O."""
        result = await self.resolver.resolve_single("NVIDIA")
        assert result is not None
        assert result.identifier == "NVDA.O"

    @pytest.mark.asyncio
    async def test_resolve_jpmorgan(self) -> None:
        """Test resolving JP Morgan to JPM.N."""
        result = await self.resolver.resolve_single("JP Morgan")
        assert result is not None
        assert result.identifier == "JPM.N"

    @pytest.mark.asyncio
    async def test_resolve_johnson_and_johnson(self) -> None:
        """Test resolving Johnson & Johnson to JNJ.N."""
        result = await self.resolver.resolve_single("Johnson & Johnson")
        assert result is not None
        assert result.identifier == "JNJ.N"

    @pytest.mark.asyncio
    async def test_resolve_j_and_j(self) -> None:
        """Test resolving J&J to JNJ.N."""
        result = await self.resolver.resolve_single("J&J")
        assert result is not None
        assert result.identifier == "JNJ.N"

    @pytest.mark.asyncio
    async def test_resolve_unknown_company(self) -> None:
        """Test that unknown companies return None."""
        result = await self.resolver.resolve_single("Unknown Company XYZ")
        assert result is None

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        """Test that resolution is case-insensitive."""
        result1 = await self.resolver.resolve_single("apple")
        result2 = await self.resolver.resolve_single("APPLE")
        result3 = await self.resolver.resolve_single("Apple")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1.identifier == result2.identifier == result3.identifier == "AAPL.O"


class TestMockEntityResolverQueryResolution:
    """Test suite for resolving entities from queries."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.resolver = MockEntityResolver()

    @pytest.mark.asyncio
    async def test_resolve_single_company_in_query(self) -> None:
        """Test resolving a single company from a query."""
        result = await self.resolver.resolve("What is Apple's revenue?")
        assert "Apple" in result or "apple" in [k.lower() for k in result.keys()]
        # Get the resolved RIC
        ric = list(result.values())[0] if result else None
        assert ric == "AAPL.O"

    @pytest.mark.asyncio
    async def test_resolve_multiple_companies_in_query(self) -> None:
        """Test resolving multiple companies from a query."""
        result = await self.resolver.resolve("Compare Apple and Microsoft revenue")
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_resolve_with_possessive(self) -> None:
        """Test resolving company names with possessive forms."""
        result = await self.resolver.resolve("What is Tesla's stock price?")
        # Should find Tesla
        values = list(result.values())
        assert "TSLA.O" in values

    @pytest.mark.asyncio
    async def test_resolve_no_companies(self) -> None:
        """Test query with no company names."""
        result = await self.resolver.resolve("What is the current stock price?")
        # May return empty or partial results depending on patterns
        # The key is it doesn't raise an error
        assert isinstance(result, dict)


class TestMockEntityResolverBatchResolution:
    """Test suite for batch entity resolution."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.resolver = MockEntityResolver()

    @pytest.mark.asyncio
    async def test_resolve_batch_all_known(self) -> None:
        """Test batch resolution with all known entities."""
        entities = ["Apple", "Microsoft", "Google", "Amazon"]
        results = await self.resolver.resolve_batch(entities)

        assert len(results) == 4
        rics = {r.identifier for r in results}
        assert "AAPL.O" in rics
        assert "MSFT.O" in rics
        assert "GOOGL.O" in rics
        assert "AMZN.O" in rics

    @pytest.mark.asyncio
    async def test_resolve_batch_partial(self) -> None:
        """Test batch resolution with some unknown entities."""
        entities = ["Apple", "Unknown Corp", "Microsoft"]
        results = await self.resolver.resolve_batch(entities)

        # Should return 2 (Apple and Microsoft)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_resolve_batch_empty(self) -> None:
        """Test batch resolution with empty list."""
        results = await self.resolver.resolve_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_resolve_batch_all_unknown(self) -> None:
        """Test batch resolution with all unknown entities."""
        entities = ["Unknown1", "Unknown2", "Unknown3"]
        results = await self.resolver.resolve_batch(entities)
        assert results == []


class TestMockEntityResolverCaching:
    """Test suite for caching behavior."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.resolver = MockEntityResolver()

    @pytest.mark.asyncio
    async def test_cache_hit(self) -> None:
        """Test that resolved entities are cached."""
        # First resolution
        result1 = await self.resolver.resolve_single("Apple")
        # Second resolution (should hit cache)
        result2 = await self.resolver.resolve_single("apple")

        assert result1 is not None
        assert result2 is not None
        # Should be the same cached object
        assert result1.identifier == result2.identifier

    @pytest.mark.asyncio
    async def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        # Resolve and cache
        await self.resolver.resolve_single("Apple")
        assert len(self.resolver._cache) > 0

        # Clear cache
        self.resolver.clear_cache()
        assert len(self.resolver._cache) == 0


class TestMockEntityResolverCustomMappings:
    """Test suite for custom mappings."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.resolver = MockEntityResolver()

    @pytest.mark.asyncio
    async def test_add_custom_mapping(self) -> None:
        """Test adding a custom company mapping."""
        # Add custom mapping
        self.resolver.add_mapping("My Company", "MYCO.O")

        # Resolve
        result = await self.resolver.resolve_single("My Company")
        assert result is not None
        assert result.identifier == "MYCO.O"

    @pytest.mark.asyncio
    async def test_initial_custom_mappings(self) -> None:
        """Test initializing with custom mappings."""
        custom = {"custom corp": "CUST.O", "another inc": "ANO.N"}
        resolver = MockEntityResolver(additional_mappings=custom)

        result1 = await resolver.resolve_single("Custom Corp")
        result2 = await resolver.resolve_single("Another Inc")

        assert result1 is not None
        assert result1.identifier == "CUST.O"
        assert result2 is not None
        assert result2.identifier == "ANO.N"


class TestMockEntityResolverComprehensiveMappings:
    """Test suite for comprehensive company mappings."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.resolver = MockEntityResolver()

    @pytest.mark.asyncio
    async def test_big_tech_companies(self) -> None:
        """Test all big tech companies are mapped."""
        companies = {
            "Apple": "AAPL.O",
            "Microsoft": "MSFT.O",
            "Google": "GOOGL.O",
            "Amazon": "AMZN.O",
            "Meta": "META.O",
            "Tesla": "TSLA.O",
            "NVIDIA": "NVDA.O",
            "Netflix": "NFLX.O",
        }

        for company, expected_ric in companies.items():
            result = await self.resolver.resolve_single(company)
            assert result is not None, f"Failed to resolve {company}"
            assert result.identifier == expected_ric, f"Wrong RIC for {company}"

    @pytest.mark.asyncio
    async def test_financial_companies(self) -> None:
        """Test financial companies are mapped."""
        companies = {
            "JPMorgan": "JPM.N",
            "Goldman Sachs": "GS.N",
            "Bank of America": "BAC.N",
            "Visa": "V.N",
            "Mastercard": "MA.N",
        }

        for company, expected_ric in companies.items():
            result = await self.resolver.resolve_single(company)
            assert result is not None, f"Failed to resolve {company}"
            assert result.identifier == expected_ric, f"Wrong RIC for {company}"

    @pytest.mark.asyncio
    async def test_healthcare_companies(self) -> None:
        """Test healthcare companies are mapped."""
        companies = {
            "Pfizer": "PFE.N",
            "Merck": "MRK.N",
            "Eli Lilly": "LLY.N",
        }

        for company, expected_ric in companies.items():
            result = await self.resolver.resolve_single(company)
            assert result is not None, f"Failed to resolve {company}"
            assert result.identifier == expected_ric, f"Wrong RIC for {company}"

    @pytest.mark.asyncio
    async def test_uk_companies(self) -> None:
        """Test UK companies are mapped with .L suffix."""
        companies = {
            "Barclays": "BARC.L",
            "HSBC": "HSBA.L",
            "Vodafone": "VOD.L",
        }

        for company, expected_ric in companies.items():
            result = await self.resolver.resolve_single(company)
            assert result is not None, f"Failed to resolve {company}"
            assert result.identifier == expected_ric, f"Wrong RIC for {company}"


class TestMockEntityResolverProtocolCompliance:
    """Test suite for protocol compliance."""

    @pytest.mark.asyncio
    async def test_implements_resolve(self) -> None:
        """Test that resolve method is implemented."""
        resolver = MockEntityResolver()
        result = await resolver.resolve("Test query")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_implements_resolve_single(self) -> None:
        """Test that resolve_single method is implemented."""
        resolver = MockEntityResolver()
        result = await resolver.resolve_single("Apple")
        assert result is None or isinstance(result, ResolvedEntity)

    @pytest.mark.asyncio
    async def test_implements_resolve_batch(self) -> None:
        """Test that resolve_batch method is implemented."""
        resolver = MockEntityResolver()
        result = await resolver.resolve_batch(["Apple", "Microsoft"])
        assert isinstance(result, list)
        assert all(isinstance(r, ResolvedEntity) for r in result)
