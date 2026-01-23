"""Unit tests for PostgresEntityRepository."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.storage.postgres.entity_repo import (
    Entity,
    EntityAlias,
    EntityMatch,
    EntityStats,
    PostgresEntityRepository,
)


class TestEntity:
    """Test suite for Entity dataclass."""

    def test_create_minimal_entity(self):
        """Test creating entity with minimal fields."""
        entity = Entity(
            id=str(uuid.uuid4()),
            primary_name="Apple Inc.",
            data_source="manual",
        )
        assert entity.primary_name == "Apple Inc."
        assert entity.data_source == "manual"
        assert entity.entity_type == "company"
        assert entity.entity_status == "active"
        assert entity.is_public is False

    def test_create_full_entity(self):
        """Test creating entity with all fields."""
        entity_id = str(uuid.uuid4())
        entity = Entity(
            id=entity_id,
            primary_name="Apple Inc.",
            data_source="sec_edgar",
            lei="HWUPKR0MPOU8FGXBT394",
            cik="0000320193",
            ticker="AAPL",
            ric="AAPL.O",
            exchange="NASDAQ",
            entity_type="company",
            entity_status="active",
            is_public=True,
            country_code="US",
            sic_code="3571",
            confidence_score=0.99,
            ric_validated=True,
        )
        assert entity.id == entity_id
        assert entity.lei == "HWUPKR0MPOU8FGXBT394"
        assert entity.ticker == "AAPL"
        assert entity.ric == "AAPL.O"
        assert entity.is_public is True
        assert entity.ric_validated is True


class TestEntityAlias:
    """Test suite for EntityAlias dataclass."""

    def test_create_alias(self):
        """Test creating entity alias."""
        alias = EntityAlias(
            id=str(uuid.uuid4()),
            entity_id=str(uuid.uuid4()),
            alias="apple",
            alias_type="trade_name",
            is_primary=False,
        )
        assert alias.alias == "apple"
        assert alias.alias_type == "trade_name"
        assert alias.is_primary is False


class TestEntityMatch:
    """Test suite for EntityMatch dataclass."""

    def test_create_exact_match(self):
        """Test creating exact match result."""
        match = EntityMatch(
            entity_id=str(uuid.uuid4()),
            primary_name="Apple Inc.",
            display_name="Apple",
            ric="AAPL.O",
            ticker="AAPL",
            exchange="NASDAQ",
            match_type="exact",
            similarity=1.0,
        )
        assert match.match_type == "exact"
        assert match.similarity == 1.0
        assert match.ric == "AAPL.O"

    def test_create_fuzzy_match(self):
        """Test creating fuzzy match result."""
        match = EntityMatch(
            entity_id=str(uuid.uuid4()),
            primary_name="Apple Inc.",
            display_name=None,
            ric="AAPL.O",
            ticker="AAPL",
            exchange=None,
            match_type="fuzzy",
            similarity=0.85,
        )
        assert match.match_type == "fuzzy"
        assert match.similarity == 0.85


class TestEntityStats:
    """Test suite for EntityStats dataclass."""

    def test_create_stats(self):
        """Test creating entity statistics."""
        stats = EntityStats(
            total_entities=1000000,
            public_entities=50000,
            private_entities=950000,
            entities_with_ric=45000,
            entities_with_validated_ric=40000,
            countries=195,
            exchanges=60,
            data_sources=3,
        )
        assert stats.total_entities == 1000000
        assert stats.public_entities == 50000
        assert stats.entities_with_ric == 45000


class TestPostgresEntityRepository:
    """Test suite for PostgresEntityRepository."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        pool.fetchrow = AsyncMock()
        pool.fetch = AsyncMock()
        pool.fetchval = AsyncMock()
        pool.execute = AsyncMock()

        # Mock acquire context manager
        conn = MagicMock()
        conn.executemany = AsyncMock()
        conn.copy_records_to_table = AsyncMock(return_value="COPY 100")
        conn.transaction = MagicMock(return_value=AsyncMock())
        conn.transaction.return_value.__aenter__ = AsyncMock()
        conn.transaction.return_value.__aexit__ = AsyncMock()

        pool.acquire = MagicMock(return_value=AsyncMock())
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock()

        return pool

    @pytest.fixture
    def sample_entity_row(self):
        """Create sample entity database row."""
        return {
            "id": uuid.uuid4(),
            "lei": "HWUPKR0MPOU8FGXBT394",
            "cik": "0000320193",
            "permid": None,
            "figi": "BBG000B9XRY4",
            "primary_name": "Apple Inc.",
            "display_name": "Apple",
            "ticker": "AAPL",
            "ric": "AAPL.O",
            "exchange": "NASDAQ",
            "entity_type": "company",
            "entity_status": "active",
            "is_public": True,
            "country_code": "US",
            "region": "California",
            "city": "Cupertino",
            "sic_code": "3571",
            "naics_code": "334220",
            "gics_sector": "Information Technology",
            "parent_entity_id": None,
            "ultimate_parent_id": None,
            "data_source": "sec_edgar",
            "confidence_score": 0.99,
            "ric_validated": True,
            "last_verified_at": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }

    @pytest.mark.asyncio
    async def test_get_entity_by_id(self, mock_pool, sample_entity_row):
        """Test fetching entity by ID."""
        mock_pool.fetchrow.return_value = sample_entity_row

        repo = PostgresEntityRepository(mock_pool)
        entity = await repo.get(str(sample_entity_row["id"]))

        assert entity is not None
        assert entity.primary_name == "Apple Inc."
        assert entity.ticker == "AAPL"
        assert entity.ric == "AAPL.O"
        mock_pool.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, mock_pool):
        """Test fetching non-existent entity."""
        mock_pool.fetchrow.return_value = None

        repo = PostgresEntityRepository(mock_pool)
        entity = await repo.get(str(uuid.uuid4()))

        assert entity is None

    @pytest.mark.asyncio
    async def test_get_entity_invalid_id(self, mock_pool):
        """Test fetching entity with invalid ID."""
        repo = PostgresEntityRepository(mock_pool)
        entity = await repo.get("invalid-uuid")

        assert entity is None
        mock_pool.fetchrow.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_lei(self, mock_pool, sample_entity_row):
        """Test fetching entity by LEI."""
        mock_pool.fetchrow.return_value = sample_entity_row

        repo = PostgresEntityRepository(mock_pool)
        entity = await repo.get_by_lei("HWUPKR0MPOU8FGXBT394")

        assert entity is not None
        assert entity.lei == "HWUPKR0MPOU8FGXBT394"

    @pytest.mark.asyncio
    async def test_get_by_cik_normalized(self, mock_pool, sample_entity_row):
        """Test CIK is normalized to 10 digits."""
        mock_pool.fetchrow.return_value = sample_entity_row

        repo = PostgresEntityRepository(mock_pool)
        await repo.get_by_cik("320193")

        # Check that CIK was normalized to 10 digits
        call_args = mock_pool.fetchrow.call_args
        assert "0000320193" in call_args[0]

    @pytest.mark.asyncio
    async def test_get_by_ticker(self, mock_pool, sample_entity_row):
        """Test fetching entity by ticker."""
        mock_pool.fetchrow.return_value = sample_entity_row

        repo = PostgresEntityRepository(mock_pool)
        entity = await repo.get_by_ticker("aapl")  # lowercase

        assert entity is not None
        assert entity.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_get_by_ric(self, mock_pool, sample_entity_row):
        """Test fetching entity by RIC."""
        mock_pool.fetchrow.return_value = sample_entity_row

        repo = PostgresEntityRepository(mock_pool)
        entity = await repo.get_by_ric("AAPL.O")

        assert entity is not None
        assert entity.ric == "AAPL.O"

    @pytest.mark.asyncio
    async def test_save_entity(self, mock_pool):
        """Test saving entity."""
        mock_pool.execute.return_value = "INSERT 0 1"

        repo = PostgresEntityRepository(mock_pool)
        entity = Entity(
            id=str(uuid.uuid4()),
            primary_name="Test Company",
            data_source="manual",
            ticker="TEST",
            ric="TEST.O",
        )

        entity_id = await repo.save(entity)

        assert entity_id == entity.id
        mock_pool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_batch_entities(self, mock_pool):
        """Test saving multiple entities in batch."""
        repo = PostgresEntityRepository(mock_pool)

        entities = [
            Entity(id=str(uuid.uuid4()), primary_name=f"Company {i}", data_source="test")
            for i in range(10)
        ]

        count = await repo.save_batch(entities)

        assert count == 10

    @pytest.mark.asyncio
    async def test_save_batch_empty(self, mock_pool):
        """Test saving empty batch."""
        repo = PostgresEntityRepository(mock_pool)
        count = await repo.save_batch([])

        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_entity(self, mock_pool):
        """Test deleting entity."""
        mock_pool.execute.return_value = "DELETE 1"

        repo = PostgresEntityRepository(mock_pool)
        result = await repo.delete(str(uuid.uuid4()))

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, mock_pool):
        """Test deleting non-existent entity."""
        mock_pool.execute.return_value = "DELETE 0"

        repo = PostgresEntityRepository(mock_pool)
        result = await repo.delete(str(uuid.uuid4()))

        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_exact_match(self, mock_pool):
        """Test resolving entity with exact match."""
        mock_pool.fetch.return_value = [
            {
                "entity_id": uuid.uuid4(),
                "primary_name": "Apple Inc.",
                "display_name": "Apple",
                "ric": "AAPL.O",
                "ticker": "AAPL",
                "exchange": "NASDAQ",
                "match_type": "exact",
                "similarity": 1.0,
            }
        ]

        repo = PostgresEntityRepository(mock_pool)
        matches = await repo.resolve("apple")

        assert len(matches) == 1
        assert matches[0].match_type == "exact"
        assert matches[0].similarity == 1.0
        assert matches[0].ric == "AAPL.O"

    @pytest.mark.asyncio
    async def test_resolve_fuzzy_match(self, mock_pool):
        """Test resolving entity with fuzzy match."""
        mock_pool.fetch.return_value = [
            {
                "entity_id": uuid.uuid4(),
                "primary_name": "Apple Inc.",
                "display_name": "Apple",
                "ric": "AAPL.O",
                "ticker": "AAPL",
                "exchange": "NASDAQ",
                "match_type": "fuzzy",
                "similarity": 0.85,
            }
        ]

        repo = PostgresEntityRepository(mock_pool)
        matches = await repo.resolve("aple", fuzzy_threshold=0.3)

        assert len(matches) == 1
        assert matches[0].match_type == "fuzzy"
        assert matches[0].similarity == 0.85

    @pytest.mark.asyncio
    async def test_resolve_no_match(self, mock_pool):
        """Test resolving entity with no match."""
        mock_pool.fetch.return_value = []

        repo = PostgresEntityRepository(mock_pool)
        matches = await repo.resolve("xyznonexistent")

        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_resolve_batch(self, mock_pool):
        """Test batch resolution."""
        mock_pool.fetch.return_value = [
            {
                "query": "apple",
                "entity_id": uuid.uuid4(),
                "primary_name": "Apple Inc.",
                "ric": "AAPL.O",
                "ticker": "AAPL",
                "match_type": "exact",
                "similarity": 1.0,
            },
            {
                "query": "microsoft",
                "entity_id": uuid.uuid4(),
                "primary_name": "Microsoft Corporation",
                "ric": "MSFT.O",
                "ticker": "MSFT",
                "match_type": "exact",
                "similarity": 1.0,
            },
        ]

        repo = PostgresEntityRepository(mock_pool)
        results = await repo.resolve_batch(["apple", "microsoft", "unknown"])

        assert len(results) == 3
        assert results["apple"] is not None
        assert results["apple"].ric == "AAPL.O"
        assert results["microsoft"] is not None
        assert results["microsoft"].ric == "MSFT.O"
        assert results["unknown"] is None

    @pytest.mark.asyncio
    async def test_resolve_batch_empty(self, mock_pool):
        """Test batch resolution with empty input."""
        repo = PostgresEntityRepository(mock_pool)
        results = await repo.resolve_batch([])

        assert results == {}

    @pytest.mark.asyncio
    async def test_add_alias(self, mock_pool):
        """Test adding entity alias."""
        mock_pool.execute.return_value = "INSERT 0 1"

        repo = PostgresEntityRepository(mock_pool)
        entity_id = str(uuid.uuid4())
        alias_id = await repo.add_alias(entity_id, "Apple", "trade_name")

        assert alias_id is not None

    @pytest.mark.asyncio
    async def test_add_alias_normalized(self, mock_pool):
        """Test alias is normalized to lowercase."""
        mock_pool.execute.return_value = "INSERT 0 1"

        repo = PostgresEntityRepository(mock_pool)
        entity_id = str(uuid.uuid4())
        await repo.add_alias(entity_id, "  APPLE  ", "trade_name")

        # Check that alias was normalized
        call_args = mock_pool.execute.call_args
        assert "apple" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_aliases(self, mock_pool):
        """Test getting entity aliases."""
        entity_id = uuid.uuid4()
        mock_pool.fetch.return_value = [
            {
                "id": uuid.uuid4(),
                "entity_id": entity_id,
                "alias": "apple",
                "alias_type": "trade_name",
                "is_primary": True,
                "created_at": datetime.now(UTC),
            },
            {
                "id": uuid.uuid4(),
                "entity_id": entity_id,
                "alias": "aapl",
                "alias_type": "ticker",
                "is_primary": False,
                "created_at": datetime.now(UTC),
            },
        ]

        repo = PostgresEntityRepository(mock_pool)
        aliases = await repo.get_aliases(str(entity_id))

        assert len(aliases) == 2
        assert aliases[0].alias == "apple"
        assert aliases[0].is_primary is True
        assert aliases[1].alias == "aapl"

    @pytest.mark.asyncio
    async def test_delete_aliases(self, mock_pool):
        """Test deleting entity aliases."""
        mock_pool.execute.return_value = "DELETE 5"

        repo = PostgresEntityRepository(mock_pool)
        count = await repo.delete_aliases(str(uuid.uuid4()))

        assert count == 5

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_pool):
        """Test getting entity statistics."""
        mock_pool.fetchrow.return_value = {
            "total_entities": 2000000,
            "public_entities": 50000,
            "private_entities": 1950000,
            "entities_with_ric": 45000,
            "entities_with_validated_ric": 40000,
            "countries": 195,
            "exchanges": 60,
            "data_sources": 3,
        }

        repo = PostgresEntityRepository(mock_pool)
        stats = await repo.get_stats()

        assert stats.total_entities == 2000000
        assert stats.public_entities == 50000
        assert stats.entities_with_ric == 45000
        assert stats.data_sources == 3

    @pytest.mark.asyncio
    async def test_get_coverage_by_source(self, mock_pool):
        """Test getting coverage breakdown by source."""
        mock_pool.fetch.return_value = [
            {
                "data_source": "gleif",
                "entity_count": 2000000,
                "with_ric": 50000,
                "ric_validated": 40000,
                "ric_coverage_pct": 2.5,
            },
            {
                "data_source": "sec_edgar",
                "entity_count": 8500,
                "with_ric": 8500,
                "ric_validated": 8500,
                "ric_coverage_pct": 100.0,
            },
        ]

        repo = PostgresEntityRepository(mock_pool)
        coverage = await repo.get_coverage_by_source()

        assert len(coverage) == 2
        assert coverage[0]["data_source"] == "gleif"
        assert coverage[0]["entity_count"] == 2000000
        assert coverage[1]["data_source"] == "sec_edgar"
        assert coverage[1]["ric_coverage_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_count_entities(self, mock_pool):
        """Test counting entities."""
        mock_pool.fetchval.return_value = 2000000

        repo = PostgresEntityRepository(mock_pool)
        count = await repo.count()

        assert count == 2000000

    @pytest.mark.asyncio
    async def test_count_entities_by_source(self, mock_pool):
        """Test counting entities by data source."""
        mock_pool.fetchval.return_value = 8500

        repo = PostgresEntityRepository(mock_pool)
        count = await repo.count(data_source="sec_edgar")

        assert count == 8500

    @pytest.mark.asyncio
    async def test_bulk_import(self, mock_pool):
        """Test bulk import using COPY protocol."""
        repo = PostgresEntityRepository(mock_pool)

        records = [
            (
                uuid.uuid4(),
                "LEI1",
                None,
                None,
                None,
                "Company 1",
                None,
                None,
                None,
                None,
                "company",
                "active",
                False,
                "US",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "gleif",
                1.0,
                False,
                None,
            ),
            (
                uuid.uuid4(),
                "LEI2",
                None,
                None,
                None,
                "Company 2",
                None,
                None,
                None,
                None,
                "company",
                "active",
                False,
                "UK",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "gleif",
                1.0,
                False,
                None,
            ),
        ]
        columns = [
            "id",
            "lei",
            "cik",
            "permid",
            "figi",
            "primary_name",
            "display_name",
            "ticker",
            "ric",
            "exchange",
            "entity_type",
            "entity_status",
            "is_public",
            "country_code",
            "region",
            "city",
            "sic_code",
            "naics_code",
            "gics_sector",
            "parent_entity_id",
            "ultimate_parent_id",
            "data_source",
            "confidence_score",
            "ric_validated",
            "last_verified_at",
        ]

        count = await repo.bulk_import(records, columns)

        assert count == 100  # From mock return value "COPY 100"

    @pytest.mark.asyncio
    async def test_search_entities(self, mock_pool, sample_entity_row):
        """Test searching entities."""
        mock_pool.fetch.return_value = [sample_entity_row]

        repo = PostgresEntityRepository(mock_pool)
        results = await repo.search("Apple", country_code="US", is_public=True)

        assert len(results) == 1
        assert results[0].primary_name == "Apple Inc."
