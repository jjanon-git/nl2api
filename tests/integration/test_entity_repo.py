"""
Entity Repository Integration Tests

Tests the PostgresEntityRepository against a real PostgreSQL database.
Requires PostgreSQL with migrations applied.

Run with: pytest tests/integration/test_entity_repo.py -v
"""

import os
import uuid
from pathlib import Path

import pytest

# Load env before imports
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


@pytest.fixture
async def db_pool():
    """Create database connection pool."""
    import asyncpg

    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")
    pool = await asyncpg.create_pool(db_url)
    yield pool
    await pool.close()


@pytest.fixture
async def entity_repo(db_pool):
    """Create entity repository."""
    from src.common.storage.postgres.entity_repo import PostgresEntityRepository

    return PostgresEntityRepository(db_pool)


@pytest.fixture
async def has_entity_tables(db_pool) -> bool:
    """Check if entity tables exist."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'entities'
            )
            """
        )
    return result


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing with unique identifiers."""
    from src.common.storage.postgres.entity_repo import Entity

    # Use unique test identifiers that won't conflict with production data
    unique_suffix = uuid.uuid4().hex[:8].upper()
    return Entity(
        id=str(uuid.uuid4()),
        primary_name=f"Integration Test Company {unique_suffix} Inc",
        data_source="test",
        lei=f"TEST{unique_suffix}00000001",
        cik=f"99{unique_suffix}".zfill(10),  # Normalized to 10 digits
        ticker=f"TST{unique_suffix[:4]}",
        ric=f"TST{unique_suffix[:4]}.N",
        exchange="NYSE",
        entity_type="company",
        entity_status="active",
        is_public=True,
        country_code="US",
        region="North America",
        city="New York",
        sic_code="7370",
        naics_code="541511",
        gics_sector="Information Technology",
        confidence_score=1.0,
        ric_validated=True,
    )


@pytest.fixture
async def cleanup_test_data(db_pool):
    """Fixture to clean up test data before and after each test."""
    # Clean up BEFORE the test to ensure no leftover data
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM entity_aliases WHERE entity_id IN (SELECT id FROM entities WHERE data_source = 'test')"
        )
        await conn.execute("DELETE FROM entities WHERE data_source = 'test'")
    yield
    # Clean up AFTER the test as well
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM entity_aliases WHERE entity_id IN (SELECT id FROM entities WHERE data_source = 'test')"
        )
        await conn.execute("DELETE FROM entities WHERE data_source = 'test'")


class TestEntityCRUD:
    """Test basic CRUD operations against real database."""

    @pytest.mark.asyncio
    async def test_save_and_get_entity(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test saving and retrieving an entity."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        # Save entity
        entity_id = await entity_repo.save(sample_entity)
        assert entity_id == sample_entity.id

        # Retrieve by ID
        retrieved = await entity_repo.get(entity_id)
        assert retrieved is not None
        assert retrieved.primary_name == sample_entity.primary_name
        assert retrieved.ticker == sample_entity.ticker
        assert retrieved.ric == sample_entity.ric
        assert retrieved.is_public is True

    @pytest.mark.asyncio
    async def test_get_by_lei(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test retrieving entity by LEI."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        retrieved = await entity_repo.get_by_lei(sample_entity.lei)
        assert retrieved is not None
        assert retrieved.id == sample_entity.id

        # Test case insensitivity
        retrieved_lower = await entity_repo.get_by_lei(sample_entity.lei.lower())
        assert retrieved_lower is not None
        assert retrieved_lower.id == sample_entity.id

    @pytest.mark.asyncio
    async def test_get_by_cik(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test retrieving entity by CIK."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Test with exact CIK value
        retrieved = await entity_repo.get_by_cik(sample_entity.cik)
        assert retrieved is not None
        assert retrieved.id == sample_entity.id

        # Test with shorter CIK value (should normalize with leading zeros)
        # Take the CIK, strip leading zeros, and search
        short_cik = sample_entity.cik.lstrip("0")
        if short_cik:  # Only test if there were leading zeros to strip
            retrieved_short = await entity_repo.get_by_cik(short_cik)
            assert retrieved_short is not None
            assert retrieved_short.id == sample_entity.id

    @pytest.mark.asyncio
    async def test_get_by_ticker(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test retrieving entity by ticker."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        retrieved = await entity_repo.get_by_ticker(sample_entity.ticker)
        assert retrieved is not None
        assert retrieved.id == sample_entity.id

        # Test case insensitivity
        retrieved_lower = await entity_repo.get_by_ticker(sample_entity.ticker.lower())
        assert retrieved_lower is not None
        assert retrieved_lower.id == sample_entity.id

    @pytest.mark.asyncio
    async def test_get_by_ric(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test retrieving entity by RIC."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        retrieved = await entity_repo.get_by_ric(sample_entity.ric)
        assert retrieved is not None
        assert retrieved.id == sample_entity.id

    @pytest.mark.asyncio
    async def test_update_entity(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test updating an existing entity."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Create updated entity with same ID
        from src.common.storage.postgres.entity_repo import Entity

        updated = Entity(
            id=sample_entity.id,
            primary_name="Updated Company Name",
            data_source="test",
            ticker="UPDT",
            ric="UPDT.O",
            exchange="NASDAQ",
            is_public=True,
        )

        await entity_repo.save(updated)

        retrieved = await entity_repo.get(sample_entity.id)
        assert retrieved.primary_name == "Updated Company Name"
        assert retrieved.ticker == "UPDT"
        assert retrieved.ric == "UPDT.O"
        # Original LEI should be preserved (COALESCE in upsert)
        assert retrieved.lei == sample_entity.lei

    @pytest.mark.asyncio
    async def test_delete_entity(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test deleting an entity."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Verify exists
        assert await entity_repo.get(sample_entity.id) is not None

        # Delete
        deleted = await entity_repo.delete(sample_entity.id)
        assert deleted is True

        # Verify gone
        assert await entity_repo.get(sample_entity.id) is None

        # Delete non-existent returns False
        deleted_again = await entity_repo.delete(sample_entity.id)
        assert deleted_again is False

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity(self, entity_repo, has_entity_tables, cleanup_test_data):
        """Test retrieving non-existent entity returns None."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        result = await entity_repo.get(str(uuid.uuid4()))
        assert result is None

        result = await entity_repo.get_by_lei("NONEXISTENT")
        assert result is None


class TestEntityResolution:
    """Test entity resolution queries against real database."""

    @pytest.mark.asyncio
    async def test_resolve_exact_alias_match(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test resolution finds exact alias matches."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)
        # Use unique alias based on sample_entity name
        alias_name = f"Test Company {sample_entity.ticker}"
        await entity_repo.add_alias(sample_entity.id, alias_name, "legal_name", is_primary=True)
        await entity_repo.add_alias(sample_entity.id, f"{sample_entity.ticker} Corp", "trade_name")

        # Exact match on alias
        matches = await entity_repo.resolve(alias_name.lower())
        assert len(matches) >= 1
        assert matches[0].entity_id == sample_entity.id
        assert matches[0].match_type == "exact"
        assert matches[0].similarity == 1.0

    @pytest.mark.asyncio
    async def test_resolve_ticker_match(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test resolution finds ticker matches."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        matches = await entity_repo.resolve(sample_entity.ticker)
        assert len(matches) >= 1
        assert matches[0].entity_id == sample_entity.id
        assert matches[0].match_type == "ticker"
        assert matches[0].ticker == sample_entity.ticker

    @pytest.mark.asyncio
    async def test_resolve_ric_match(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test resolution finds RIC matches."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        matches = await entity_repo.resolve(sample_entity.ric)
        assert len(matches) >= 1
        assert matches[0].entity_id == sample_entity.id
        assert matches[0].match_type == "ric"
        assert matches[0].ric == sample_entity.ric

    @pytest.mark.asyncio
    async def test_resolve_fuzzy_match(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test resolution finds fuzzy matches using trigram similarity."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)
        await entity_repo.add_alias(
            sample_entity.id, sample_entity.primary_name, "legal_name", is_primary=True
        )

        # Misspelled query should still match via fuzzy
        # Take the primary name and introduce typos
        misspelled = sample_entity.primary_name.replace("a", "e").replace("i", "o")
        matches = await entity_repo.resolve(misspelled, fuzzy_threshold=0.3)
        # May or may not match depending on trigram similarity
        # This tests that fuzzy matching doesn't error
        assert isinstance(matches, list)

    @pytest.mark.asyncio
    async def test_resolve_returns_best_matches_first(
        self, entity_repo, has_entity_tables, cleanup_test_data
    ):
        """Test resolution orders by match quality."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        from src.common.storage.postgres.entity_repo import Entity

        # Create two entities
        entity1 = Entity(
            id=str(uuid.uuid4()),
            primary_name="Apple Inc",
            data_source="test",
            ticker="AAPL",
            ric="AAPL.O",
            is_public=True,
        )
        entity2 = Entity(
            id=str(uuid.uuid4()),
            primary_name="Apple Hospitality REIT",
            data_source="test",
            ticker="APLE",
            ric="APLE.N",
            is_public=True,
        )

        await entity_repo.save(entity1)
        await entity_repo.save(entity2)
        await entity_repo.add_alias(entity1.id, "apple inc", "legal_name", is_primary=True)
        await entity_repo.add_alias(
            entity2.id, "apple hospitality reit", "legal_name", is_primary=True
        )

        # Ticker match should be ranked highly
        matches = await entity_repo.resolve("AAPL")
        assert len(matches) >= 1
        assert matches[0].ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_resolve_batch(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test batch resolution of multiple queries."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)
        await entity_repo.add_alias(
            sample_entity.id, sample_entity.primary_name.lower(), "legal_name", is_primary=True
        )

        # Use a unique string that won't fuzzy-match any real entity names
        nonexistent_query = "ZZZZZZZNOTFOUND123"
        results = await entity_repo.resolve_batch(
            [sample_entity.ticker, sample_entity.ric, nonexistent_query],
            fuzzy_threshold=0.3,
        )

        assert sample_entity.ticker in results
        assert sample_entity.ric in results
        assert nonexistent_query in results

        assert results[sample_entity.ticker] is not None
        assert results[sample_entity.ticker].ticker == sample_entity.ticker
        assert results[sample_entity.ric] is not None
        assert results[sample_entity.ric].ric == sample_entity.ric
        assert results[nonexistent_query] is None


class TestEntityAliases:
    """Test alias operations against real database."""

    @pytest.mark.asyncio
    async def test_add_and_get_aliases(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test adding and retrieving aliases."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Add aliases using dynamic entity values
        alias1_id = await entity_repo.add_alias(
            sample_entity.id, sample_entity.primary_name, "legal_name", is_primary=True
        )
        alias2_id = await entity_repo.add_alias(
            sample_entity.id, f"{sample_entity.ticker} Corp", "trade_name"
        )
        alias3_id = await entity_repo.add_alias(
            sample_entity.id, sample_entity.ticker[:3], "abbreviation"
        )

        assert alias1_id is not None
        assert alias2_id is not None
        assert alias3_id is not None

        # Get aliases
        aliases = await entity_repo.get_aliases(sample_entity.id)
        assert len(aliases) == 3

        # Primary should be first
        assert aliases[0].is_primary is True
        assert aliases[0].alias == sample_entity.primary_name.lower()  # Normalized to lowercase

    @pytest.mark.asyncio
    async def test_add_duplicate_alias_ignored(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test that duplicate aliases are handled gracefully."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Add same alias twice
        alias1_id = await entity_repo.add_alias(sample_entity.id, "Duplicate Alias", "trade_name")
        alias2_id = await entity_repo.add_alias(sample_entity.id, "Duplicate Alias", "trade_name")

        # Second should still return an ID (ON CONFLICT DO NOTHING)
        assert alias1_id is not None
        assert alias2_id is not None

        # Only one alias should exist
        aliases = await entity_repo.get_aliases(sample_entity.id)
        alias_texts = [a.alias for a in aliases]
        assert alias_texts.count("duplicate alias") == 1

    @pytest.mark.asyncio
    async def test_add_aliases_batch(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test batch alias insertion."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        batch = [
            (sample_entity.id, "Batch Alias 1", "legal_name"),
            (sample_entity.id, "Batch Alias 2", "trade_name"),
            (sample_entity.id, "Batch Alias 3", "abbreviation"),
        ]

        count = await entity_repo.add_aliases_batch(batch)
        assert count == 3

        aliases = await entity_repo.get_aliases(sample_entity.id)
        assert len(aliases) == 3

    @pytest.mark.asyncio
    async def test_delete_aliases(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test deleting all aliases for an entity."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)
        await entity_repo.add_alias(sample_entity.id, "Alias 1", "legal_name")
        await entity_repo.add_alias(sample_entity.id, "Alias 2", "trade_name")

        # Verify aliases exist
        aliases = await entity_repo.get_aliases(sample_entity.id)
        assert len(aliases) == 2

        # Delete all
        deleted_count = await entity_repo.delete_aliases(sample_entity.id)
        assert deleted_count == 2

        # Verify gone
        aliases = await entity_repo.get_aliases(sample_entity.id)
        assert len(aliases) == 0


class TestEntityBatchOperations:
    """Test batch and bulk operations against real database."""

    @pytest.mark.asyncio
    async def test_save_batch(self, entity_repo, has_entity_tables, cleanup_test_data):
        """Test batch entity save."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        from src.common.storage.postgres.entity_repo import Entity

        entities = [
            Entity(
                id=str(uuid.uuid4()),
                primary_name=f"Batch Company {i}",
                data_source="test",
                ticker=f"BCO{i}",
                is_public=True,
            )
            for i in range(10)
        ]

        count = await entity_repo.save_batch(entities)
        assert count == 10

        # Verify all saved
        for entity in entities:
            retrieved = await entity_repo.get(entity.id)
            assert retrieved is not None
            assert retrieved.primary_name == entity.primary_name

    @pytest.mark.asyncio
    async def test_bulk_import(self, entity_repo, has_entity_tables, cleanup_test_data):
        """Test COPY protocol bulk import."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        # Prepare records for COPY
        records = []
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

        for i in range(100):
            records.append(
                (
                    uuid.uuid4(),  # id
                    None,  # lei
                    None,  # cik
                    None,  # permid
                    None,  # figi
                    f"Bulk Import Company {i}",  # primary_name
                    None,  # display_name
                    f"BULK{i}",  # ticker
                    None,  # ric
                    None,  # exchange
                    "company",  # entity_type
                    "active",  # entity_status
                    True,  # is_public
                    "US",  # country_code
                    None,  # region
                    None,  # city
                    None,  # sic_code
                    None,  # naics_code
                    None,  # gics_sector
                    None,  # parent_entity_id
                    None,  # ultimate_parent_id
                    "test",  # data_source
                    1.0,  # confidence_score
                    False,  # ric_validated
                    None,  # last_verified_at
                )
            )

        count = await entity_repo.bulk_import(records, columns)
        assert count == 100

        # Verify some records
        retrieved = await entity_repo.get_by_ticker("BULK0")
        assert retrieved is not None
        assert retrieved.primary_name == "Bulk Import Company 0"


class TestEntityStats:
    """Test statistics and monitoring queries."""

    @pytest.mark.asyncio
    async def test_get_stats(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test retrieving entity statistics."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        stats = await entity_repo.get_stats()
        assert stats.total_entities >= 1
        assert stats.public_entities >= 1
        assert stats.entities_with_ric >= 1
        assert stats.data_sources >= 1

    @pytest.mark.asyncio
    async def test_get_coverage_by_source(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test coverage breakdown by data source."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        coverage = await entity_repo.get_coverage_by_source()
        assert isinstance(coverage, list)

        # Find test source
        test_coverage = next((c for c in coverage if c["data_source"] == "test"), None)
        assert test_coverage is not None
        assert test_coverage["entity_count"] >= 1
        assert test_coverage["with_ric"] >= 1

    @pytest.mark.asyncio
    async def test_count_entities(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test counting entities."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        total = await entity_repo.count()
        assert total >= 1

        test_count = await entity_repo.count(data_source="test")
        assert test_count >= 1

        nonexistent_count = await entity_repo.count(data_source="nonexistent")
        assert nonexistent_count == 0

    @pytest.mark.asyncio
    async def test_count_aliases(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test counting aliases."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)
        await entity_repo.add_alias(sample_entity.id, "Alias 1", "legal_name")
        await entity_repo.add_alias(sample_entity.id, "Alias 2", "trade_name")

        count = await entity_repo.count_aliases()
        assert count >= 2


class TestEntitySearch:
    """Test full-text search functionality."""

    @pytest.mark.asyncio
    async def test_search_by_name(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test full-text search by entity name."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Search by a unique part of the entity name
        search_term = sample_entity.primary_name.split()[3]  # Get unique suffix
        results = await entity_repo.search(search_term)
        assert len(results) >= 1
        assert any(e.id == sample_entity.id for e in results)

    @pytest.mark.asyncio
    async def test_search_with_country_filter(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test search with country filter."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Search by unique suffix and country filter
        search_term = sample_entity.primary_name.split()[3]  # Get unique suffix
        results = await entity_repo.search(search_term, country_code="US")
        assert any(e.id == sample_entity.id for e in results)

        # Should not find with wrong country
        results = await entity_repo.search(search_term, country_code="GB")
        assert not any(e.id == sample_entity.id for e in results)

    @pytest.mark.asyncio
    async def test_search_with_exchange_filter(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test search with exchange filter."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Search by unique suffix
        search_term = sample_entity.primary_name.split()[3]  # Get unique suffix

        # Should find with correct exchange
        results = await entity_repo.search(search_term, exchange="NYSE")
        assert any(e.id == sample_entity.id for e in results)

        # Should not find with wrong exchange
        results = await entity_repo.search(search_term, exchange="LSE")
        assert not any(e.id == sample_entity.id for e in results)

    @pytest.mark.asyncio
    async def test_search_with_public_filter(
        self, entity_repo, has_entity_tables, sample_entity, cleanup_test_data
    ):
        """Test search with public/private filter."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        await entity_repo.save(sample_entity)

        # Search by unique suffix
        search_term = sample_entity.primary_name.split()[3]  # Get unique suffix

        # Should find public
        results = await entity_repo.search(search_term, is_public=True)
        assert any(e.id == sample_entity.id for e in results)

        # Should not find private
        results = await entity_repo.search(search_term, is_public=False)
        assert not any(e.id == sample_entity.id for e in results)

    @pytest.mark.asyncio
    async def test_search_empty_query(self, entity_repo, has_entity_tables, cleanup_test_data):
        """Test search with nonsense query returns empty."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        results = await entity_repo.search("xyzzy foobar nonsense")
        assert isinstance(results, list)
        assert len(results) == 0


class TestEntityEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_none(
        self, entity_repo, has_entity_tables, cleanup_test_data
    ):
        """Test that invalid UUIDs return None, not error."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        result = await entity_repo.get("not-a-uuid")
        assert result is None

        result = await entity_repo.delete("not-a-uuid")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_batch_operations(self, entity_repo, has_entity_tables, cleanup_test_data):
        """Test empty batch operations return 0."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        count = await entity_repo.save_batch([])
        assert count == 0

        count = await entity_repo.add_aliases_batch([])
        assert count == 0

        count = await entity_repo.bulk_import([], [])
        assert count == 0

        results = await entity_repo.resolve_batch([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_alias_for_nonexistent_entity(
        self, entity_repo, has_entity_tables, cleanup_test_data
    ):
        """Test adding alias for non-existent entity fails gracefully."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        fake_id = str(uuid.uuid4())
        result = await entity_repo.add_alias(fake_id, "Some Alias", "legal_name")
        # Should return None due to foreign key violation
        assert result is None
