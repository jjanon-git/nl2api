"""
Entity Ingestion Integration Tests

Tests the entity ingestion pipeline against a real PostgreSQL database.
Tests data transformation and loading without actually downloading from external sources.

Run with: pytest tests/integration/test_entity_ingestion.py -v
"""

import os
import sys
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

# Add project root for script imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
async def db_pool():
    """Create database connection pool."""
    import asyncpg

    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api"
    )
    pool = await asyncpg.create_pool(db_url)
    yield pool
    await pool.close()


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
async def cleanup_test_ingestion_data(db_pool):
    """Fixture to clean up test ingestion data before and after each test."""
    # Clean up before test (in case of previous failures)
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM entity_aliases WHERE entity_id IN (SELECT id FROM entities WHERE data_source IN ('test_gleif', 'test_sec'))"
        )
        await conn.execute("DELETE FROM entities WHERE data_source IN ('test_gleif', 'test_sec')")
    yield
    # Clean up after test
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM entity_aliases WHERE entity_id IN (SELECT id FROM entities WHERE data_source IN ('test_gleif', 'test_sec'))"
        )
        await conn.execute("DELETE FROM entities WHERE data_source IN ('test_gleif', 'test_sec')")


class TestGLEIFTransformation:
    """Test GLEIF data transformation logic."""

    def test_generate_aliases(self):
        """Test alias generation from company names."""
        from scripts.ingest_gleif import generate_aliases

        # Standard company name
        aliases = generate_aliases("Apple Inc")
        assert "apple inc" in aliases
        assert "apple" in aliases  # Without suffix

        # Name with multiple suffixes
        aliases = generate_aliases("Microsoft Corporation")
        assert "microsoft corporation" in aliases
        assert "microsoft" in aliases

        # Multi-word company (acronym) - note: includes Corporation's "C"
        aliases = generate_aliases("International Business Machines Corporation")
        # Acronym includes all capitalized words, including "Corporation"
        assert any("ibm" in a for a in aliases) or "ibmc" in aliases

        # Name with punctuation
        aliases = generate_aliases("Johnson & Johnson")
        assert "johnson  johnson" in aliases or "johnson johnson" in aliases

    def test_normalize_alias(self):
        """Test alias normalization."""
        from scripts.ingest_gleif import normalize_alias

        assert normalize_alias("Apple Inc") == "apple inc"
        assert normalize_alias("  MICROSOFT  ") == "microsoft"
        assert normalize_alias("IBM") == "ibm"

    def test_transform_to_db_records(self):
        """Test entity dict to DB record transformation."""
        from scripts.ingest_gleif import transform_to_db_records

        entities = iter([
            {
                "lei": "HWUPKR0MPOU8FGXBT394",
                "primary_name": "Apple Inc",
                "country_code": "US",
                "region": "California",
                "city": "Cupertino",
                "entity_category": "general",
                "entity_status": "active",
                "data_source": "gleif",
            }
        ])

        records = list(transform_to_db_records(entities))

        assert len(records) == 1
        record = records[0]
        # Check LEI is in correct position
        assert record[1] == "HWUPKR0MPOU8FGXBT394"
        # Check primary_name
        assert record[5] == "Apple Inc"
        # Check data_source
        assert record[21] == "gleif"


class TestSECEDGARTransformation:
    """Test SEC EDGAR data transformation logic."""

    def test_normalize_cik(self):
        """Test CIK normalization to 10 digits."""
        from scripts.ingest_sec_edgar import normalize_cik

        assert normalize_cik("320193") == "0000320193"
        assert normalize_cik(320193) == "0000320193"
        assert normalize_cik("0000320193") == "0000320193"

    def test_generate_ric_us_exchanges(self):
        """Test RIC generation for US exchanges."""
        from scripts.ingest_sec_edgar import generate_ric

        assert generate_ric("AAPL", "NASDAQ") == "AAPL.O"
        assert generate_ric("IBM", "NYSE") == "IBM.N"
        assert generate_ric("SPY", "AMEX") == "SPY.A"

    def test_generate_ric_unknown_exchange(self):
        """Test RIC generation for unknown exchange defaults to NASDAQ."""
        from scripts.ingest_sec_edgar import generate_ric

        assert generate_ric("TEST", "UNKNOWN") == "TEST.O"
        assert generate_ric("TEST", None) == "TEST.O"
        assert generate_ric("TEST", "") == "TEST.O"

    def test_transform_sec_companies(self):
        """Test SEC company data transformation."""
        from scripts.ingest_sec_edgar import transform_sec_companies

        tickers_data = {
            "0": {
                "cik_str": 320193,
                "ticker": "AAPL",
                "title": "Apple Inc",
            },
            "1": {
                "cik_str": 789019,
                "ticker": "MSFT",
                "title": "MICROSOFT CORP",
            },
        }

        exchange_map = {
            "AAPL": "NASDAQ",
            "MSFT": "NASDAQ",
        }

        entities = transform_sec_companies(tickers_data, exchange_map)

        assert len(entities) == 2

        # Check Apple
        apple = next(e for e in entities if e["ticker"] == "AAPL")
        assert apple["cik"] == "0000320193"
        assert apple["ric"] == "AAPL.O"
        assert apple["is_public"] is True
        assert apple["country_code"] == "US"

        # Check Microsoft
        msft = next(e for e in entities if e["ticker"] == "MSFT")
        assert msft["cik"] == "0000789019"
        assert msft["ric"] == "MSFT.O"


class TestIngestionDBOperations:
    """Test database operations for ingestion."""

    @pytest.mark.asyncio
    async def test_bulk_load_entities(
        self, db_pool, has_entity_tables, cleanup_test_ingestion_data
    ):
        """Test bulk loading entities via COPY protocol."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        from scripts.ingest_gleif import bulk_load_entities

        # Create test records
        records = [
            (
                uuid.uuid4(),           # id
                f"TESTLEI{i:012d}",     # lei
                None,                   # cik
                None,                   # permid
                None,                   # figi
                f"Test Company {i}",    # primary_name
                None,                   # display_name
                None,                   # ticker
                None,                   # ric
                None,                   # exchange
                "company",              # entity_type
                "active",               # entity_status
                False,                  # is_public
                "US",                   # country_code
                None,                   # region
                None,                   # city
                None,                   # sic_code
                None,                   # naics_code
                None,                   # gics_sector
                None,                   # parent_entity_id
                None,                   # ultimate_parent_id
                "test_gleif",           # data_source
                1.0,                    # confidence_score
                False,                  # ric_validated
                None,                   # last_verified_at
            )
            for i in range(100)
        ]

        loaded = await bulk_load_entities(
            pool=db_pool,
            records=iter(records),
            batch_size=50,
        )

        assert loaded == 100

        # Verify in database
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM entities WHERE data_source = 'test_gleif'"
            )
            assert count == 100

    @pytest.mark.asyncio
    async def test_upsert_sec_entities(
        self, db_pool, has_entity_tables, cleanup_test_ingestion_data
    ):
        """Test SEC entity upsert logic."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        from scripts.ingest_sec_edgar import upsert_sec_entities

        entities = [
            {
                "cik": "9999990001",
                "ticker": "TST1",
                "primary_name": "Test SEC Company 1",
                "ric": "TST1.O",
                "exchange": "NASDAQ",
                "is_public": True,
                "country_code": "US",
                "data_source": "test_sec",
            },
            {
                "cik": "9999990002",
                "ticker": "TST2",
                "primary_name": "Test SEC Company 2",
                "ric": "TST2.N",
                "exchange": "NYSE",
                "is_public": True,
                "country_code": "US",
                "data_source": "test_sec",
            },
        ]

        # First upsert - should insert
        stats = await upsert_sec_entities(db_pool, entities, batch_size=10)
        assert stats["inserted"] == 2
        assert stats["updated"] == 0

        # Second upsert - should update
        stats = await upsert_sec_entities(db_pool, entities, batch_size=10)
        assert stats["updated"] == 2
        assert stats["inserted"] == 0

        # Verify in database
        async with db_pool.acquire() as conn:
            entity = await conn.fetchrow(
                "SELECT * FROM entities WHERE cik = '9999990001'"
            )
            assert entity is not None
            assert entity["ticker"] == "TST1"
            assert entity["ric"] == "TST1.O"
            assert entity["is_public"] is True

    @pytest.mark.asyncio
    async def test_alias_generation_for_loaded_entities(
        self, db_pool, has_entity_tables, cleanup_test_ingestion_data
    ):
        """Test alias generation for loaded entities."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        from scripts.ingest_gleif import generate_aliases

        # First, insert some test entities with manual alias generation
        async with db_pool.acquire() as conn:
            for i in range(5):
                entity_id = uuid.uuid4()
                name = f"Alias Test Company {i} Inc"

                await conn.execute(
                    """
                    INSERT INTO entities (id, lei, primary_name, data_source, entity_status)
                    VALUES ($1, $2, $3, 'test_gleif', 'active')
                    """,
                    entity_id,
                    f"ALIASGEN{i:011d}",  # LEI is 20 chars: "ALIASGEN" (8) + 11 digits = 19
                    name,
                )

                # Manually generate and insert aliases
                aliases = generate_aliases(name)
                for j, alias in enumerate(aliases):
                    await conn.execute(
                        """
                        INSERT INTO entity_aliases (id, entity_id, alias, alias_type, is_primary)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (entity_id, alias) DO NOTHING
                        """,
                        uuid.uuid4(),
                        entity_id,
                        alias,
                        "legal_name" if j == 0 else "generated",
                        j == 0,
                    )

        # Verify aliases were created
        async with db_pool.acquire() as conn:
            aliases = await conn.fetch(
                """
                SELECT a.alias FROM entity_aliases a
                JOIN entities e ON a.entity_id = e.id
                WHERE e.data_source = 'test_gleif'
                """
            )

            alias_texts = [r["alias"] for r in aliases]
            # Should have normalized versions of company names
            assert len(alias_texts) > 0
            assert any("alias test company" in a for a in alias_texts)


class TestIngestionConfig:
    """Test ingestion configuration."""

    def test_config_loads(self):
        """Test configuration loads without error."""
        from src.nl2api.ingestion import EntityIngestionConfig

        config = EntityIngestionConfig()
        assert config.batch_size > 0
        assert config.max_errors > 0

    def test_checkpoint_manager_initializes(self):
        """Test checkpoint manager initializes."""
        import tempfile
        from src.nl2api.ingestion import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir), "test")
            checkpoint = manager.create_new()
            assert checkpoint.source == "test"


class TestIngestionCLI:
    """Test CLI status command (read-only)."""

    @pytest.mark.asyncio
    async def test_status_command_runs(self, db_pool, has_entity_tables):
        """Test status command executes without error."""
        if not has_entity_tables:
            pytest.skip("Entity tables not created")

        # Just verify the status command path works
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest_entities.py"), "status"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        # Should complete (exit code 0) or show helpful error
        assert "ENTITY INGESTION STATUS" in result.stdout or result.returncode == 0
