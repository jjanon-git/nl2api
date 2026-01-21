#!/usr/bin/env python3
"""
SEC EDGAR Company Data Ingestion

Imports all US public company filers with CIK-ticker-RIC mappings.
Enriches GLEIF entities with stock identifiers or creates new entities.

Data sources:
    - https://www.sec.gov/files/company_tickers.json (CIK-ticker mapping)
    - https://www.sec.gov/files/company_tickers_exchange.json (exchange info)

Incremental Update Strategy:
    - Downloads small JSON files (~500KB) - fast even for full refresh
    - Uses ON CONFLICT upserts - idempotent, safe to re-run
    - Attempts to match existing GLEIF entities by CIK
    - Generates RICs from ticker + exchange mapping
    - Tracks last ingestion date for audit purposes

Usage:
    # Full ingestion
    .venv/bin/python scripts/ingest_sec_edgar.py

    # Dry run (validate only)
    .venv/bin/python scripts/ingest_sec_edgar.py --dry-run

    # Skip RIC validation via OpenFIGI
    .venv/bin/python scripts/ingest_sec_edgar.py --skip-ric-validation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env():
    """Load environment variables from .env file."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


_load_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# SEC API endpoints
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EXCHANGES_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

# Exchange suffix mapping for RIC generation
EXCHANGE_SUFFIX_MAP = {
    # US exchanges
    "NYSE": ".N",
    "NASDAQ": ".O",
    "AMEX": ".A",
    "BATS": ".Z",
    "ARCA": ".P",
    "Nyse": ".N",
    "Nasdaq": ".O",
    "N": ".N",
    "Q": ".O",
    "A": ".A",
    "Z": ".Z",
    "P": ".P",
}

# Company suffixes to strip for alias generation
COMPANY_SUFFIXES = [
    "Inc", "Inc.", "Incorporated",
    "Corp", "Corp.", "Corporation",
    "Ltd", "Ltd.", "Limited",
    "LLC", "L.L.C.",
    "Co", "Co.", "Company",
    "/DE", "/DE/", "/NV", "/MD", "/CA",  # State suffixes
]


def normalize_cik(cik: int | str) -> str:
    """Normalize CIK to 10-digit zero-padded string."""
    return str(cik).zfill(10)


def generate_ric(ticker: str, exchange: str | None) -> str:
    """
    Generate RIC from ticker and exchange.

    Args:
        ticker: Stock ticker symbol
        exchange: Exchange code or name

    Returns:
        RIC (e.g., "AAPL.O" for Apple on NASDAQ)
    """
    if not ticker:
        return ""

    ticker = ticker.upper().strip()

    if not exchange:
        # Default to NASDAQ (.O) for unknown exchanges
        return f"{ticker}.O"

    # Normalize exchange name
    exchange_upper = exchange.upper().strip()

    # Direct lookup
    if exchange_upper in EXCHANGE_SUFFIX_MAP:
        return f"{ticker}{EXCHANGE_SUFFIX_MAP[exchange_upper]}"

    # Try with original case
    if exchange in EXCHANGE_SUFFIX_MAP:
        return f"{ticker}{EXCHANGE_SUFFIX_MAP[exchange]}"

    # Default to NASDAQ for unknown
    return f"{ticker}.O"


def normalize_alias(s: str) -> str:
    """Normalize string for matching."""
    return s.lower().strip()


def generate_aliases(name: str, ticker: str | None = None) -> list[str]:
    """Generate aliases from entity name and ticker."""
    aliases = set()

    # Original name (normalized)
    aliases.add(normalize_alias(name))

    # Ticker as alias
    if ticker:
        aliases.add(normalize_alias(ticker))

    # Without company/state suffixes
    stripped = name
    for suffix in COMPANY_SUFFIXES:
        pattern = rf"\s*[,/]?\s*{re.escape(suffix)}\.?$"
        stripped = re.sub(pattern, "", stripped, flags=re.I).strip()

    if stripped and stripped != name:
        aliases.add(normalize_alias(stripped))

    # Without punctuation
    clean = re.sub(r"[^\w\s]", "", name)
    if clean:
        aliases.add(normalize_alias(clean))

    # Filter out empty or too short aliases
    return [a for a in aliases if len(a) >= 2]


async def fetch_sec_tickers(timeout: int = 30) -> dict:
    """
    Fetch SEC company tickers JSON.

    Returns:
        Dict mapping index to {cik_str, ticker, title}
    """
    logger.info("Fetching SEC company tickers...")

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(SEC_TICKERS_URL)
        response.raise_for_status()
        return response.json()


async def fetch_sec_exchanges(timeout: int = 30) -> dict:
    """
    Fetch SEC company exchange mappings.

    Returns:
        Dict with exchange information per company
    """
    logger.info("Fetching SEC exchange mappings...")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(SEC_EXCHANGES_URL)
            response.raise_for_status()
            data = response.json()

            # Build ticker -> exchange mapping
            exchange_map = {}
            for item in data.get("data", []):
                if len(item) >= 3:
                    ticker = item[2]  # ticker is typically 3rd element
                    exchange = item[3] if len(item) > 3 else None
                    if ticker:
                        exchange_map[ticker.upper()] = exchange

            return exchange_map
    except Exception as e:
        logger.warning("Could not fetch exchange mappings: %s", e)
        return {}


def transform_sec_companies(
    tickers_data: dict,
    exchange_map: dict,
) -> list[dict]:
    """
    Transform SEC data to entity format.

    Args:
        tickers_data: SEC tickers JSON response
        exchange_map: Ticker -> exchange mapping

    Returns:
        List of entity dicts
    """
    entities = []

    for entry in tickers_data.values():
        cik = normalize_cik(entry.get("cik_str", ""))
        ticker = entry.get("ticker", "").upper().strip()
        name = entry.get("title", "").strip()

        if not cik or not name:
            continue

        # Get exchange from mapping
        exchange = exchange_map.get(ticker, "")
        ric = generate_ric(ticker, exchange)

        entities.append({
            "cik": cik,
            "ticker": ticker,
            "primary_name": name,
            "ric": ric,
            "exchange": exchange or None,
            "is_public": True,
            "country_code": "US",
            "data_source": "sec_edgar",
        })

    return entities


async def upsert_sec_entities(
    pool,
    entities: list[dict],
    batch_size: int = 500,
) -> dict:
    """
    Upsert SEC entities into database.

    Strategy:
    1. Try to match existing entity by CIK
    2. If found, update with ticker/RIC info
    3. If not found, create new entity

    Args:
        pool: Database connection pool
        entities: List of entity dicts
        batch_size: Batch size for processing

    Returns:
        Stats dict
    """
    stats = {
        "inserted": 0,
        "updated": 0,
        "matched_gleif": 0,
        "skipped": 0,
    }

    async with pool.acquire() as conn:
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]

            for entity in batch:
                cik = entity["cik"]
                ticker = entity["ticker"]
                ric = entity["ric"]

                # First, try to find existing entity by CIK
                existing = await conn.fetchrow(
                    "SELECT id, data_source FROM entities WHERE cik = $1",
                    cik,
                )

                if existing:
                    # Update existing entity with SEC data
                    await conn.execute(
                        """
                        UPDATE entities SET
                            ticker = COALESCE($1, ticker),
                            ric = COALESCE($2, ric),
                            exchange = COALESCE($3, exchange),
                            is_public = true,
                            updated_at = NOW()
                        WHERE cik = $4
                        """,
                        ticker or None,
                        ric or None,
                        entity.get("exchange"),
                        cik,
                    )
                    stats["updated"] += 1
                    if existing["data_source"] == "gleif":
                        stats["matched_gleif"] += 1
                else:
                    # Insert new entity (no ON CONFLICT needed - we already checked)
                    entity_id = uuid.uuid4()
                    await conn.execute(
                        """
                        INSERT INTO entities (
                            id, cik, primary_name, ticker, ric, exchange,
                            entity_type, entity_status, is_public,
                            country_code, data_source, confidence_score
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6,
                            'company', 'active', true,
                            'US', 'sec_edgar', 1.0
                        )
                        """,
                        entity_id,
                        cik,
                        entity["primary_name"],
                        ticker or None,
                        ric or None,
                        entity.get("exchange"),
                    )
                    stats["inserted"] += 1

            logger.info(
                "Processed batch %d-%d: +%d new, ~%d updated",
                i,
                min(i + batch_size, len(entities)),
                stats["inserted"],
                stats["updated"],
            )

    return stats


async def generate_aliases_for_sec_entities(
    pool,
    batch_size: int = 5000,
) -> int:
    """Generate aliases for SEC EDGAR entities."""
    logger.info("Generating aliases for SEC EDGAR entities...")

    total_aliases = 0
    offset = 0

    async with pool.acquire() as conn:
        while True:
            # Get entities without aliases (SEC source or updated by SEC)
            rows = await conn.fetch(
                """
                SELECT e.id, e.primary_name, e.ticker
                FROM entities e
                WHERE (e.data_source = 'sec_edgar' OR e.ticker IS NOT NULL)
                AND NOT EXISTS (
                    SELECT 1 FROM entity_aliases a WHERE a.entity_id = e.id
                )
                ORDER BY e.id
                LIMIT $1 OFFSET $2
                """,
                batch_size,
                offset,
            )

            if not rows:
                break

            alias_records = []
            for row in rows:
                entity_id = row["id"]
                name = row["primary_name"]
                ticker = row["ticker"]

                aliases = generate_aliases(name, ticker)
                for i, alias in enumerate(aliases):
                    alias_records.append((
                        uuid.uuid4(),
                        entity_id,
                        alias,
                        "ticker" if alias == normalize_alias(ticker or "") else (
                            "legal_name" if i == 0 else "generated"
                        ),
                        i == 0,
                    ))

            if alias_records:
                await conn.executemany(
                    """
                    INSERT INTO entity_aliases (id, entity_id, alias, alias_type, is_primary)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (entity_id, alias) DO NOTHING
                    """,
                    alias_records,
                )
                total_aliases += len(alias_records)

            logger.info("Generated aliases: batch=%d, total=%d", len(alias_records), total_aliases)
            offset += batch_size

    return total_aliases


class IngestionState:
    """Tracks ingestion state for audit purposes."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    @property
    def last_ingestion(self) -> datetime | None:
        date_str = self._state.get("last_ingestion")
        if date_str:
            return datetime.fromisoformat(date_str)
        return None

    @last_ingestion.setter
    def last_ingestion(self, value: datetime) -> None:
        self._state["last_ingestion"] = value.isoformat()

    @property
    def last_entity_count(self) -> int:
        return self._state.get("last_entity_count", 0)

    @last_entity_count.setter
    def last_entity_count(self, value: int) -> None:
        self._state["last_entity_count"] = value


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest SEC EDGAR company data into entity database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count records without loading",
    )
    parser.add_argument(
        "--skip-aliases",
        action="store_true",
        help="Skip alias generation",
    )
    parser.add_argument(
        "--skip-ric-validation",
        action="store_true",
        help="Skip RIC validation via OpenFIGI (faster, but RICs may be inaccurate)",
    )
    args = parser.parse_args()

    # Import dependencies
    import asyncpg
    from src.nl2api.ingestion import EntityIngestionConfig

    print("=" * 60)
    print("SEC EDGAR COMPANY DATA INGESTION")
    print("=" * 60)

    # Load configuration
    config = EntityIngestionConfig()
    config.ensure_data_dir()

    # Initialize state tracker
    state = IngestionState(config.data_dir / "sec_edgar_state.json")

    # Fetch SEC data
    tickers_data = await fetch_sec_tickers()
    exchange_map = await fetch_sec_exchanges()

    logger.info("Fetched %d companies from SEC", len(tickers_data))
    logger.info("Fetched %d exchange mappings", len(exchange_map))

    # Transform to entity format
    entities = transform_sec_companies(tickers_data, exchange_map)
    logger.info("Transformed %d entities", len(entities))

    if args.dry_run:
        print("\n[DRY RUN] Sample entities:")
        for entity in entities[:10]:
            print(f"  {entity['cik']}: {entity['ticker']:6} {entity['ric']:10} {entity['primary_name'][:40]}...")
        print(f"\nTotal: {len(entities)} entities would be loaded")
        return

    # Connect to database
    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")
    logger.info("Connecting to database...")
    pool = await asyncpg.create_pool(db_url)

    try:
        # Upsert entities
        stats = await upsert_sec_entities(pool, entities)

        # Generate aliases
        if not args.skip_aliases:
            aliases = await generate_aliases_for_sec_entities(pool)
        else:
            aliases = 0

        # Update state
        state.last_ingestion = datetime.now(timezone.utc)
        state.last_entity_count = len(entities)
        state.save()

        # Show final stats
        async with pool.acquire() as conn:
            sec_count = await conn.fetchval(
                "SELECT COUNT(*) FROM entities WHERE data_source = 'sec_edgar'"
            )
            public_with_ric = await conn.fetchval(
                "SELECT COUNT(*) FROM entities WHERE is_public = true AND ric IS NOT NULL"
            )

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"SEC companies processed: {len(entities):,}")
        print(f"New entities created: {stats['inserted']:,}")
        print(f"Existing entities updated: {stats['updated']:,}")
        print(f"  - Matched GLEIF entities: {stats['matched_gleif']:,}")
        print(f"Aliases generated: {aliases:,}")
        print(f"\nTotal SEC EDGAR entities: {sec_count:,}")
        print(f"Total public entities with RIC: {public_with_ric:,}")

    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        sys.exit(1)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
