#!/usr/bin/env python3
"""
Backfill SEC filing metadata in rag_documents.

Adds new metadata fields to existing SEC filing chunks:
- fiscal_quarter: Quarter number (1-4) for 10-Q filings, null for 10-K
- is_amendment: Boolean indicating if filing is an amendment (/A)

Usage:
    python scripts/backfill_sec_metadata.py
    python scripts/backfill_sec_metadata.py --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def get_stats(conn: asyncpg.Connection) -> dict:
    """Get current metadata statistics."""
    stats = {}

    # Total SEC filing documents
    stats["total"] = await conn.fetchval(
        "SELECT COUNT(*) FROM rag_documents WHERE document_type = 'sec_filing'"
    )

    # Documents with fiscal_quarter
    stats["has_fiscal_quarter"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM rag_documents
        WHERE document_type = 'sec_filing'
        AND metadata ? 'fiscal_quarter'
        """
    )

    # Documents with is_amendment
    stats["has_is_amendment"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM rag_documents
        WHERE document_type = 'sec_filing'
        AND metadata ? 'is_amendment'
        """
    )

    # 10-Q filings (should have fiscal_quarter)
    stats["10q_filings"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM rag_documents
        WHERE document_type = 'sec_filing'
        AND metadata->>'filing_type' LIKE '10-Q%'
        """
    )

    # 10-K filings (fiscal_quarter should be null)
    stats["10k_filings"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM rag_documents
        WHERE document_type = 'sec_filing'
        AND metadata->>'filing_type' LIKE '10-K%'
        """
    )

    # Amendment filings
    stats["amendments"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM rag_documents
        WHERE document_type = 'sec_filing'
        AND metadata->>'filing_type' LIKE '%/A'
        """
    )

    return stats


async def backfill_fiscal_quarter(conn: asyncpg.Connection, dry_run: bool = False) -> int:
    """
    Backfill fiscal_quarter for 10-Q filings.

    Derives quarter from period_of_report month:
    - Q1: months 1-3
    - Q2: months 4-6
    - Q3: months 7-9
    - Q4: months 10-12
    """
    if dry_run:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM rag_documents
            WHERE document_type = 'sec_filing'
            AND metadata->>'filing_type' LIKE '10-Q%'
            AND NOT (metadata ? 'fiscal_quarter')
            AND metadata->>'period_of_report' IS NOT NULL
            """
        )
        logger.info(f"[DRY RUN] Would update {count} 10-Q records with fiscal_quarter")
        return count

    # Update 10-Q filings with derived fiscal_quarter
    result = await conn.execute(
        """
        UPDATE rag_documents
        SET metadata = metadata || jsonb_build_object(
            'fiscal_quarter',
            CASE
                WHEN EXTRACT(MONTH FROM (metadata->>'period_of_report')::timestamp) <= 3 THEN 1
                WHEN EXTRACT(MONTH FROM (metadata->>'period_of_report')::timestamp) <= 6 THEN 2
                WHEN EXTRACT(MONTH FROM (metadata->>'period_of_report')::timestamp) <= 9 THEN 3
                ELSE 4
            END
        )
        WHERE document_type = 'sec_filing'
        AND metadata->>'filing_type' LIKE '10-Q%'
        AND NOT (metadata ? 'fiscal_quarter')
        AND metadata->>'period_of_report' IS NOT NULL
        """
    )
    count = int(result.split()[-1])
    logger.info(f"Updated {count} 10-Q records with fiscal_quarter")
    return count


async def backfill_10k_fiscal_quarter(conn: asyncpg.Connection, dry_run: bool = False) -> int:
    """
    Backfill fiscal_quarter as null for 10-K filings.

    10-K filings cover the full fiscal year, not a specific quarter.
    """
    if dry_run:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM rag_documents
            WHERE document_type = 'sec_filing'
            AND metadata->>'filing_type' LIKE '10-K%'
            AND NOT (metadata ? 'fiscal_quarter')
            """
        )
        logger.info(f"[DRY RUN] Would update {count} 10-K records with fiscal_quarter=null")
        return count

    # Update 10-K filings with null fiscal_quarter
    result = await conn.execute(
        """
        UPDATE rag_documents
        SET metadata = metadata || '{"fiscal_quarter": null}'::jsonb
        WHERE document_type = 'sec_filing'
        AND metadata->>'filing_type' LIKE '10-K%'
        AND NOT (metadata ? 'fiscal_quarter')
        """
    )
    count = int(result.split()[-1])
    logger.info(f"Updated {count} 10-K records with fiscal_quarter=null")
    return count


async def backfill_is_amendment(conn: asyncpg.Connection, dry_run: bool = False) -> int:
    """
    Backfill is_amendment flag based on filing_type.

    Amendments have filing_type ending in "/A" (e.g., 10-K/A, 10-Q/A).
    """
    if dry_run:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM rag_documents
            WHERE document_type = 'sec_filing'
            AND NOT (metadata ? 'is_amendment')
            """
        )
        logger.info(f"[DRY RUN] Would update {count} records with is_amendment")
        return count

    # Update all SEC filings with is_amendment
    result = await conn.execute(
        """
        UPDATE rag_documents
        SET metadata = metadata || jsonb_build_object(
            'is_amendment',
            (metadata->>'filing_type' LIKE '%/A')
        )
        WHERE document_type = 'sec_filing'
        AND NOT (metadata ? 'is_amendment')
        """
    )
    count = int(result.split()[-1])
    logger.info(f"Updated {count} records with is_amendment")
    return count


async def main(dry_run: bool = False) -> None:
    """Run the backfill."""
    database_url = "postgresql://nl2api:nl2api@localhost:5432/nl2api"

    logger.info("Connecting to database...")
    conn = await asyncpg.connect(database_url)

    try:
        # Get initial stats
        logger.info("Initial statistics:")
        stats = await get_stats(conn)
        for key, value in stats.items():
            logger.info(f"  {key}: {value:,}")

        # Run backfills
        logger.info("\nRunning backfills...")

        total_updated = 0
        total_updated += await backfill_fiscal_quarter(conn, dry_run)
        total_updated += await backfill_10k_fiscal_quarter(conn, dry_run)
        total_updated += await backfill_is_amendment(conn, dry_run)

        # Get final stats
        if not dry_run:
            logger.info("\nFinal statistics:")
            stats = await get_stats(conn)
            for key, value in stats.items():
                logger.info(f"  {key}: {value:,}")

        logger.info(f"\nTotal records {'would be ' if dry_run else ''}updated: {total_updated:,}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill SEC filing metadata")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
