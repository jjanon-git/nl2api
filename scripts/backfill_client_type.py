#!/usr/bin/env python
"""
Backfill client_type for existing scorecards.

This script updates scorecards that have NULL client_type to use
'internal' as the default value, ensuring historical data is included
in client-based comparisons.

Usage:
    python scripts/backfill_client_type.py
    python scripts/backfill_client_type.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def backfill_scorecards(dry_run: bool = False) -> None:
    """
    Backfill client_type and eval_mode for existing scorecards.

    Args:
        dry_run: If True, only show what would be updated without making changes
    """
    from src.evalkit.common.storage.postgres.client import close_pool, get_pool

    try:
        pool = await get_pool()

        # Count scorecards with NULL client_type
        count_query = """
            SELECT COUNT(*) FROM scorecards WHERE client_type IS NULL
        """
        count_result = await pool.fetchval(count_query)
        logger.info(f"Found {count_result} scorecards with NULL client_type")

        if count_result == 0:
            logger.info("No scorecards need backfilling. Exiting.")
            return

        if dry_run:
            logger.info("DRY RUN - No changes will be made")

            # Show sample of affected scorecards
            sample_query = """
                SELECT scorecard_id, test_case_id, batch_id, timestamp
                FROM scorecards
                WHERE client_type IS NULL
                ORDER BY timestamp DESC
                LIMIT 10
            """
            samples = await pool.fetch(sample_query)

            logger.info("\nSample scorecards that would be updated:")
            for row in samples:
                logger.info(
                    f"  - {row['scorecard_id']} (batch: {row['batch_id']}, "
                    f"timestamp: {row['timestamp']})"
                )

            if count_result > 10:
                logger.info(f"  ... and {count_result - 10} more")

            return

        # Perform the backfill
        logger.info("Starting backfill...")

        update_query = """
            UPDATE scorecards
            SET client_type = 'internal',
                eval_mode = COALESCE(eval_mode, 'orchestrator')
            WHERE client_type IS NULL
        """

        result = await pool.execute(update_query)

        # Parse the result to get affected row count
        # asyncpg returns something like "UPDATE 1234"
        affected = int(result.split()[-1]) if result else 0

        logger.info(f"Successfully updated {affected} scorecards")

        # Verify the update
        verify_query = """
            SELECT COUNT(*) FROM scorecards WHERE client_type IS NULL
        """
        remaining = await pool.fetchval(verify_query)

        if remaining == 0:
            logger.info("Verification passed: All scorecards now have client_type set")
        else:
            logger.warning(
                f"Verification warning: {remaining} scorecards still have NULL client_type"
            )

    except Exception as e:
        logger.error(f"Error during backfill: {e}")
        raise

    finally:
        await close_pool()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backfill client_type for existing scorecards")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    args = parser.parse_args()

    try:
        asyncio.run(backfill_scorecards(dry_run=args.dry_run))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
