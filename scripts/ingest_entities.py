#!/usr/bin/env python3
"""
Entity Ingestion CLI

Unified command-line interface for entity data ingestion.
Supports GLEIF and SEC EDGAR data sources.

Usage:
    # Ingest all sources (auto mode)
    .venv/bin/python scripts/ingest_entities.py ingest

    # Ingest specific source
    .venv/bin/python scripts/ingest_entities.py ingest --source gleif
    .venv/bin/python scripts/ingest_entities.py ingest --source sec_edgar

    # Full vs delta mode
    .venv/bin/python scripts/ingest_entities.py ingest --mode full
    .venv/bin/python scripts/ingest_entities.py ingest --mode delta

    # Check status
    .venv/bin/python scripts/ingest_entities.py status

    # Reset state (force full re-ingestion)
    .venv/bin/python scripts/ingest_entities.py reset --source gleif --confirm

    # Dry run
    .venv/bin/python scripts/ingest_entities.py ingest --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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


async def cmd_ingest(args):
    """Run entity ingestion."""
    from src.nl2api.ingestion import EntityIngestionConfig

    config = EntityIngestionConfig()
    sources = args.source.split(",") if args.source else ["gleif", "sec_edgar"]

    results = {}

    for source in sources:
        source = source.strip().lower()
        logger.info("=" * 60)
        logger.info("INGESTING: %s", source.upper())
        logger.info("=" * 60)

        if source == "gleif":
            result = await ingest_gleif(config, args)
        elif source == "sec_edgar":
            result = await ingest_sec_edgar(config, args)
        else:
            logger.error("Unknown source: %s", source)
            continue

        results[source] = result

    return results


async def ingest_gleif(config, args):
    """Run GLEIF ingestion."""
    import subprocess

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ingest_gleif.py"),
        f"--mode={args.mode}",
        f"--batch-size={args.batch_size}",
    ]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.resume:
        cmd.append("--resume")
    if args.skip_aliases:
        cmd.append("--skip-aliases")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    return {"exit_code": result.returncode}


async def ingest_sec_edgar(config, args):
    """Run SEC EDGAR ingestion."""
    import subprocess

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ingest_sec_edgar.py"),
    ]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_aliases:
        cmd.append("--skip-aliases")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    return {"exit_code": result.returncode}


async def cmd_status(args):
    """Show ingestion status."""
    import asyncpg
    from src.nl2api.ingestion import EntityIngestionConfig

    config = EntityIngestionConfig()

    print("=" * 60)
    print("ENTITY INGESTION STATUS")
    print("=" * 60)

    # Load state files
    gleif_state_file = config.data_dir / "gleif_state.json"
    sec_state_file = config.data_dir / "sec_edgar_state.json"

    print("\n[GLEIF State]")
    if gleif_state_file.exists():
        with open(gleif_state_file) as f:
            state = json.load(f)
        print(f"  Last full ingestion: {state.get('last_full_ingestion', 'Never')}")
        print(f"  Last delta date: {state.get('last_delta_date', 'Never')}")
    else:
        print("  No ingestion performed yet")

    print("\n[SEC EDGAR State]")
    if sec_state_file.exists():
        with open(sec_state_file) as f:
            state = json.load(f)
        print(f"  Last ingestion: {state.get('last_ingestion', 'Never')}")
        print(f"  Last entity count: {state.get('last_entity_count', 0):,}")
    else:
        print("  No ingestion performed yet")

    # Database stats
    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")

    try:
        pool = await asyncpg.create_pool(db_url)
        try:
            async with pool.acquire() as conn:
                # Total entities
                total = await conn.fetchval("SELECT COUNT(*) FROM entities")

                # By source
                by_source = await conn.fetch(
                    """
                    SELECT data_source, COUNT(*) as count,
                           COUNT(*) FILTER (WHERE is_public) as public_count,
                           COUNT(*) FILTER (WHERE ric IS NOT NULL) as with_ric
                    FROM entities
                    WHERE entity_status = 'active'
                    GROUP BY data_source
                    ORDER BY count DESC
                    """
                )

                # Aliases
                total_aliases = await conn.fetchval("SELECT COUNT(*) FROM entity_aliases")

                print("\n[Database Statistics]")
                print(f"  Total entities: {total:,}")
                print(f"  Total aliases: {total_aliases:,}")

                print("\n  By Source:")
                for row in by_source:
                    print(
                        f"    {row['data_source']:12} {row['count']:>10,} entities "
                        f"({row['public_count']:,} public, {row['with_ric']:,} with RIC)"
                    )

                # Coverage stats
                print("\n[Coverage Statistics]")
                stats = await conn.fetchrow("SELECT * FROM entity_stats")
                if stats:
                    print(f"  Public companies: {stats['public_entities']:,}")
                    print(f"  Private entities: {stats['private_entities']:,}")
                    print(f"  With RIC: {stats['entities_with_ric']:,}")
                    print(f"  RIC validated: {stats['entities_with_validated_ric']:,}")
                    print(f"  Countries: {stats['countries']}")
                    print(f"  Exchanges: {stats['exchanges']}")

        finally:
            await pool.close()
    except Exception as e:
        print(f"\n[Database Error] Could not connect: {e}")


async def cmd_reset(args):
    """Reset ingestion state."""
    from src.nl2api.ingestion import CheckpointManager, EntityIngestionConfig

    if not args.confirm:
        print("ERROR: Must use --confirm to reset ingestion state")
        print("This will force a full re-ingestion on next run")
        return

    config = EntityIngestionConfig()

    sources = args.source.split(",") if args.source else ["gleif", "sec_edgar"]

    for source in sources:
        source = source.strip().lower()
        print(f"Resetting {source}...")

        # Delete state file
        state_file = config.data_dir / f"{source}_state.json"
        if state_file.exists():
            state_file.unlink()
            print(f"  Deleted: {state_file}")

        # Delete checkpoint
        checkpoint_mgr = CheckpointManager(config.checkpoint_dir, source)
        if checkpoint_mgr.delete():
            print(f"  Deleted checkpoint for {source}")

    print("\nReset complete. Next ingestion will be a full refresh.")


def main():
    parser = argparse.ArgumentParser(
        description="Entity ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run entity ingestion")
    ingest_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source(s) to ingest: gleif, sec_edgar, or comma-separated list (default: all)",
    )
    ingest_parser.add_argument(
        "--mode",
        choices=["auto", "full", "delta"],
        default="auto",
        help="Ingestion mode (default: auto)",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without loading",
    )
    ingest_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size (default: 50000)",
    )
    ingest_parser.add_argument(
        "--skip-aliases",
        action="store_true",
        help="Skip alias generation",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show ingestion status")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset ingestion state")
    reset_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source(s) to reset (default: all)",
    )
    reset_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm reset (required)",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(cmd_ingest(args))
    elif args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "reset":
        asyncio.run(cmd_reset(args))


if __name__ == "__main__":
    main()
