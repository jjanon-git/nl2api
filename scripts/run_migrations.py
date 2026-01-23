#!/usr/bin/env python3
"""
Run PostgreSQL Migrations

Executes all SQL migrations in the src/common/storage/postgres/migrations/
directory in alphanumeric order. Tracks applied migrations in a
migration_history table.

Usage:
    .venv/bin/python scripts/run_migrations.py
"""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg

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

MIGRATIONS_DIR = PROJECT_ROOT / "src" / "common" / "storage" / "postgres" / "migrations"
DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("EVAL_POSTGRES_URL")
    or "postgresql://nl2api:nl2api@localhost:5432/nl2api"
)


async def run_migrations():
    print("Connecting to database...")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        return

    try:
        # Create migration history table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_history (
                migration_name TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Get already applied migrations
        applied = await conn.fetch("SELECT migration_name FROM migration_history")
        applied_names = {row["migration_name"] for row in applied}

        # Get all migration files
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        if not migration_files:
            print(f"No migration files found in {MIGRATIONS_DIR}")
            return

        for migration_file in migration_files:
            name = migration_file.name
            if name in applied_names:
                print(f"Skipping {name} (already applied)")
                continue

            print(f"Applying {name}...")
            sql = migration_file.read_text()

            # Start transaction for each migration
            async with conn.transaction():
                try:
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO migration_history (migration_name) VALUES ($1)", name
                    )
                    print(f"Successfully applied {name}")
                except Exception as e:
                    print(f"ERROR applying {name}: {e}")
                    raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
