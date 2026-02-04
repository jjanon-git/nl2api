#!/usr/bin/env python3
"""Export database to compressed dump for GitHub Release upload.

Usage:
    python scripts/db-export.py                    # Export all tables
    python scripts/db-export.py --tables rag      # Export only RAG data
    python scripts/db-export.py --tables entities # Export only entity data
    python scripts/db-export.py --tables fixtures # Export only test fixtures

Output: exports/evalkit_<tables>_<timestamp>.dump.gz
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Table groups for selective export
TABLE_GROUPS = {
    "all": None,  # None = export everything
    "rag": [
        "rag_documents",
        "sec_filings",
        "sec_filing_ingestion_jobs",
        "indexing_checkpoint",
    ],
    "entities": [
        "entities",
        "entity_aliases",
    ],
    "fixtures": [
        "test_cases",
        "batch_jobs",
        "scorecards",
    ],
    "minimal": [
        # Just enough to run evaluations (no entities/RAG)
        "test_cases",
        "batch_jobs",
    ],
}


def get_db_url() -> tuple[str, str, str, str]:
    """Get database connection details from environment."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_DB", "evalkit")
    return host, port, user, password, database


def run_pg_dump(
    output_path: Path,
    tables: list[str] | None = None,
) -> bool:
    """Run pg_dump with optional table filtering."""
    host, port, user, password, database = get_db_url()

    # Build pg_dump command
    cmd = [
        "pg_dump",
        "-h",
        host,
        "-p",
        port,
        "-U",
        user,
        "-d",
        database,
        "-Fc",  # Custom format (compressed, supports parallel restore)
        "-v",  # Verbose
    ]

    # Add table filters if specified
    if tables:
        for table in tables:
            cmd.extend(["-t", table])

    # Set password via environment
    env = os.environ.copy()
    env["PGPASSWORD"] = password

    print(f"Exporting database to {output_path}...")
    print(f"Tables: {tables if tables else 'all'}")

    # Run pg_dump, piping through gzip
    dump_path = output_path.with_suffix("")  # Remove .gz for intermediate
    try:
        with open(dump_path, "wb") as f:
            subprocess.run(cmd, stdout=f, env=env, check=True)

        # Compress with gzip
        print("Compressing...")
        subprocess.run(["gzip", "-f", str(dump_path)], check=True)

        final_size = output_path.stat().st_size / (1024 * 1024)
        print(f"Export complete: {output_path} ({final_size:.1f} MB)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during export: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: pg_dump not found. Install PostgreSQL client tools.", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Export database for GitHub Release")
    parser.add_argument(
        "--tables",
        choices=list(TABLE_GROUPS.keys()),
        default="all",
        help="Which table group to export (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Output directory (default: exports/)",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evalkit_{args.tables}_{timestamp}.dump.gz"
    output_path = args.output_dir / filename

    # Get tables to export
    tables = TABLE_GROUPS[args.tables]

    # Run export
    success = run_pg_dump(output_path, tables)

    if success:
        print()
        print("Next steps:")
        print("  1. Upload to GitHub Release:")
        print(
            f"     gh release create data-{timestamp} --title 'Database snapshot ({args.tables})' \\"
        )
        print(f"       --notes 'Exported {args.tables} tables' {output_path}")
        print()
        print("  2. Or use the upload script:")
        print(f"     python scripts/db-upload.py {output_path}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
