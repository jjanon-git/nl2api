#!/usr/bin/env python3
"""Restore database from GitHub Release dump.

Usage:
    # Download and restore latest
    python scripts/db-restore.py --download data-all-20260203

    # Restore from local file
    python scripts/db-restore.py exports/evalkit_all_20260203.dump.gz

    # Restore only specific tables (if dump contains more)
    python scripts/db-restore.py exports/evalkit_all.dump.gz --tables rag_documents entities

Options:
    --download TAG    Download from GitHub Release first
    --clean           Drop and recreate database before restore
    --tables          Restore only specific tables from dump
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_db_url() -> tuple[str, str, str, str, str]:
    """Get database connection details from environment."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_DB", "evalkit")
    return host, port, user, password, database


def download_release(tag: str, output_dir: Path) -> Path | None:
    """Download dump file from GitHub Release."""
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading from release {tag}...")
    result = subprocess.run(
        ["gh", "release", "download", tag, "-p", "*.dump.gz", "-D", str(output_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error downloading release: {result.stderr}", file=sys.stderr)
        return None

    # Find the downloaded file
    dumps = list(output_dir.glob("*.dump.gz"))
    if not dumps:
        print("Error: No .dump.gz file found in release", file=sys.stderr)
        return None

    return dumps[0]


def decompress_if_needed(dump_path: Path) -> Path:
    """Decompress .gz file if needed, return path to .dump file."""
    if dump_path.suffix == ".gz":
        print(f"Decompressing {dump_path.name}...")
        subprocess.run(["gunzip", "-kf", str(dump_path)], check=True)
        return dump_path.with_suffix("")  # Remove .gz
    return dump_path


def run_pg_restore(
    dump_path: Path,
    clean: bool = False,
    tables: list[str] | None = None,
) -> bool:
    """Run pg_restore to restore the database."""
    host, port, user, password, database = get_db_url()

    # Decompress if needed
    actual_dump = decompress_if_needed(dump_path)

    # Build pg_restore command
    cmd = [
        "pg_restore",
        "-h",
        host,
        "-p",
        port,
        "-U",
        user,
        "-d",
        database,
        "-v",  # Verbose
        "--no-owner",  # Don't set ownership
        "--no-acl",  # Don't restore access privileges
    ]

    if clean:
        cmd.append("--clean")  # Drop objects before recreating
        cmd.append("--if-exists")  # Don't error if objects don't exist

    # Add table filters if specified
    if tables:
        for table in tables:
            cmd.extend(["-t", table])

    cmd.append(str(actual_dump))

    # Set password via environment
    env = os.environ.copy()
    env["PGPASSWORD"] = password

    print(f"Restoring database from {actual_dump.name}...")
    if tables:
        print(f"Tables: {tables}")
    if clean:
        print("Mode: clean (drop and recreate)")

    try:
        # pg_restore returns non-zero for warnings too, so we don't use check=True
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        # Check for actual errors (not just warnings)
        if result.returncode != 0:
            # Filter out common warnings
            errors = [
                line
                for line in result.stderr.split("\n")
                if "ERROR" in line and "already exists" not in line
            ]
            if errors:
                print("Errors during restore:", file=sys.stderr)
                for error in errors:
                    print(f"  {error}", file=sys.stderr)
                return False

        print("Restore complete!")
        return True

    except FileNotFoundError:
        print("Error: pg_restore not found. Install PostgreSQL client tools.", file=sys.stderr)
        return False


def verify_restore() -> dict:
    """Verify restore by checking table row counts."""
    host, port, user, password, database = get_db_url()

    query = """
    SELECT
        schemaname || '.' || relname as table,
        n_live_tup as rows
    FROM pg_stat_user_tables
    WHERE n_live_tup > 0
    ORDER BY n_live_tup DESC;
    """

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    result = subprocess.run(
        ["psql", "-h", host, "-p", port, "-U", user, "-d", database, "-c", query],
        env=env,
        capture_output=True,
        text=True,
    )

    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Restore database from dump file")
    parser.add_argument(
        "dump_file",
        type=Path,
        nargs="?",
        help="Path to the .dump.gz file (or use --download)",
    )
    parser.add_argument(
        "--download",
        metavar="TAG",
        help="Download from GitHub Release tag first",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop and recreate tables before restore",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Restore only specific tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory for downloaded files (default: exports/)",
    )
    args = parser.parse_args()

    # Determine dump file path
    if args.download:
        dump_path = download_release(args.download, args.output_dir)
        if not dump_path:
            sys.exit(1)
    elif args.dump_file:
        dump_path = args.dump_file
        if not dump_path.exists():
            print(f"Error: File not found: {dump_path}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        print("\nError: Provide a dump file or use --download TAG", file=sys.stderr)
        sys.exit(1)

    # Run restore
    success = run_pg_restore(dump_path, clean=args.clean, tables=args.tables)

    if success:
        print()
        print("Verifying restore...")
        print(verify_restore())
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
