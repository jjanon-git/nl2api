#!/usr/bin/env python3
"""
Load Test Cases Script

Loads test cases from JSON files into the storage backend.

Usage:
    # Load from a directory of JSON files
    python -m scripts.load_test_cases tests/fixtures/lseg/generated/

    # Load a single file
    python -m scripts.load_test_cases tests/fixtures/search_products.json

    # Dry run (validate without saving)
    python -m scripts.load_test_cases --dry-run tests/fixtures/

    # Use specific backend
    EVAL_BACKEND=memory python -m scripts.load_test_cases tests/fixtures/
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from CONTRACTS import TestCase, TestCaseMetadata, ToolCall
from src.storage import StorageConfig, create_repositories, close_repositories

console = Console()


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError):
        return False


def parse_test_case(data: dict[str, Any], source_file: str) -> TestCase:
    """Parse a test case from JSON data."""
    # Handle expected_tool_calls
    tool_calls = []
    for tc in data.get("expected_tool_calls", []):
        tool_calls.append(ToolCall(
            tool_name=tc.get("tool_name") or tc.get("name"),
            arguments=tc.get("arguments", {}),
        ))

    # Handle metadata
    metadata_data = data.get("metadata", {})

    # Preserve original ID in tags if it's not a valid UUID
    original_id = data.get("id")
    tags = list(metadata_data.get("tags", []))
    if original_id and not is_valid_uuid(original_id):
        tags.append(f"original_id:{original_id}")

    metadata = TestCaseMetadata(
        api_version=metadata_data.get("api_version", "v1.0.0"),
        complexity_level=metadata_data.get("complexity_level", 1),
        tags=tuple(tags),
        author=metadata_data.get("author"),
        source=source_file,
    )

    # Build test case kwargs - only include id if it's a valid UUID
    kwargs: dict[str, Any] = {
        "nl_query": data["nl_query"],
        "expected_tool_calls": tuple(tool_calls),
        "expected_raw_data": data.get("expected_raw_data"),
        "expected_nl_response": data.get("expected_nl_response", ""),
        "metadata": metadata,
    }

    # Only include ID if it's a valid UUID, otherwise let it auto-generate
    if original_id and is_valid_uuid(original_id):
        kwargs["id"] = original_id

    return TestCase(**kwargs)


def find_test_files(path: Path) -> list[Path]:
    """Find all JSON test case files in a path."""
    if path.is_file():
        return [path] if path.suffix == ".json" else []

    # Find all JSON files recursively
    return list(path.rglob("*.json"))


async def load_test_cases(
    paths: list[Path],
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Load test cases from files into the storage backend.

    Args:
        paths: List of file or directory paths
        dry_run: If True, validate only without saving
        verbose: Show detailed output

    Returns:
        Tuple of (loaded_count, skipped_count, errors)
    """
    # Find all test files
    all_files: list[Path] = []
    for path in paths:
        all_files.extend(find_test_files(path))

    if not all_files:
        console.print("[yellow]No JSON files found[/yellow]")
        return 0, 0, []

    console.print(f"Found [cyan]{len(all_files)}[/cyan] JSON files")

    # Initialize storage (unless dry run)
    test_case_repo = None
    if not dry_run:
        config = StorageConfig()
        console.print(f"Using backend: [cyan]{config.backend}[/cyan]")
        test_case_repo, _ = await create_repositories(config)

    loaded = 0
    skipped = 0
    errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading test cases...", total=len(all_files))

        for file_path in all_files:
            progress.update(task, description=f"Processing {file_path.name}...")

            try:
                data = json.loads(file_path.read_text())

                # Handle both single test case and array of test cases
                test_cases_data = data if isinstance(data, list) else [data]

                for tc_data in test_cases_data:
                    # Skip if it's clearly not a test case
                    if "nl_query" not in tc_data:
                        if verbose:
                            console.print(f"  [dim]Skipping {file_path.name}: no nl_query[/dim]")
                        skipped += 1
                        continue

                    test_case = parse_test_case(tc_data, str(file_path))

                    if dry_run:
                        if verbose:
                            console.print(f"  [green]Valid:[/green] {test_case.id} - {test_case.nl_query[:50]}...")
                        loaded += 1
                    else:
                        await test_case_repo.save(test_case)
                        if verbose:
                            console.print(f"  [green]Saved:[/green] {test_case.id}")
                        loaded += 1

            except json.JSONDecodeError as e:
                errors.append(f"{file_path}: Invalid JSON - {e}")
            except KeyError as e:
                errors.append(f"{file_path}: Missing required field - {e}")
            except Exception as e:
                errors.append(f"{file_path}: {type(e).__name__} - {e}")

            progress.advance(task)

    # Cleanup
    if not dry_run:
        await close_repositories()

    return loaded, skipped, errors


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load test cases into the storage backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories containing test cases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without saving to database",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # Validate paths exist
    for path in args.paths:
        if not path.exists():
            console.print(f"[red]Error:[/red] Path not found: {path}")
            sys.exit(1)

    # Run loader
    loaded, skipped, errors = asyncio.run(
        load_test_cases(args.paths, args.dry_run, args.verbose)
    )

    # Summary
    console.print()
    table = Table(title="Load Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Loaded", f"[green]{loaded}[/green]")
    table.add_row("Skipped", f"[yellow]{skipped}[/yellow]")
    table.add_row("Errors", f"[red]{len(errors)}[/red]")

    console.print(table)

    if errors:
        console.print()
        console.print("[red]Errors:[/red]")
        for error in errors[:10]:  # Show first 10 errors
            console.print(f"  {error}")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
