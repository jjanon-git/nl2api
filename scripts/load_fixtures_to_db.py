#!/usr/bin/env python3
"""
Load Generated Fixtures to Database

Loads test cases from generated JSON fixtures into the PostgreSQL database
for use with the batch evaluation system.

Usage:
    python scripts/load_fixtures_to_db.py --category entity_resolution
    python scripts/load_fixtures_to_db.py --all
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncpg  # noqa: E402

from CONTRACTS import TestCaseStatus  # noqa: E402

FIXTURE_DIR = Path("tests/fixtures/lseg/generated")


async def load_fixtures_to_db(
    pool: asyncpg.Pool,
    category: str,
    limit: int | None = None,
    clear_existing: bool = False,
) -> int:
    """
    Load fixtures from a category into the database.

    Args:
        pool: Database connection pool
        category: Category name (e.g., 'entity_resolution')
        limit: Maximum number of fixtures to load
        clear_existing: Whether to clear existing test cases for this category

    Returns:
        Number of fixtures loaded
    """
    fixture_path = FIXTURE_DIR / category / f"{category}.json"

    if not fixture_path.exists():
        print(f"  Fixture file not found: {fixture_path}")
        return 0

    with open(fixture_path) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    if limit:
        test_cases = test_cases[:limit]

    print(f"  Loading {len(test_cases)} test cases from {category}...")

    if clear_existing:
        async with pool.acquire() as conn:
            # Clear by tag since we don't have category column
            result = await conn.execute(
                "DELETE FROM test_cases WHERE $1 = ANY(tags)",
                category
            )
            print(f"  Cleared existing test cases: {result}")

    loaded = 0
    skipped_empty_tool_calls = 0
    async with pool.acquire() as conn:
        for tc in test_cases:
            # Skip test cases with empty expected_tool_calls (e.g., negative cases)
            # as they don't conform to TestCase contract (min_length=1)
            if not tc.get("expected_tool_calls"):
                skipped_empty_tool_calls += 1
                continue

            try:
                # Generate UUID for the test case
                test_uuid = uuid.uuid4()

                # Extract tags
                tags = tc.get("tags", [])
                # Add category and subcategory as tags for filtering
                category_val = tc.get("category", category)
                subcategory = tc.get("subcategory", "")
                if category_val and category_val not in tags:
                    tags = [category_val] + list(tags)
                if subcategory and subcategory not in tags:
                    tags.append(subcategory)

                # Build tool calls JSON
                expected_tool_calls = tc.get("expected_tool_calls", [])
                tool_calls_json = json.dumps(expected_tool_calls)

                # Prepare other fields - using existing schema fields
                # expected_response: Synthetic API response data (for semantics eval)
                # For entity_resolution, we also store metadata for the response generator
                metadata = tc.get("metadata", {})
                expected_response = tc.get("expected_response")

                # Merge metadata into expected_response to preserve input_entity etc.
                # This is critical for entity_resolution tests where input_entity
                # lives in metadata but is needed by the response generator.
                if expected_response is None:
                    expected_response = metadata
                elif metadata:
                    # Merge metadata into expected_response (metadata takes precedence
                    # for any duplicate keys since it has the evaluation-specific data)
                    expected_response = {**expected_response, **metadata}

                expected_response_json = (
                    json.dumps(expected_response) if expected_response else None
                )
                # expected_nl_response: NL summary (for semantics eval)
                expected_nl_response = tc.get("expected_nl_response") or ""
                complexity = tc.get("complexity", 1)

                # Insert into database using existing schema
                await conn.execute(
                    """
                    INSERT INTO test_cases (
                        id, nl_query, expected_tool_calls, expected_response,
                        expected_nl_response, api_version, complexity_level, tags,
                        status, source, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3::jsonb, $4::jsonb,
                        $5, $6, $7, $8,
                        $9, $10, NOW(), NOW()
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        nl_query = EXCLUDED.nl_query,
                        expected_tool_calls = EXCLUDED.expected_tool_calls,
                        expected_response = EXCLUDED.expected_response,
                        expected_nl_response = EXCLUDED.expected_nl_response,
                        complexity_level = EXCLUDED.complexity_level,
                        tags = EXCLUDED.tags,
                        updated_at = NOW()
                    """,
                    test_uuid,
                    tc.get("nl_query", ""),
                    tool_calls_json,
                    expected_response_json,
                    expected_nl_response,
                    "v1.0.0",  # api_version
                    min(complexity, 5),  # complexity_level (1-5)
                    tags,
                    TestCaseStatus.ACTIVE.value,
                    f"fixture:{category}",  # source
                )
                loaded += 1

            except Exception as e:
                print(f"  Error loading test case {tc.get('id')}: {e}")
                continue

    if skipped_empty_tool_calls > 0:
        print(f"  Skipped {skipped_empty_tool_calls} test cases with empty tool calls")

    return loaded


async def main():
    parser = argparse.ArgumentParser(description="Load fixtures to database")
    parser.add_argument(
        "--category",
        type=str,
        help="Category to load (e.g., entity_resolution)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all categories"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum fixtures per category"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing test cases for the category before loading"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://nl2api:nl2api@localhost:5432/nl2api"
        ),
        help="Database URL"
    )

    args = parser.parse_args()

    if not args.category and not args.all:
        parser.error("Must specify --category or --all")

    pool = await asyncpg.create_pool(args.db_url, min_size=2, max_size=5)

    try:
        if args.all:
            categories = [d.name for d in FIXTURE_DIR.iterdir() if d.is_dir()]
        else:
            categories = [args.category]

        total_loaded = 0
        for category in categories:
            print(f"\nProcessing category: {category}")
            loaded = await load_fixtures_to_db(
                pool, category, args.limit, args.clear
            )
            total_loaded += loaded
            print(f"  Loaded {loaded} test cases")

        print(f"\n{'='*50}")
        print(f"Total loaded: {total_loaded} test cases")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
