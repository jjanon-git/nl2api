#!/usr/bin/env python3
"""
Update RAG Fixture Chunk IDs

After re-ingesting documents, chunk UUIDs change. This script updates
fixture files with current chunk IDs by matching on content.

Usage:
    python scripts/update-rag-fixture-chunk-ids.py --fixture tests/fixtures/rag/sec_evaluation_set_verified.json
    python scripts/update-rag-fixture-chunk-ids.py --fixture tests/fixtures/rag/sec_evaluation_set_verified.json --dry-run

Process:
1. Load fixture file with stale chunk IDs
2. For each test case with relevant_chunk_ids:
   a. Use the query to retrieve current chunks
   b. If retrieval finds relevant content, update the chunk IDs
   c. Track cases where no matching chunks found
3. Save updated fixture (or print changes in dry-run mode)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def get_chunks_for_query(
    pool: asyncpg.Pool,
    query: str,
    company: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    Retrieve current chunks that match a query.

    Uses the same retrieval logic as the RAG pipeline.
    """
    async with pool.acquire() as conn:
        # Simple text search for now - could use vector search if embeddings available
        sql = """
            SELECT id, content, metadata
            FROM rag_documents
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
        """
        params = [query]

        if company:
            sql += " AND metadata->>'company' = $2"
            params.append(company)

        sql += f" LIMIT {limit}"

        rows = await conn.fetch(sql, *params)
        return [
            {
                "id": str(row["id"]),
                "content": row["content"][:500],
                "metadata": row["metadata"],
            }
            for row in rows
        ]


async def update_fixture_chunk_ids(
    pool: asyncpg.Pool,
    fixture_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Update chunk IDs in fixture file based on current database state.
    """
    with open(fixture_path) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])

    stats = {
        "total": len(test_cases),
        "updated": 0,
        "no_match": 0,
        "no_chunk_ids": 0,
        "errors": 0,
    }

    for tc in test_cases:
        old_chunk_ids = tc.get("relevant_chunk_ids", [])

        if not old_chunk_ids:
            stats["no_chunk_ids"] += 1
            continue

        query = tc.get("query", "")
        company = tc.get("company")

        try:
            # Get current chunks for this query
            current_chunks = await get_chunks_for_query(pool, query, company, limit=5)

            if current_chunks:
                new_chunk_ids = [c["id"] for c in current_chunks]

                if set(new_chunk_ids) != set(old_chunk_ids):
                    if dry_run:
                        print(f"\n[{tc.get('id', 'unknown')}] {query[:60]}...")
                        print(f"  Old: {old_chunk_ids[:3]}...")
                        print(f"  New: {new_chunk_ids[:3]}...")
                    else:
                        tc["relevant_chunk_ids"] = new_chunk_ids

                    stats["updated"] += 1

                # Update metadata
                tc.setdefault("metadata", {})
                tc["metadata"]["source_chunk_retrievable"] = True
                tc["metadata"]["chunk_ids_updated_at"] = "auto"
            else:
                stats["no_match"] += 1
                if dry_run:
                    print(f"\n[{tc.get('id', 'unknown')}] No matching chunks found")
                    print(f"  Query: {query[:60]}...")

        except Exception as e:
            stats["errors"] += 1
            print(f"Error processing {tc.get('id', 'unknown')}: {e}")

    # Save updated fixture
    if not dry_run and stats["updated"] > 0:
        # Update metadata
        from datetime import datetime

        data["_meta"]["chunk_ids_updated_at"] = datetime.now().isoformat()
        data["_meta"]["source_chunk_retrievable"] = sum(
            1 for tc in test_cases if tc.get("metadata", {}).get("source_chunk_retrievable")
        )

        output_path = fixture_path.with_suffix(".updated.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved updated fixture to: {output_path}")

    return stats


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    postgres_url = os.getenv(
        "NL2API_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
    )

    pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)

    try:
        fixture_path = Path(args.fixture)

        if not fixture_path.exists():
            print(f"Fixture file not found: {fixture_path}")
            sys.exit(1)

        print(f"Updating chunk IDs in: {fixture_path}")
        if args.dry_run:
            print("(DRY RUN - no changes will be saved)\n")

        stats = await update_fixture_chunk_ids(
            pool,
            fixture_path,
            dry_run=args.dry_run,
        )

        print("\n" + "=" * 50)
        print("Summary:")
        print(f"  Total test cases: {stats['total']}")
        print(f"  Updated: {stats['updated']}")
        print(f"  No matching chunks: {stats['no_match']}")
        print(f"  No chunk IDs (skipped): {stats['no_chunk_ids']}")
        print(f"  Errors: {stats['errors']}")

    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update RAG fixture chunk IDs")
    parser.add_argument(
        "--fixture",
        type=str,
        required=True,
        help="Path to fixture file to update",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without saving",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
