#!/usr/bin/env python3
"""
Load RAG Evaluation Fixtures to Database

Loads RAG test cases from the evaluation JSON file into PostgreSQL
for use with the batch evaluation system (--pack rag).

Usage:
    python scripts/load-rag-fixtures.py
    python scripts/load-rag-fixtures.py --limit 50
    python scripts/load-rag-fixtures.py --clear  # Clear existing RAG test cases first
    python scripts/load-rag-fixtures.py --validate  # Validate chunk IDs exist in DB

IMPORTANT: Only use verified fixture files (suffix: _verified.json).
Unverified fixtures may contain chunk IDs that don't exist in the database.
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
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Default fixture path
RAG_FIXTURE_PATH = Path("tests/fixtures/rag/sec_evaluation_set_verified.json")


def convert_to_rag_format(test_case: dict) -> dict | None:
    """
    Convert SEC evaluation test case format to RAGPack format.

    Input format (sec_evaluation_set.json):
        {
            "id": "simple_001",
            "query": "What was the total notional amount...",
            "category": "simple_factual",
            "company": "GS",
            "relevant_chunk_ids": ["uuid-1", "uuid-2"],
            "answer_keywords": ["$37,127,322", ...],
            "difficulty": "easy",
            "metadata": {...}
        }

    Alternative format (citation_required.json, etc.):
        {
            "id": "rag-cite-001",
            "input": {"query": "..."},
            "expected": {"behavior": "answer", "relevant_docs": [...]}
        }

    Adversarial format (adversarial_test_set.json):
        {
            "id": "adv_financial_001",
            "query": "Should I buy Apple stock?",
            "category": "financial_advice",
            "company": "AAPL",
            "expected": {"behavior": "reject"},
            "metadata": {...}
        }

    Output format (RAGPack TestCase):
        input_json: {"query": "..."}
        expected_json: {
            "relevant_docs": ["uuid-1", "uuid-2"],
            "behavior": "answer" | "reject",
            "answer_keywords": [...]
        }

    Returns None if test case format is invalid.
    """
    # Handle both formats: top-level "query" or nested "input.query"
    query = test_case.get("query") or test_case.get("input", {}).get("query")
    if not query:
        return None  # Skip invalid test cases

    # Include company context in input for retrieval enhancement
    input_json = {
        "query": query,
    }

    # Add company_name from metadata to input (for retrieval context)
    metadata = test_case.get("metadata", {})
    if metadata.get("company_name"):
        input_json["company_name"] = metadata["company_name"]
    elif test_case.get("company"):
        # Fallback: use company ticker if company_name not available
        input_json["company"] = test_case["company"]

    # Check for explicit expected block (alternative format has relevant_docs inside expected)
    expected_block = test_case.get("expected", {})
    behavior = expected_block.get("behavior", "answer")  # Default to "answer"

    # relevant_docs can be at top level (as relevant_chunk_ids) or in expected block
    relevant_docs = test_case.get("relevant_chunk_ids") or expected_block.get("relevant_docs") or []

    expected_json = {
        "relevant_docs": relevant_docs,
        "behavior": behavior,
    }

    # Include answer keywords for answer relevance evaluation
    if test_case.get("answer_keywords"):
        expected_json["answer_keywords"] = test_case["answer_keywords"]

    # Include requires_citations from expected block
    if expected_block.get("requires_citations"):
        expected_json["requires_citations"] = expected_block["requires_citations"]

    # Include expected section for context
    if test_case.get("expected_section"):
        expected_json["expected_section"] = test_case["expected_section"]

    return {
        "input_json": input_json,
        "expected_json": expected_json,
    }


async def load_rag_fixtures(
    pool: asyncpg.Pool,
    fixture_path: Path,
    limit: int | None = None,
    clear_existing: bool = False,
) -> int:
    """
    Load RAG fixtures into the database.

    Args:
        pool: Database connection pool
        fixture_path: Path to the RAG evaluation JSON file
        limit: Maximum number of test cases to load
        clear_existing: Whether to clear existing RAG test cases

    Returns:
        Number of fixtures loaded
    """
    if not fixture_path.exists():
        print(f"Error: Fixture file not found: {fixture_path}")
        return 0

    with open(fixture_path) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    meta = data.get("_meta", {})

    print(f"Fixture: {meta.get('name', 'unknown')}")
    print(f"Total test cases in file: {len(test_cases)}")

    if limit:
        test_cases = test_cases[:limit]
        print(f"Loading first {limit} test cases")

    if clear_existing:
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM test_cases WHERE 'rag' = ANY(tags)")
            print(f"Cleared existing RAG test cases: {result}")

    loaded = 0
    errors = 0

    async with pool.acquire() as conn:
        for tc in test_cases:
            try:
                # Generate UUID
                test_uuid = uuid.uuid4()

                # Convert to RAG format
                rag_format = convert_to_rag_format(tc)
                if rag_format is None:
                    errors += 1
                    continue

                # Build tags
                tags = ["rag", "sec_filing"]
                category = tc.get("category", "")
                if category:
                    tags.append(category)
                    # Add adversarial tag for rejection test cases
                    if category.startswith(("financial_", "investment_", "pii_", "out_of_scope")):
                        tags.append("adversarial")
                    if category == "edge_cases_should_answer":
                        tags.append("edge_case")
                difficulty = tc.get("difficulty", "")
                if difficulty:
                    tags.append(difficulty)
                company = tc.get("company", "")
                if company:
                    tags.append(f"company:{company}")
                # Add behavior tag
                expected_block = tc.get("expected", {})
                if expected_block.get("behavior") == "reject":
                    tags.append("should_reject")

                # Map difficulty to complexity level
                difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
                complexity = difficulty_map.get(difficulty, 2)

                # Store original metadata for reference
                metadata = tc.get("metadata", {})
                metadata["original_id"] = tc.get("id", "")
                metadata["company"] = company
                metadata["expected_section"] = tc.get("expected_section", "")

                # Build source_metadata for metrics tracking
                source_metadata_json = json.dumps(
                    {
                        "source_type": "synthetic",
                        "review_status": "approved",
                        "generator_name": "rag_eval_generator",
                    }
                )

                # Insert into database
                # Note: expected_tool_calls and api_version are required (NOT NULL)
                # For RAG tests: empty tool calls, use "rag-v1.0" as api_version
                await conn.execute(
                    """
                    INSERT INTO test_cases (
                        id, nl_query, expected_tool_calls, input_json, expected_json,
                        expected_response, api_version, complexity_level, tags,
                        status, source, source_type, source_metadata, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3::jsonb, $4::jsonb, $5::jsonb,
                        $6::jsonb, $7, $8, $9,
                        $10, $11, $12::data_source_type, $13::jsonb, NOW(), NOW()
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        nl_query = EXCLUDED.nl_query,
                        expected_tool_calls = EXCLUDED.expected_tool_calls,
                        input_json = EXCLUDED.input_json,
                        expected_json = EXCLUDED.expected_json,
                        expected_response = EXCLUDED.expected_response,
                        api_version = EXCLUDED.api_version,
                        complexity_level = EXCLUDED.complexity_level,
                        tags = EXCLUDED.tags,
                        source_type = EXCLUDED.source_type,
                        source_metadata = EXCLUDED.source_metadata,
                        updated_at = NOW()
                    """,
                    test_uuid,
                    rag_format["input_json"]["query"],  # nl_query for backwards compat
                    "[]",  # Empty tool calls for RAG tests
                    json.dumps(rag_format["input_json"]),
                    json.dumps(rag_format["expected_json"]),
                    json.dumps(metadata),  # Store metadata in expected_response
                    "v1.0.0",  # API version (semver format required)
                    complexity,
                    tags,
                    "active",
                    "generated:rag_eval",
                    "synthetic",  # source_type for metrics
                    source_metadata_json,
                )
                loaded += 1

            except Exception as e:
                print(f"  Error loading {tc.get('id', 'unknown')}: {e}")
                errors += 1

    print(f"\nLoaded: {loaded} test cases")
    if errors:
        print(f"Errors: {errors}")

    return loaded


async def validate_chunk_ids(pool: asyncpg.Pool, fixture_path: Path) -> dict:
    """
    Validate that chunk IDs in fixtures exist in the rag_documents table.

    Returns:
        Dict with validation results: total, valid, invalid, missing_ids
    """
    with open(fixture_path) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])

    # Collect all chunk IDs
    all_chunk_ids = set()
    for tc in test_cases:
        chunk_ids = tc.get("relevant_chunk_ids", [])
        all_chunk_ids.update(chunk_ids)

    print(f"Total unique chunk IDs in fixtures: {len(all_chunk_ids)}")

    # Check which exist in database
    async with pool.acquire() as conn:
        existing = await conn.fetch(
            """
            SELECT id::text FROM rag_documents
            WHERE id = ANY($1::uuid[])
            """,
            list(all_chunk_ids),
        )
        existing_ids = {str(row["id"]) for row in existing}

    missing_ids = all_chunk_ids - existing_ids
    valid_count = len(existing_ids)
    invalid_count = len(missing_ids)

    print(f"Chunks found in DB: {valid_count}")
    print(f"Chunks NOT in DB: {invalid_count}")

    if missing_ids:
        print("\nFirst 10 missing chunk IDs:")
        for chunk_id in list(missing_ids)[:10]:
            print(f"  - {chunk_id}")

    # Check test case coverage
    cases_with_valid_chunks = 0
    cases_with_no_valid_chunks = 0
    for tc in test_cases:
        chunk_ids = set(tc.get("relevant_chunk_ids", []))
        if chunk_ids & existing_ids:
            cases_with_valid_chunks += 1
        else:
            cases_with_no_valid_chunks += 1

    print(f"\nTest cases with at least one valid chunk: {cases_with_valid_chunks}")
    print(f"Test cases with NO valid chunks: {cases_with_no_valid_chunks}")

    return {
        "total_chunks": len(all_chunk_ids),
        "valid_chunks": valid_count,
        "invalid_chunks": invalid_count,
        "missing_ids": list(missing_ids)[:50],
        "cases_with_valid_chunks": cases_with_valid_chunks,
        "cases_with_no_valid_chunks": cases_with_no_valid_chunks,
    }


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    postgres_url = os.getenv(
        "NL2API_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
    )

    pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)

    try:
        fixture_path = Path(args.fixture)

        # Warn if using unverified fixture
        if "_verified" not in fixture_path.name:
            print("⚠️  WARNING: Loading unverified fixture file!")
            print("   Only use *_verified.json files for evaluation.")
            print("   Run with --validate to check chunk ID validity.\n")

        # Validate mode - check chunk IDs exist
        if args.validate:
            print("Validating chunk IDs in fixtures...\n")
            results = await validate_chunk_ids(pool, fixture_path)
            if results["invalid_chunks"] > 0:
                print(f"\n❌ Validation FAILED: {results['invalid_chunks']} chunks not in database")
                sys.exit(1)
            else:
                print("\n✅ Validation PASSED: All chunks exist in database")
            return

        await load_rag_fixtures(
            pool,
            fixture_path,
            limit=args.limit,
            clear_existing=args.clear,
        )

        # Show verification
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM test_cases WHERE 'rag' = ANY(tags)")
            print(f"\nTotal RAG test cases in database: {count}")

            # Show sample
            sample = await conn.fetchrow(
                """
                SELECT id, nl_query, input_json, expected_json, tags
                FROM test_cases
                WHERE 'rag' = ANY(tags)
                LIMIT 1
                """
            )
            if sample:
                print("\nSample test case:")
                print(f"  ID: {sample['id']}")
                print(f"  Query: {sample['nl_query'][:80]}...")
                print(f"  Input: {sample['input_json']}")
                print(f"  Expected: {json.dumps(sample['expected_json'])[:100]}...")
                print(f"  Tags: {sample['tags']}")

    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load RAG evaluation fixtures")
    parser.add_argument(
        "--fixture",
        type=str,
        default=str(RAG_FIXTURE_PATH),
        help=f"Path to RAG fixture file (default: {RAG_FIXTURE_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of test cases to load",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing RAG test cases before loading",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate chunk IDs exist in database (don't load)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
