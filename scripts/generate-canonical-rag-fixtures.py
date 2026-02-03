#!/usr/bin/env python3
"""
Generate Canonical RAG Test Fixtures

Creates test cases with verified ground truth by generating questions
directly FROM documents in the current database. Since questions are
derived from specific documents, the document IDs are guaranteed correct.

Usage:
    # Generate 50 canonical test cases
    python scripts/generate-canonical-rag-fixtures.py --count 50

    # Preview without saving
    python scripts/generate-canonical-rag-fixtures.py --count 10 --dry-run

    # Generate for specific companies
    python scripts/generate-canonical-rag-fixtures.py --companies AAPL,MSFT,GOOGL --count 30

Output:
    tests/fixtures/rag/canonical_retrieval_set.json

Process:
1. Sample documents from rag_documents table
2. For each document, generate 1-2 questions using LLM
3. Verify retrieval can find the source document
4. Save as fixtures with document IDs as ground truth
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Question generation prompt
QUESTION_GEN_PROMPT = """You are generating test questions for a RAG system that answers questions about SEC filings.

Given this excerpt from an SEC filing, generate {num_questions} specific factual question(s) that can ONLY be answered using information in this excerpt.

Requirements:
- Questions must be answerable from THIS specific text
- Questions should ask about concrete facts, numbers, dates, or named entities
- Avoid vague questions like "What does the company do?"
- Include the type of question: simple_factual, temporal_comparative, or complex_analytical

SEC Filing Excerpt:
Company: {company}
Filing Type: {filing_type}
Section: {section}

Content:
{content}

Respond in JSON format:
{{
  "questions": [
    {{
      "question": "What was the total revenue in Q3 2023?",
      "category": "simple_factual",
      "answer_keywords": ["$1.2 billion", "revenue", "Q3"],
      "difficulty": "easy"
    }}
  ]
}}
"""


async def get_sample_documents(
    pool: asyncpg.Pool,
    count: int = 50,
    companies: list[str] | None = None,
) -> list[dict]:
    """
    Sample documents from rag_documents table.

    Prioritizes documents with:
    - Sufficient content length
    - Diverse companies
    - Different filing sections
    """
    async with pool.acquire() as conn:
        # Build query
        conditions = ["LENGTH(content) > 500"]
        params: list[Any] = []
        param_idx = 1

        if companies:
            conditions.append(f"metadata->>'company' = ANY(${param_idx}::text[])")
            params.append(companies)
            param_idx += 1

        # Sample with diversity - get more than needed, then dedupe
        query = f"""
            SELECT
                id,
                content,
                metadata
            FROM rag_documents
            WHERE {" AND ".join(conditions)}
            ORDER BY RANDOM()
            LIMIT {count * 3}
        """

        rows = await conn.fetch(query, *params)

        # Dedupe by company to ensure diversity
        seen_companies: dict[str, int] = {}
        max_per_company = max(3, count // 10)  # At least 3, or 10% of total

        documents = []
        for row in rows:
            metadata = row["metadata"] or {}
            company = metadata.get("company", "unknown")

            if seen_companies.get(company, 0) >= max_per_company:
                continue

            seen_companies[company] = seen_companies.get(company, 0) + 1
            documents.append(
                {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "metadata": metadata,
                }
            )

            if len(documents) >= count:
                break

        return documents


async def generate_questions_for_document(
    doc: dict,
    num_questions: int = 1,
) -> list[dict]:
    """
    Generate questions for a document using LLM.
    """
    from src.evalkit.common.llm import create_openai_client, openai_chat_completion

    client = create_openai_client(async_client=True)

    metadata = doc["metadata"]
    content = doc["content"][:3000]  # Limit content length

    prompt = QUESTION_GEN_PROMPT.format(
        num_questions=num_questions,
        company=metadata.get("company", "Unknown"),
        filing_type=metadata.get("form_type", "10-K"),
        section=metadata.get("section", "Unknown"),
        content=content,
    )

    try:
        response = await openai_chat_completion(
            client,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )

        result_text = response.choices[0].message.content

        # Parse JSON from response
        # Handle potential markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())
        return result.get("questions", [])

    except Exception as e:
        print(f"  Error generating questions: {e}")
        return []


async def verify_retrieval(
    pool: asyncpg.Pool,
    query: str,
    expected_doc_id: str,
    company: str | None = None,
) -> tuple[bool, float]:
    """
    Verify that retrieval can find the expected document.

    Returns (found, score).
    """
    # Simple text search verification
    async with pool.acquire() as conn:
        sql = """
            SELECT id,
                   ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
            FROM rag_documents
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
        """
        params = [query]

        if company:
            sql += " AND metadata->>'company' = $2"
            params.append(company)

        sql += " ORDER BY rank DESC LIMIT 10"

        rows = await conn.fetch(sql, *params)

        for i, row in enumerate(rows):
            if str(row["id"]) == expected_doc_id:
                # Found in top 10, return position-based score
                return True, 1.0 - (i * 0.1)

        return False, 0.0


async def generate_canonical_fixtures(
    pool: asyncpg.Pool,
    count: int = 50,
    companies: list[str] | None = None,
    verify: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Generate canonical test fixtures with verified ground truth.
    """
    print(f"Sampling {count} documents...")
    documents = await get_sample_documents(pool, count, companies)
    print(f"  Found {len(documents)} documents")

    test_cases = []
    stats = {
        "total_docs": len(documents),
        "questions_generated": 0,
        "retrieval_verified": 0,
        "retrieval_failed": 0,
        "errors": 0,
    }

    for i, doc in enumerate(documents):
        print(f"\n[{i + 1}/{len(documents)}] Processing doc {doc['id'][:8]}...")

        # Generate questions
        questions = await generate_questions_for_document(doc, num_questions=1)

        if not questions:
            stats["errors"] += 1
            continue

        for q in questions:
            stats["questions_generated"] += 1

            # Build test case
            test_case = {
                "id": f"canonical_{len(test_cases) + 1:03d}",
                "query": q["question"],
                "category": q.get("category", "simple_factual"),
                "company": doc["metadata"].get("company"),
                "relevant_chunk_ids": [doc["id"]],
                "answer_keywords": q.get("answer_keywords", []),
                "difficulty": q.get("difficulty", "medium"),
                "metadata": {
                    "company_name": doc["metadata"].get("company_name"),
                    "filing_type": doc["metadata"].get("form_type"),
                    "section": doc["metadata"].get("section"),
                    "source_doc_id": doc["id"],
                    "generated_at": datetime.now().isoformat(),
                    "generator": "canonical_fixture_generator",
                },
            }

            # Verify retrieval if requested
            if verify:
                found, score = await verify_retrieval(
                    pool,
                    q["question"],
                    doc["id"],
                    doc["metadata"].get("company"),
                )
                test_case["metadata"]["retrieval_verified"] = found
                test_case["metadata"]["retrieval_score"] = score

                if found:
                    stats["retrieval_verified"] += 1
                    print(f"  ✓ Question: {q['question'][:60]}...")
                else:
                    stats["retrieval_failed"] += 1
                    print(f"  ✗ Retrieval failed: {q['question'][:60]}...")
            else:
                print(f"  ? Question: {q['question'][:60]}...")

            test_cases.append(test_case)

    # Build output
    output = {
        "_meta": {
            "name": "canonical_retrieval_evaluation",
            "capability": "rag_retrieval",
            "description": "Canonical test cases with verified ground truth for retrieval evaluation",
            "methodology": "Questions generated from documents, retrieval verified",
            "generated_at": datetime.now().isoformat(),
            "total_cases": len(test_cases),
            "retrieval_verified": stats["retrieval_verified"],
            "schema_version": "2.1",
        },
        "test_cases": test_cases,
    }

    # Save if not dry run
    if not dry_run:
        output_path = PROJECT_ROOT / "tests" / "fixtures" / "rag" / "canonical_retrieval_set.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {output_path}")

    return stats


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    postgres_url = os.getenv(
        "NL2API_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
    )

    pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)

    try:
        companies = args.companies.split(",") if args.companies else None

        print("=" * 60)
        print("Generating Canonical RAG Fixtures")
        print("=" * 60)
        print(f"  Target count: {args.count}")
        print(f"  Companies filter: {companies or 'all'}")
        print(f"  Verify retrieval: {not args.skip_verify}")
        print(f"  Dry run: {args.dry_run}")
        print()

        stats = await generate_canonical_fixtures(
            pool,
            count=args.count,
            companies=companies,
            verify=not args.skip_verify,
            dry_run=args.dry_run,
        )

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Documents sampled: {stats['total_docs']}")
        print(f"  Questions generated: {stats['questions_generated']}")
        print(f"  Retrieval verified: {stats['retrieval_verified']}")
        print(f"  Retrieval failed: {stats['retrieval_failed']}")
        print(f"  Errors: {stats['errors']}")

        if stats["retrieval_verified"] > 0:
            pct = stats["retrieval_verified"] / stats["questions_generated"] * 100
            print(f"  Verification rate: {pct:.1f}%")

    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate canonical RAG fixtures")
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of test cases to generate (default: 50)",
    )
    parser.add_argument(
        "--companies",
        type=str,
        default=None,
        help="Comma-separated list of company tickers to filter by",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip retrieval verification step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without saving fixtures",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
