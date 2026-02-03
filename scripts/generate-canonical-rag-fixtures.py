#!/usr/bin/env python3
"""
Generate Canonical RAG Test Fixtures

Creates test cases with verified ground truth by generating questions
directly FROM documents in the current database. Since questions are
derived from specific documents, the document IDs are guaranteed correct.

Usage:
    # Generate 50 canonical test cases (uses gpt-5-nano by default)
    python scripts/generate-canonical-rag-fixtures.py --count 50

    # Preview without saving
    python scripts/generate-canonical-rag-fixtures.py --count 10 --dry-run

    # Generate for specific companies
    python scripts/generate-canonical-rag-fixtures.py --companies AAPL,MSFT,GOOGL --count 30

    # Use a different model
    python scripts/generate-canonical-rag-fixtures.py --count 50 --model gpt-4o-mini

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

CRITICAL Requirements for unique retrievability:
- Questions MUST include the company name (e.g., "What was Apple Inc's total revenue...")
- Questions MUST reference specific numbers, dollar amounts, percentages, or dates from the text
- Questions should ask about concrete facts that are UNIQUE to this document
- The combination of company name + specific metric + time period should uniquely identify this content
- Avoid generic questions that could apply to multiple documents

Good example: "What was Tesla Inc's net income in Q3 2024 according to their 10-Q filing?"
Bad example: "What was the company's revenue?"

Include the type of question: simple_factual, temporal_comparative, or complex_analytical

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
      "question": "What was [COMPANY]'s total revenue for fiscal year 2024?",
      "category": "simple_factual",
      "answer_keywords": ["$1.2 billion", "revenue", "2024"],
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
            conditions.append(f"metadata->>'ticker' = ANY(${param_idx}::text[])")
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
            metadata = row["metadata"]
            # Handle both dict and string (asyncpg may return jsonb as string)
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}
            metadata = metadata or {}
            company = metadata.get("ticker") or metadata.get("company", "unknown")

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
    model: str = "gpt-5-nano",
) -> list[dict]:
    """
    Generate questions for a document using LLM.
    """
    from src.evalkit.common.llm import create_openai_client, openai_chat_completion

    client = create_openai_client(async_client=True)

    metadata = doc["metadata"]
    content = doc["content"][:3000]  # Limit content length

    company = metadata.get("company_name") or metadata.get("ticker") or "Unknown"
    filing_type = metadata.get("filing_type") or metadata.get("form_type") or "10-K"
    prompt = QUESTION_GEN_PROMPT.format(
        num_questions=num_questions,
        company=company,
        filing_type=filing_type,
        section=metadata.get("section", "Unknown"),
        content=content,
    )

    try:
        response = await openai_chat_completion(
            client,
            model=model,
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
        # Handle both {"questions": [...]} and direct list [...] formats
        if isinstance(result, list):
            return result
        return result.get("questions", [])

    except Exception as e:
        print(f"  Error generating questions: {e}")
        return []


async def verify_retrieval_with_hybrid(
    retriever,
    query: str,
    expected_doc_id: str,
    ticker: str | None = None,
) -> tuple[bool, float, int]:
    """
    Verify retrieval using the actual hybrid retriever (vector + keyword).

    This ensures fixtures are verified with the same retrieval logic used in evaluation.

    Returns (found, score, position).
    """
    try:
        results = await retriever.retrieve(
            query=query,
            document_types=None,
            limit=10,
            threshold=0.0,
            use_cache=False,
            ticker=ticker,
        )

        for i, result in enumerate(results):
            if str(result.id) == expected_doc_id:
                # Found in top 10, return position-based score
                return True, 1.0 - (i * 0.1), i + 1

        return False, 0.0, -1

    except Exception as e:
        print(f"    Retrieval error: {e}")
        return False, 0.0, -1


async def generate_canonical_fixtures(
    pool: asyncpg.Pool,
    count: int = 50,
    companies: list[str] | None = None,
    dry_run: bool = False,
    model: str = "gpt-5-nano",
    include_unverified: bool = False,
    concurrency: int = 20,
    restart: bool = False,
) -> dict:
    """
    Generate canonical test fixtures with verified ground truth.

    Verification is MANDATORY - uses the actual hybrid retriever to ensure
    questions are retrievable. Only verified questions are included by default.

    Args:
        pool: Database connection pool
        count: Target number of test cases (will sample more to account for failures)
        companies: Optional list of tickers to filter by
        dry_run: If True, don't save to file
        model: LLM model for question generation
        include_unverified: If True, include questions that failed verification
                           (NOT RECOMMENDED - will cause evaluation failures)
        concurrency: Number of documents to process concurrently (default 20)
        restart: If True, start fresh and delete existing fixtures
    """
    from src.rag.retriever.embedders import OpenAIEmbedder
    from src.rag.retriever.retriever import HybridRAGRetriever

    output_path = PROJECT_ROOT / "tests" / "fixtures" / "rag" / "canonical_retrieval_set.json"

    # Handle resume vs restart
    existing_cases = []
    used_doc_ids: set[str] = set()

    if restart:
        if output_path.exists():
            output_path.unlink()
            print("Deleted existing fixture file (--restart specified)")
    elif output_path.exists():
        # Resume from existing file
        with open(output_path) as f:
            existing_data = json.load(f)
        existing_cases = [
            c
            for c in existing_data.get("test_cases", [])
            if c.get("metadata", {}).get("retrieval_verified")
        ]
        used_doc_ids = {
            c.get("metadata", {}).get("source_doc_id")
            for c in existing_cases
            if c.get("metadata", {}).get("source_doc_id")
        }
        print(f"Resuming: found {len(existing_cases)} existing verified fixtures")
        print(f"  Excluding {len(used_doc_ids)} already-used document IDs")

        if len(existing_cases) >= count:
            print(f"  Already have {len(existing_cases)} >= {count} target. Nothing to do.")
            return {
                "total_docs": 0,
                "questions_generated": 0,
                "retrieval_verified": len(existing_cases),
                "retrieval_failed": 0,
                "errors": 0,
                "position_distribution": {},
            }

    # Initialize the hybrid retriever (same as used in evaluation)
    print("Initializing hybrid retriever for verification...")
    api_key = os.getenv("NL2API_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("NL2API_OPENAI_API_KEY required for retrieval verification")

    embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
    retriever = HybridRAGRetriever(
        pool=pool,
        embedding_dimension=1536,
        vector_weight=0.7,
        keyword_weight=0.3,
    )
    retriever.set_embedder(embedder)
    print("  Retriever initialized.\n")

    # Calculate remaining needed
    remaining_needed = count - len(existing_cases)

    # Sample more documents than needed to account for verification failures
    # ~10% verification rate observed, so need ~12x buffer
    sample_count = int(remaining_needed * 12)  # 12x buffer for ~10% verification rate
    print(f"Sampling {sample_count} documents (target: {remaining_needed} more verified)...")
    documents = await get_sample_documents(pool, sample_count, companies)

    # Filter out already-used documents
    if used_doc_ids:
        documents = [d for d in documents if d["id"] not in used_doc_ids]
        print(f"  After filtering used docs: {len(documents)} documents")
    else:
        print(f"  Found {len(documents)} documents")

    test_cases = []
    verified_cases = list(existing_cases)  # Start with existing
    stats = {
        "total_docs": len(documents),
        "questions_generated": 0,
        "retrieval_verified": len(existing_cases),
        "retrieval_failed": 0,
        "errors": 0,
        "position_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "6+": 0},
    }

    # Process documents in parallel batches for speed
    batch_size = concurrency  # Number of documents to process concurrently

    async def process_single_doc(doc: dict, doc_idx: int) -> dict | None:
        """Process a single document: generate question and verify retrieval."""
        try:
            questions = await generate_questions_for_document(doc, num_questions=1, model=model)
            if not questions:
                return {"error": True, "doc_id": doc["id"]}

            q = questions[0]
            metadata = doc["metadata"]
            ticker = metadata.get("ticker") or metadata.get("company")

            # Verify with hybrid retriever
            found, score, position = await verify_retrieval_with_hybrid(
                retriever,
                q["question"],
                doc["id"],
                ticker,
            )

            return {
                "error": False,
                "doc": doc,
                "question": q,
                "ticker": ticker,
                "found": found,
                "score": score,
                "position": position,
            }
        except Exception as e:
            return {"error": True, "doc_id": doc["id"], "exception": str(e)}

    doc_idx = 0
    while doc_idx < len(documents) and len(verified_cases) < count:
        # Get next batch
        batch_end = min(doc_idx + batch_size, len(documents))
        batch = documents[doc_idx:batch_end]

        print(
            f"\n[Batch {doc_idx + 1}-{batch_end}/{len(documents)}] Processing {len(batch)} docs in parallel..."
        )

        # Process batch in parallel
        results = await asyncio.gather(*[process_single_doc(doc, i) for i, doc in enumerate(batch)])

        # Process results
        for result in results:
            if result is None or result.get("error"):
                stats["errors"] += 1
                continue

            stats["questions_generated"] += 1
            q = result["question"]
            doc = result["doc"]
            metadata = doc["metadata"]

            test_case = {
                "id": f"canonical_{len(test_cases) + 1:03d}",
                "query": q["question"],
                "category": q.get("category", "simple_factual"),
                "company": result["ticker"],
                "relevant_chunk_ids": [doc["id"]],
                "answer_keywords": q.get("answer_keywords", []),
                "difficulty": q.get("difficulty", "medium"),
                "metadata": {
                    "company_name": metadata.get("company_name"),
                    "filing_type": metadata.get("filing_type") or metadata.get("form_type"),
                    "section": metadata.get("section"),
                    "source_doc_id": doc["id"],
                    "generated_at": datetime.now().isoformat(),
                    "generator": "canonical_fixture_generator",
                    "generator_model": model,
                    "retrieval_verified": result["found"],
                    "retrieval_score": result["score"],
                    "retrieval_position": result["position"],
                },
            }

            if result["found"]:
                stats["retrieval_verified"] += 1
                pos_key = result["position"] if result["position"] <= 5 else "6+"
                stats["position_distribution"][pos_key] += 1
                print(f"  ✓ [pos={result['position']}] {q['question'][:55]}...")
                verified_cases.append(test_case)
            else:
                stats["retrieval_failed"] += 1

            test_cases.append(test_case)

        doc_idx = batch_end
        print(
            f"  Progress: {len(verified_cases)}/{count} verified ({stats['retrieval_verified']}/{stats['questions_generated']} = {stats['retrieval_verified'] * 100 // max(1, stats['questions_generated'])}%)"
        )

    # Only include verified cases (unless explicitly including unverified)
    output_cases = test_cases if include_unverified else verified_cases

    # Build output
    output = {
        "_meta": {
            "name": "canonical_retrieval_evaluation",
            "capability": "rag_retrieval",
            "description": "Canonical test cases with VERIFIED ground truth for retrieval evaluation",
            "methodology": "Questions generated from documents, verified retrievable with hybrid retriever",
            "generated_at": datetime.now().isoformat(),
            "total_cases": len(output_cases),
            "retrieval_verified": stats["retrieval_verified"],
            "retrieval_failed": stats["retrieval_failed"],
            "verification_rate": f"{stats['retrieval_verified'] / max(1, stats['questions_generated']) * 100:.1f}%",
            "position_distribution": stats["position_distribution"],
            "schema_version": "2.2",
            "generator_model": model,
        },
        "test_cases": output_cases,
    }

    # Save if not dry run
    if not dry_run:
        output_path = PROJECT_ROOT / "tests" / "fixtures" / "rag" / "canonical_retrieval_set.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {len(output_cases)} verified fixtures to: {output_path}")

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
        print("Generating Canonical RAG Fixtures (with retrieval verification)")
        print("=" * 60)
        print(f"  Target count: {args.count} verified fixtures")
        print(f"  Model: {args.model}")
        print(f"  Concurrency: {args.concurrency} docs/batch")
        print(f"  Companies filter: {companies or 'all'}")
        print(f"  Dry run: {args.dry_run}")
        print(f"  Restart: {args.restart}")
        print()

        stats = await generate_canonical_fixtures(
            pool,
            count=args.count,
            companies=companies,
            dry_run=args.dry_run,
            model=args.model,
            concurrency=args.concurrency,
            restart=args.restart,
        )

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Documents sampled: {stats['total_docs']}")
        print(f"  Questions generated: {stats['questions_generated']}")
        print(f"  Retrieval verified: {stats['retrieval_verified']}")
        print(f"  Retrieval failed: {stats['retrieval_failed']}")
        print(f"  Errors: {stats['errors']}")

        if stats["questions_generated"] > 0:
            pct = stats["retrieval_verified"] / stats["questions_generated"] * 100
            print(f"  Verification rate: {pct:.1f}%")

        if stats["retrieval_verified"] > 0:
            print("\n  Position distribution (where in top-10 the chunk was found):")
            for pos, count in stats["position_distribution"].items():
                if count > 0:
                    print(f"    Position {pos}: {count}")

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
        "--dry-run",
        action="store_true",
        help="Preview without saving fixtures",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="OpenAI model for question generation (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent API calls (default: 20, reduce if hitting rate limits)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start fresh, deleting existing fixtures (default: resume from existing)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
