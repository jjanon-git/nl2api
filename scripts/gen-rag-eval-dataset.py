#!/usr/bin/env python3
"""
Generate RAG evaluation dataset from SEC filings.

CORRECT METHODOLOGY:
1. Sample chunks with interesting content
2. Use Claude to generate a question that the chunk answers
3. Store that chunk as ground truth "relevant" chunk

This ensures the evaluation actually measures retrieval accuracy.

Output:
- 50 simple factual queries (single chunk answer)
- 30 complex analytical queries (may need multiple chunks)
- 20 temporal/comparative queries

Each query includes:
- The natural language question (generated from chunk content)
- The chunk ID(s) that contain the answer
- Keywords extracted from the answer
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


QUESTION_GENERATION_PROMPT = """You are generating evaluation questions for a RAG (Retrieval-Augmented Generation) system that searches SEC filings.

Given the following excerpt from a SEC filing for {company} ({ticker}), generate {question_type}.

SEC Filing Excerpt:
---
{content}
---

Requirements:
1. The question MUST be answerable from the excerpt above
2. The question should be natural - how a financial analyst would ask
3. Do NOT include the company name if it can be inferred from context
4. The question should be specific enough that this excerpt is clearly the best answer

{additional_instructions}

Respond with ONLY a JSON object in this format:
{{
  "question": "Your generated question here",
  "answer_keywords": ["keyword1", "keyword2", "keyword3"],
  "difficulty": "easy|medium|hard"
}}"""


SIMPLE_INSTRUCTIONS = """
Generate a simple factual question that can be answered directly from the text.
Examples: "What products does the company sell?", "What is the company's primary business segment?"
"""

ANALYTICAL_INSTRUCTIONS = """
Generate an analytical question that requires understanding the text's implications.
Examples: "How does the company's cost structure affect profitability?", "What strategic challenges does the company face?"
"""

TEMPORAL_INSTRUCTIONS = """
Generate a question about changes, trends, or time-based comparisons mentioned in the text.
Examples: "How did revenue change compared to last year?", "What new initiatives were launched recently?"
"""


async def get_diverse_chunks(
    pool: asyncpg.Pool,
    category: str,
    limit: int = 50,
) -> list[dict]:
    """Get diverse chunks suitable for question generation."""

    # Different section preferences by category
    if category == "simple_factual":
        sections = ["business", "properties"]
        min_length = 500
        max_length = 2000
    elif category == "complex_analytical":
        sections = ["mda", "risk_factors"]
        min_length = 800
        max_length = 3000
    else:  # temporal
        sections = ["mda"]
        min_length = 600
        max_length = 2500

    section_filter = ", ".join(f"'{s}'" for s in sections)

    async with pool.acquire() as conn:
        # Get chunks with good content for question generation
        rows = await conn.fetch(
            f"""
            WITH ranked_chunks AS (
                SELECT
                    id,
                    content,
                    metadata,
                    LENGTH(content) as content_length,
                    ROW_NUMBER() OVER (
                        PARTITION BY metadata->>'ticker'
                        ORDER BY RANDOM()
                    ) as company_rank
                FROM rag_documents
                WHERE document_type = 'sec_filing'
                  AND metadata->>'section' IN ({section_filter})
                  AND LENGTH(content) BETWEEN $1 AND $2
                  AND content NOT LIKE '%Table of Contents%'
                  AND content NOT LIKE '%Forward-Looking Statements%'
                  AND embedding IS NOT NULL
            )
            SELECT id, content, metadata
            FROM ranked_chunks
            WHERE company_rank <= 3  -- Max 3 per company for diversity
            ORDER BY RANDOM()
            LIMIT $3
        """,
            min_length,
            max_length,
            limit * 2,
        )

        return [
            {
                "id": str(row["id"]),
                "content": row["content"],
                "metadata": row["metadata"]
                if isinstance(row["metadata"], dict)
                else json.loads(row["metadata"]),
            }
            for row in rows
        ]


def extract_keywords(content: str, question: str) -> list[str]:
    """Extract relevant keywords from content based on the question."""
    # Find financial figures
    figures = re.findall(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion))?|\d+(?:\.\d+)?%", content)

    # Find proper nouns
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)

    # Find key business terms
    business_terms = re.findall(
        r"\b(?:revenue|profit|margin|growth|segment|product|service|market|customer|operation)\b",
        content.lower(),
    )

    keywords = list(set(figures[:3] + proper_nouns[:5] + business_terms[:3]))
    return keywords[:8]


async def generate_question_from_chunk(
    client: Anthropic,
    chunk: dict,
    category: str,
) -> dict | None:
    """Use Claude to generate a question from chunk content."""

    if category == "simple_factual":
        question_type = "a simple factual question"
        instructions = SIMPLE_INSTRUCTIONS
    elif category == "complex_analytical":
        question_type = "an analytical question requiring understanding"
        instructions = ANALYTICAL_INSTRUCTIONS
    else:
        question_type = "a temporal or comparative question"
        instructions = TEMPORAL_INSTRUCTIONS

    metadata = chunk["metadata"]
    company = metadata.get("company_name", "Unknown Company")
    ticker = metadata.get("ticker", "UNKN")

    prompt = QUESTION_GENERATION_PROMPT.format(
        company=company,
        ticker=ticker,
        content=chunk["content"][:2500],  # Limit content length
        question_type=question_type,
        additional_instructions=instructions,
    )

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse JSON response
        response_text = response.content[0].text.strip()
        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
            response_text = re.sub(r"\n?```$", "", response_text)

        result = json.loads(response_text)

        return {
            "question": result["question"],
            "answer_keywords": result.get("answer_keywords", []),
            "difficulty": result.get("difficulty", "medium"),
        }

    except Exception as e:
        logger.warning(f"Failed to generate question: {e}")
        return None


async def generate_eval_cases(
    pool: asyncpg.Pool,
    client: Anthropic,
    num_simple: int = 50,
    num_analytical: int = 30,
    num_temporal: int = 20,
) -> list[dict]:
    """Generate evaluation test cases with proper ground truth."""
    test_cases = []

    categories = [
        ("simple_factual", num_simple),
        ("complex_analytical", num_analytical),
        ("temporal_comparative", num_temporal),
    ]

    for category, target_count in categories:
        logger.info(f"Generating {target_count} {category} queries...")

        # Get candidate chunks
        chunks = await get_diverse_chunks(pool, category, limit=target_count * 2)
        logger.info(f"  Found {len(chunks)} candidate chunks")

        generated = 0
        for i, chunk in enumerate(chunks):
            if generated >= target_count:
                break

            # Generate question from chunk
            result = await generate_question_from_chunk(client, chunk, category)
            if not result:
                continue

            metadata = chunk["metadata"]

            test_cases.append(
                {
                    "id": f"{category.split('_')[0]}_{generated + 1:03d}",
                    "query": result["question"],
                    "category": category,
                    "company": metadata.get("ticker", "UNKNOWN"),
                    "expected_section": metadata.get("section", "unknown"),
                    "relevant_chunk_ids": [chunk["id"]],
                    "answer_keywords": result["answer_keywords"],
                    "difficulty": result["difficulty"],
                    "metadata": {
                        "company_name": metadata.get("company_name"),
                        "filing_date": metadata.get("filing_date"),
                        "form_type": metadata.get("form_type"),
                        "chunk_preview": chunk["content"][:200] + "...",
                    },
                }
            )
            generated += 1

            if generated % 10 == 0:
                logger.info(f"  Generated {generated}/{target_count} for {category}")

        logger.info(f"  Completed {generated} {category} queries")

    return test_cases


async def main(args: argparse.Namespace):
    """Main entry point."""
    # Check for API key (support both naming conventions)
    api_key = os.getenv("NL2API_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "NL2API_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY not set. Required for question generation."
        )
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    postgres_url = os.getenv(
        "NL2API_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
    )
    pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)

    try:
        # Check embedding coverage
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COUNT(*) as total,
                       COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embedding
                FROM rag_documents
            """)
            total, with_embedding = row["total"], row["with_embedding"]
            logger.info(
                f"Embedding coverage: {with_embedding}/{total} ({100 * with_embedding / total:.1f}%)"
            )

        # Generate test cases
        test_cases = await generate_eval_cases(
            pool,
            client,
            num_simple=args.num_simple,
            num_analytical=args.num_analytical,
            num_temporal=args.num_temporal,
        )

        logger.info(f"Generated {len(test_cases)} total test cases")

        # Create output structure
        output = {
            "_meta": {
                "name": "sec_filing_rag_evaluation",
                "capability": "rag_retrieval",
                "methodology": "Questions generated from chunk content by Claude 3.5 Haiku",
                "generated_at": datetime.now().isoformat(),
                "total_cases": len(test_cases),
                "categories": {
                    "simple_factual": sum(
                        1 for t in test_cases if t["category"] == "simple_factual"
                    ),
                    "complex_analytical": sum(
                        1 for t in test_cases if t["category"] == "complex_analytical"
                    ),
                    "temporal_comparative": sum(
                        1 for t in test_cases if t["category"] == "temporal_comparative"
                    ),
                },
                "schema_version": "2.0",
            },
            "test_cases": test_cases,
        }

        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved evaluation dataset to {output_path}")

        # Print summary
        print("\n=== Evaluation Dataset Summary ===")
        print(f"Total test cases: {len(test_cases)}")
        for category, count in output["_meta"]["categories"].items():
            print(f"  {category}: {count}")
        print("\nMethodology: Questions generated from actual chunk content")
        print(f"Output: {output_path}")

    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG evaluation dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="tests/fixtures/rag/sec_evaluation_set.json",
        help="Output file path",
    )
    parser.add_argument(
        "--num-simple",
        type=int,
        default=50,
        help="Number of simple factual queries",
    )
    parser.add_argument(
        "--num-analytical",
        type=int,
        default=30,
        help="Number of complex analytical queries",
    )
    parser.add_argument(
        "--num-temporal",
        type=int,
        default=20,
        help="Number of temporal/comparative queries",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
