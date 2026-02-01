#!/usr/bin/env python3
"""
Enrich raw questions with metadata for RAG evaluation.

Input: Text file with one question per line, or JSON with {"questions": [...]}
Output: JSON fixture file ready for load_rag_fixtures.py

Usage:
    python scripts/enrich_rag_questions.py questions.txt -o tests/fixtures/rag/my_eval.json
    python scripts/enrich_rag_questions.py questions.json -o tests/fixtures/rag/my_eval.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import anthropic

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ENRICHMENT_PROMPT = """Analyze this financial question and extract metadata.

Question: {question}

Return a JSON object with:
1. "category": One of:
   - "simple_factual" (direct fact lookup, single data point)
   - "complex_analytical" (requires reasoning, multiple data points)
   - "temporal_comparative" (involves time periods, trends, changes)

2. "difficulty": One of "easy", "medium", "hard" based on:
   - easy: Single fact, common metric (revenue, profit)
   - medium: Requires some context or calculation
   - hard: Multiple sources, nuanced interpretation

3. "answer_keywords": List of 3-5 words/phrases that a correct answer MUST contain.
   Focus on: metrics mentioned, time periods, key terms from the question.

4. "company": Ticker symbol if mentioned or inferable, else "UNKNOWN"

5. "company_name": Full company name if known, else null

Return ONLY valid JSON, no explanation."""


async def enrich_question(client: anthropic.AsyncAnthropic, question: str) -> dict:
    """Use Claude to analyze a question and generate metadata."""
    response = await client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": ENRICHMENT_PROMPT.format(question=question)}],
    )

    text = response.content[0].text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text)


async def enrich_questions(questions: list[str], concurrency: int = 5) -> list[dict]:
    """Enrich a list of questions with metadata."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(idx: int, question: str) -> dict:
        async with semaphore:
            try:
                metadata = await enrich_question(client, question)
                logger.info(
                    f"  [{idx + 1}/{len(questions)}] {question[:50]}... -> {metadata['category']}"
                )
                return {
                    "id": f"q{idx + 1:03d}",
                    "query": question,
                    "category": metadata.get("category", "simple_factual"),
                    "company": metadata.get("company", "UNKNOWN"),
                    "relevant_chunk_ids": [],  # Will be filled by retrieval
                    "answer_keywords": metadata.get("answer_keywords", []),
                    "difficulty": metadata.get("difficulty", "medium"),
                    "metadata": {
                        "company_name": metadata.get("company_name"),
                        "source": "manual_enrichment",
                    },
                }
            except Exception as e:
                logger.warning(f"  [{idx + 1}] Failed: {e}")
                return {
                    "id": f"q{idx + 1:03d}",
                    "query": question,
                    "category": "simple_factual",
                    "company": "UNKNOWN",
                    "relevant_chunk_ids": [],
                    "answer_keywords": [],
                    "difficulty": "medium",
                    "metadata": {"error": str(e)},
                }

    logger.info(f"Enriching {len(questions)} questions...")
    tasks = [process_one(i, q) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)
    return results


def load_questions(input_path: Path) -> list[str]:
    """Load questions from txt or json file."""
    content = input_path.read_text()

    if input_path.suffix == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return [
                q if isinstance(q, str) else q.get("query", q.get("question", "")) for q in data
            ]
        elif "questions" in data:
            return data["questions"]
        elif "test_cases" in data:
            return [tc.get("query", tc.get("question", "")) for tc in data["test_cases"]]
        else:
            raise ValueError("JSON must have 'questions' or 'test_cases' key, or be a list")
    else:
        # Text file: one question per line
        return [line.strip() for line in content.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Enrich questions with RAG evaluation metadata")
    parser.add_argument("input", type=Path, help="Input file (txt or json)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent API calls")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    questions = load_questions(args.input)
    logger.info(f"Loaded {len(questions)} questions from {args.input}")

    enriched = asyncio.run(enrich_questions(questions, args.concurrency))

    # Build output fixture
    output = {
        "_meta": {
            "name": args.output.stem,
            "capability": "rag_retrieval",
            "methodology": "manual_questions_enriched",
            "total_cases": len(enriched),
        },
        "test_cases": enriched,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    logger.info(f"\nWrote {len(enriched)} enriched test cases to {args.output}")

    # Summary
    categories = {}
    for tc in enriched:
        cat = tc["category"]
        categories[cat] = categories.get(cat, 0) + 1
    logger.info("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
