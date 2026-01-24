"""
SEC RAG Answer Generator (Phase 2).

Generates reference answers for SEC filing questions by:
1. Loading questions from tests/fixtures/rag/sec_filings/questions.json
2. Retrieving context from pgvector using HybridRAGRetriever
3. Generating answers with citations using Claude
4. Saving complete test cases with ground truth

Run AFTER SEC ingestion completes.

Usage:
    # Test run
    python scripts/generate_sec_rag_answers.py --limit 10

    # Full generation
    python scripts/generate_sec_rag_answers.py
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

# TODO: Import after ingestion infrastructure is ready
# from src.nl2api.rag.retriever import HybridRAGRetriever


ANSWER_SYSTEM_PROMPT = """You are a financial analyst answering questions based on SEC filing excerpts.

Given a question and retrieved context from 10-K/10-Q filings, provide a concise, factual answer with source citations.

Rules:
- Answer ONLY based on the provided context - do not use external knowledge
- Include specific numbers, dates, and facts from the filings
- If the context doesn't contain enough information to answer, respond with: {"answer": null, "reason": "...", "citations": []}
- Keep answers concise (1-3 sentences for simple questions, up to 5 for complex)
- Use precise financial terminology
- Do NOT speculate beyond what the filings state

Output format: JSON object with "answer" (plain text, no citations embedded) and "citations" array:
{
  "answer": "Plain text answer here",
  "citations": [
    {"filing": "AAPL 10-K FY2023", "section": "Item 8", "chunk_id": "..."},
    {"filing": "AAPL 10-K FY2023", "section": "Item 7", "chunk_id": "..."}
  ]
}"""


async def generate_answers(
    questions_path: Path,
    output_dir: Path,
    limit: int | None = None,
    use_sonnet_for_complex: bool = True,
):
    """
    Generate reference answers for SEC RAG questions.

    This is Phase 2 - run after SEC ingestion completes.
    """
    # Load questions
    with open(questions_path) as f:
        data = json.load(f)

    questions = data["test_cases"]
    if limit:
        questions = questions[:limit]

    print(f"Loaded {len(questions)} questions from {questions_path}")
    print("=" * 60)

    # TODO: Initialize retriever and LLM client
    # retriever = HybridRAGRetriever(...)
    # client = anthropic.AsyncAnthropic()

    completed_cases = []

    for i, question in enumerate(questions):
        query = question["input"]["query"]

        # Skip rejection cases - they don't need answers
        if question["expected"]["behavior"] == "reject":
            completed_cases.append(question)
            continue

        print(f"[{i + 1}/{len(questions)}] Processing: {query[:60]}...")

        # TODO: Implement when retriever is ready
        # 1. Retrieve relevant chunks
        # chunks = await retriever.retrieve(query, top_k=5)
        #
        # 2. Build context string with chunk metadata
        # context = build_context_string(chunks)
        #
        # 3. Generate answer (use Sonnet for complex_queries, Haiku for others)
        # subcategory = question["subcategory"]
        # model = "claude-sonnet-4-20250514" if subcategory == "complex_queries" else "claude-3-5-haiku-20241022"
        # response = await client.messages.create(...)
        #
        # 4. Parse response and add to question
        # question["expected"]["relevant_docs"] = [c.id for c in chunks]
        # question["expected"]["answer"] = response["answer"]
        # question["expected"]["citations"] = response["citations"]

        # Placeholder for now
        question["expected"]["relevant_docs"] = []
        question["expected"]["answer"] = None
        question["expected"]["citations"] = []

        completed_cases.append(question)

    # Save completed test cases by category
    save_by_category(completed_cases, output_dir)

    print(f"\nCompleted {len(completed_cases)} test cases")
    print("NOTE: This is a stub - implement retrieval after SEC ingestion completes")


def save_by_category(test_cases: list[dict], output_dir: Path):
    """Save test cases split by subcategory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by subcategory
    by_category: dict[str, list] = {}
    for tc in test_cases:
        subcat = tc["subcategory"]
        if subcat not in by_category:
            by_category[subcat] = []
        by_category[subcat].append(tc)

    # Save each category
    for subcat, cases in by_category.items():
        output_path = output_dir / f"{subcat}.json"
        data = {
            "_meta": {
                "name": f"sec_filings_rag_{subcat}",
                "capability": "rag_evaluation",
                "schema_version": "1.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "phase": "complete",
                "generator": "scripts/generate_sec_rag_answers.py",
            },
            "test_cases": cases,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(cases)} cases to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Generate SEC RAG reference answers")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("tests/fixtures/rag/sec_filings/questions.json"),
        help="Path to questions file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures/rag/sec_filings"),
        help="Output directory for complete test cases",
    )
    args = parser.parse_args()

    if not args.questions.exists():
        print(f"Error: Questions file not found: {args.questions}")
        print(
            "Run question generator first: python -m scripts.generators.sec_rag_question_generator"
        )
        return

    await generate_answers(
        questions_path=args.questions,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    asyncio.run(main())
