"""
SEC RAG Answer Generator (Phase 2).

Generates reference answers for SEC filing questions by:
1. Loading questions from tests/fixtures/rag/sec_filings/questions.json
2. Retrieving context from pgvector using HybridRAGRetriever
3. Generating answers with citations using Claude
4. Saving complete test cases with ground truth

Run AFTER SEC ingestion completes.

Usage:
    # Test run (5 questions)
    python scripts/generate_sec_rag_answers.py --limit 5

    # Full generation
    python scripts/generate_sec_rag_answers.py

    # Use OpenAI embeddings instead of local
    python scripts/generate_sec_rag_answers.py --embedder openai
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic
import asyncpg
from dotenv import load_dotenv

from src.rag.retriever.embedders import LocalEmbedder, OpenAIEmbedder
from src.rag.retriever.protocols import DocumentType, RetrievalResult
from src.rag.retriever.retriever import HybridRAGRetriever


class CompanyExtractor:
    """Extracts company names and tickers from queries."""

    def __init__(self, companies_path: Path):
        """Load company data and build lookup structures."""
        with open(companies_path) as f:
            data = json.load(f)

        self.companies = data.get("companies", [])

        # Build lookup structures
        self.ticker_to_name: dict[str, str] = {}
        self.name_patterns: list[tuple[re.Pattern, str]] = []

        for company in self.companies:
            ticker = company["ticker"]
            name = company["name"]
            self.ticker_to_name[ticker.upper()] = ticker

            # Create patterns for company names (case insensitive)
            # Handle variations like "Apple Inc." -> matches "Apple", "Apple Inc", etc.
            base_name = name.split()[0]  # First word
            if len(base_name) > 3:  # Avoid matching short words
                pattern = re.compile(rf"\b{re.escape(base_name)}\b", re.IGNORECASE)
                self.name_patterns.append((pattern, ticker))

            # Full name pattern
            full_pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
            self.name_patterns.append((full_pattern, ticker))

    def extract_ticker(self, query: str) -> str | None:
        """
        Extract company ticker from query.

        Tries:
        1. Direct ticker mention (e.g., "AAPL's revenue")
        2. Company name mention (e.g., "Apple's revenue")

        Returns ticker or None.
        """
        # Check for ticker patterns (uppercase 1-5 letters followed by 's or space)
        ticker_pattern = re.compile(r"\b([A-Z]{1,5})(?:'s|'s|\s|$)")
        for match in ticker_pattern.finditer(query):
            candidate = match.group(1)
            if candidate in self.ticker_to_name:
                return self.ticker_to_name[candidate]

        # Check for company name patterns
        for pattern, ticker in self.name_patterns:
            if pattern.search(query):
                return ticker

        return None


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """You are a financial analyst answering questions based on SEC filing excerpts.

Given a question and retrieved context from 10-K/10-Q filings, provide a concise, factual answer.

Rules:
- Answer ONLY based on the provided context - do not use external knowledge
- Include specific numbers, dates, and facts from the filings
- If the context doesn't contain enough information to answer, respond with: {"answer": null, "reason": "Insufficient context", "citations": []}
- Keep answers concise (1-3 sentences for simple questions, up to 5 for complex)
- Use precise financial terminology
- Do NOT speculate beyond what the filings state
- Do NOT provide investment advice

Output format: JSON object with "answer" and "citations" array:
{
  "answer": "Plain text answer here",
  "citations": [
    {"filing": "AAPL 10-K FY2023", "section": "Item 8", "chunk_id": "abc123"}
  ]
}

If you cannot answer from the context, use:
{
  "answer": null,
  "reason": "Brief explanation of why the context is insufficient",
  "citations": []
}"""


def format_context(chunks: list[RetrievalResult]) -> str:
    """Format retrieved chunks as context for the LLM."""
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata or {}
        filing_info = f"{metadata.get('ticker', 'Unknown')} {metadata.get('filing_type', '10-K')} {metadata.get('fiscal_year', '')}"
        section = metadata.get("section", "Unknown Section")

        context_parts.append(
            f"--- Document {i} (Score: {chunk.score:.3f}) ---\n"
            f"Filing: {filing_info}\n"
            f"Section: {section}\n"
            f"Chunk ID: {chunk.id}\n"
            f"Content:\n{chunk.content}\n"
        )

    return "\n".join(context_parts)


def extract_json_from_response(content: str) -> dict[str, Any]:
    """Extract JSON object from LLM response, handling various formats."""
    content = content.strip()

    # Handle markdown code blocks
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end > start:
            content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end > start:
            content = content[start:end].strip()

    # Find the first JSON object by matching braces
    if "{" in content:
        start = content.find("{")
        depth = 0
        end = start
        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        content = content[start:end]

    return json.loads(content)


async def generate_answer(
    client: anthropic.AsyncAnthropic,
    query: str,
    context: str,
    use_sonnet: bool = False,
) -> dict[str, Any]:
    """Generate an answer using Claude with retrieved context."""
    model = "claude-sonnet-4-20250514" if use_sonnet else "claude-3-5-haiku-20241022"

    user_prompt = f"""Question: {query}

Context from SEC filings:
{context}

Please answer the question based only on the provided context. Output valid JSON only, no other text."""

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1000,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = response.content[0].text
        return extract_json_from_response(content)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        # Try to extract answer from raw text
        raw = response.content[0].text if response else ""
        return {"answer": None, "reason": f"Failed to parse response: {raw[:200]}", "citations": []}
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"answer": None, "reason": str(e), "citations": []}


async def retrieve_with_company_filter(
    pool: asyncpg.Pool,
    retriever: HybridRAGRetriever,
    query: str,
    ticker: str | None,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Retrieve documents with optional company filtering.

    If ticker is provided, uses a two-stage approach:
    1. First retrieves from company-specific documents
    2. Falls back to general retrieval if not enough results

    This addresses the hybrid scoring imbalance where vector similarity
    can override keyword matching for company names.
    """
    if ticker:
        # Company-filtered retrieval using direct SQL
        async with pool.acquire() as conn:
            # Get embedding for the query
            query_embedding_list = await retriever._embedder.embed(query)
            query_embedding = "[" + ",".join(str(x) for x in query_embedding_list) + "]"

            # Build ticker filter as JSON for @> operator
            # Note: Using @> with jsonb is more reliable with vector queries
            # than ->> with parameterized strings
            ticker_filter = json.dumps({"ticker": ticker})

            # Retrieve from company-specific documents
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    content,
                    document_type,
                    domain,
                    field_code,
                    example_query,
                    example_api_call,
                    metadata,
                    1 - (embedding <=> $1::vector) as score
                FROM rag_documents
                WHERE document_type = 'sec_filing'
                AND metadata @> $2::jsonb
                AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                query_embedding,
                ticker_filter,
                top_k,
            )

            if rows:
                results = []
                for row in rows:
                    raw_metadata = row["metadata"]
                    if raw_metadata is None:
                        metadata = {}
                    elif isinstance(raw_metadata, str):
                        try:
                            metadata = json.loads(raw_metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        metadata = raw_metadata

                    results.append(
                        RetrievalResult(
                            id=str(row["id"]),
                            content=row["content"],
                            document_type=DocumentType(row["document_type"]),
                            score=float(row["score"]),
                            domain=row["domain"],
                            field_code=row["field_code"],
                            example_query=row["example_query"],
                            example_api_call=row["example_api_call"],
                            metadata=metadata,
                        )
                    )
                return results

    # Fall back to standard retrieval
    return await retriever.retrieve(
        query=query,
        document_types=[DocumentType.SEC_FILING],
        limit=top_k,
        threshold=0.2,
        use_cache=False,
    )


async def process_question(
    question: dict,
    pool: asyncpg.Pool,
    retriever: HybridRAGRetriever,
    client: anthropic.AsyncAnthropic,
    company_extractor: CompanyExtractor | None,
    top_k: int = 5,
) -> dict:
    """Process a single question: retrieve context and generate answer."""
    query = question["input"]["query"]
    subcategory = question.get("subcategory", "")

    # Skip rejection cases - they don't need answers
    if question["expected"]["behavior"] == "reject":
        return question

    # Try to extract company ticker from query or metadata
    ticker = None
    if company_extractor:
        ticker = company_extractor.extract_ticker(query)
    # Also check question metadata as fallback
    if not ticker:
        ticker = question.get("metadata", {}).get("ticker")

    # Retrieve relevant chunks (with company filtering if available)
    try:
        chunks = await retrieve_with_company_filter(
            pool=pool,
            retriever=retriever,
            query=query,
            ticker=ticker,
            top_k=top_k,
        )
        if ticker:
            logger.debug(f"Company-filtered retrieval for {ticker}: {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Retrieval failed for '{query[:50]}...': {e}")
        chunks = []

    # Build context string
    context = format_context(chunks)

    # Generate answer (use Sonnet for complex queries)
    use_sonnet = subcategory == "complex_queries"
    result = await generate_answer(client, query, context, use_sonnet=use_sonnet)

    # Update question with ground truth
    question["expected"]["relevant_docs"] = [str(c.id) for c in chunks]
    question["expected"]["answer"] = result.get("answer")
    question["expected"]["citations"] = result.get("citations", [])

    if result.get("reason"):
        question["expected"]["no_answer_reason"] = result["reason"]

    # Record which ticker was used for filtering
    if ticker:
        question["metadata"]["retrieval_ticker"] = ticker

    return question


async def generate_answers(
    questions_path: Path,
    output_dir: Path,
    limit: int | None = None,
    embedder_type: str = "local",
    top_k: int = 5,
    concurrency: int = 10,
    resume: bool = False,
):
    """
    Generate reference answers for SEC RAG questions.

    This is Phase 2 - run after SEC ingestion completes.
    """
    # Load questions
    with open(questions_path) as f:
        data = json.load(f)

    all_questions = data["test_cases"]
    if limit:
        all_questions = all_questions[:limit]

    # Filter to only answerable questions (not rejections)
    answerable = [q for q in all_questions if q["expected"]["behavior"] != "reject"]
    rejections = [q for q in all_questions if q["expected"]["behavior"] == "reject"]

    # Resume: load existing answers and skip completed questions
    completed_ids: set[str] = set()
    existing_answers: dict[str, dict] = {}
    output_path = output_dir / "questions_with_answers.json"

    if resume and output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
        for tc in existing_data.get("test_cases", []):
            if tc.get("expected", {}).get("answer") is not None:
                completed_ids.add(tc["id"])
                existing_answers[tc["id"]] = tc
        print(f"Resuming: found {len(completed_ids)} already completed")

        # Filter out already completed
        answerable = [q for q in answerable if q["id"] not in completed_ids]

    print(f"Loaded {len(all_questions)} questions from {questions_path}")
    print(f"  - Answerable: {len(answerable)}")
    print(f"  - Already completed: {len(completed_ids)}")
    print(f"  - Rejections: {len(rejections)} (skipped)")
    print("=" * 60)

    # Connect to database
    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api"
    )
    pool = await asyncpg.create_pool(database_url)

    try:
        # Check for SEC filing documents
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_documents WHERE document_type = 'sec_filing'"
            )
            print(f"Found {count} SEC filing chunks in database")
            if count == 0:
                print("ERROR: No SEC filing chunks found. Run SEC ingestion first.")
                return

        # Initialize embedder
        if embedder_type == "openai":
            api_key = os.getenv("NL2API_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required for openai embedder")
            embedder = OpenAIEmbedder(api_key=api_key)
            embedding_dim = 1536
            print("Using OpenAI embeddings (1536 dims)")
        else:
            embedder = LocalEmbedder()
            embedding_dim = embedder.dimension
            print(f"Using local embeddings ({embedding_dim} dims)")

        # Initialize retriever
        retriever = HybridRAGRetriever(
            pool=pool,
            embedding_dimension=embedding_dim,
            vector_weight=0.7,
            keyword_weight=0.3,
        )
        retriever.set_embedder(embedder)

        # Initialize Anthropic client
        api_key = os.getenv("NL2API_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY)")
        client = anthropic.AsyncAnthropic(api_key=api_key)

        # Initialize company extractor for improved retrieval
        companies_path = PROJECT_ROOT / "scripts" / "data" / "sec" / "sp500_companies.json"
        company_extractor = None
        if companies_path.exists():
            company_extractor = CompanyExtractor(companies_path)
            print(f"Loaded {len(company_extractor.companies)} companies for ticker extraction")
        else:
            print("Warning: Company list not found, using metadata-only ticker extraction")

        # Process questions with concurrency
        stats = {
            "processed": 0,
            "with_answer": 0,
            "no_answer": 0,
            "errors": 0,
            "company_filtered": 0,
        }

        print(f"\nProcessing {len(answerable)} questions with {concurrency} concurrent requests...")

        # Semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(idx: int, question: dict) -> dict:
            """Process a single question with semaphore-controlled concurrency."""
            async with semaphore:
                query = question["input"]["query"]
                try:
                    result = await process_question(
                        question, pool, retriever, client, company_extractor, top_k
                    )

                    # Log result
                    ticker_info = result.get("metadata", {}).get("retrieval_ticker", "")
                    ticker_note = f" [{ticker_info}]" if ticker_info else ""
                    if result["expected"].get("answer"):
                        print(f"[{idx + 1}/{len(answerable)}] ✓{ticker_note} {query[:50]}...")
                    else:
                        print(f"[{idx + 1}/{len(answerable)}] ✗{ticker_note} {query[:50]}...")

                    return result
                except Exception as e:
                    logger.error(f"Error processing question {idx + 1}: {e}")
                    question["expected"]["answer"] = None
                    question["expected"]["relevant_docs"] = []
                    question["expected"]["no_answer_reason"] = f"Processing error: {e}"
                    print(f"[{idx + 1}/{len(answerable)}] ERROR: {query[:50]}...")
                    return question

        # Process all questions concurrently (with semaphore limiting)
        tasks = [process_with_semaphore(i, q) for i, q in enumerate(answerable)]
        completed = await asyncio.gather(*tasks)

        # Calculate stats
        for result in completed:
            stats["processed"] += 1
            if result.get("metadata", {}).get("retrieval_ticker"):
                stats["company_filtered"] += 1
            if result["expected"].get("answer"):
                stats["with_answer"] += 1
            elif result["expected"].get("no_answer_reason"):
                if "error" in result["expected"]["no_answer_reason"].lower():
                    stats["errors"] += 1
                else:
                    stats["no_answer"] += 1
            else:
                stats["no_answer"] += 1

        # Merge with existing answers if resuming
        completed = list(completed)  # Convert from tuple
        if existing_answers:
            # Add back previously completed answers
            completed.extend(existing_answers.values())

        # Add rejection cases back (unchanged)
        completed.extend(rejections)

        print("\n" + "=" * 60)
        print("Generation complete!")
        print(f"  Processed: {stats['processed']}")
        print(f"  With answer: {stats['with_answer']}")
        print(f"  No answer: {stats['no_answer']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Company-filtered retrieval: {stats['company_filtered']}")
        print(f"  Rejections (unchanged): {len(rejections)}")
        if stats["processed"] > 0:
            success_rate = stats["with_answer"] / stats["processed"] * 100
            print(f"  Success rate: {success_rate:.1f}%")

        # Save results
        save_results(completed, output_dir, data.get("_meta", {}))

    finally:
        await pool.close()


def save_results(test_cases: list[dict], output_dir: Path, original_meta: dict):
    """Save completed test cases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all to single file
    output_path = output_dir / "questions_with_answers.json"
    data = {
        "_meta": {
            **original_meta,
            "phase": "complete",
            "answers_generated_at": datetime.now(UTC).isoformat(),
            "generator": "scripts/generate_sec_rag_answers.py",
        },
        "test_cases": test_cases,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(test_cases)} test cases to {output_path}")

    # Also save by subcategory for easier evaluation
    by_category: dict[str, list] = {}
    for tc in test_cases:
        subcat = tc.get("subcategory", "unknown")
        if subcat not in by_category:
            by_category[subcat] = []
        by_category[subcat].append(tc)

    for subcat, cases in by_category.items():
        cat_path = output_dir / f"{subcat}.json"
        cat_data = {
            "_meta": {
                "name": f"sec_filings_rag_{subcat}",
                "capability": "rag_evaluation",
                "schema_version": "1.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "phase": "complete",
            },
            "test_cases": cases,
        }
        with open(cat_path, "w") as f:
            json.dump(cat_data, f, indent=2)
        print(f"  - {subcat}: {len(cases)} cases -> {cat_path.name}")


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
    parser.add_argument(
        "--embedder",
        choices=["local", "openai"],
        default="local",
        help="Embedder to use (local=sentence-transformers, openai=OpenAI API)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress, skipping already completed questions",
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
        embedder_type=args.embedder,
        top_k=args.top_k,
        concurrency=args.concurrency,
        resume=args.resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
