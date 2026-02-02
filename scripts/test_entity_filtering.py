#!/usr/bin/env python3
"""
Test script to compare retrieval with and without entity filtering.

Measures the impact of the ticker parameter on retrieval quality.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestCase:
    """Test case with query and expected ticker."""

    query: str
    ticker: str
    description: str


# Test cases with entity-specific queries
TEST_CASES = [
    TestCase(
        query="What were Apple's total revenues in fiscal year 2024?",
        ticker="AAPL",
        description="Apple revenue query",
    ),
    TestCase(
        query="What is Microsoft's cloud revenue growth?",
        ticker="MSFT",
        description="Microsoft cloud query",
    ),
    TestCase(
        query="What are Google's risk factors related to AI?",
        ticker="GOOGL",
        description="Google AI risks query",
    ),
    TestCase(
        query="What is Amazon's free cash flow?",
        ticker="AMZN",
        description="Amazon FCF query",
    ),
    TestCase(
        query="What are Tesla's production numbers?",
        ticker="TSLA",
        description="Tesla production query",
    ),
]


async def test_retrieval(
    pool, embedder, query: str, expected_ticker: str, filter_ticker: str | None = None
) -> dict:
    """Run retrieval and return metrics.

    Args:
        pool: Database connection pool
        embedder: Embedding model
        query: Search query
        expected_ticker: The ticker we expect to find (for measuring precision)
        filter_ticker: Ticker to pass as filter (None = no filtering)
    """
    from src.rag.retriever.retriever import HybridRAGRetriever

    retriever = HybridRAGRetriever(pool)
    retriever.set_embedder(embedder)

    results = await retriever.retrieve(
        query=query,
        limit=10,
        threshold=0.0,
        ticker=filter_ticker,  # This is the filter passed to the retriever
        use_cache=False,
    )

    # Check how many results are from the expected ticker
    correct_ticker_count = 0
    wrong_ticker_count = 0
    tickers_found = []

    for r in results:
        result_ticker = r.metadata.get("ticker") if r.metadata else None
        tickers_found.append(result_ticker)
        if result_ticker == expected_ticker:  # Compare against expected, not filter
            correct_ticker_count += 1
        else:
            wrong_ticker_count += 1

    return {
        "total_results": len(results),
        "correct_ticker": correct_ticker_count,
        "wrong_ticker": wrong_ticker_count,
        "precision": correct_ticker_count / len(results) if results else 0,
        "top_result_ticker": (
            results[0].metadata.get("ticker") if results and results[0].metadata else None
        ),
        "top_result_score": results[0].score if results else 0,
        "unique_tickers": list(set(tickers_found)),
    }


async def main():
    """Main test function."""
    import asyncpg

    from src.rag.retriever.embedders import OpenAIEmbedder

    # Connect to database
    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        user="nl2api",
        password="nl2api",
        database="nl2api",
        min_size=1,
        max_size=5,
    )

    # Create embedder with API key from environment
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NL2API_OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY or NL2API_OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    embedder = OpenAIEmbedder(api_key=api_key)

    print("=" * 80)
    print("Entity Filtering A/B Test")
    print("=" * 80)
    print()

    # Aggregate metrics
    without_filter_correct = 0
    without_filter_total = 0
    with_filter_correct = 0
    with_filter_total = 0

    for tc in TEST_CASES:
        print(f"\n{tc.description}")
        print(f"  Query: {tc.query[:60]}...")
        print(f"  Expected ticker: {tc.ticker}")

        # Test WITHOUT entity filtering
        without = await test_retrieval(pool, embedder, tc.query, tc.ticker, filter_ticker=None)
        print("\n  WITHOUT entity filter:")
        print(f"    Results: {without['total_results']}")
        print(f"    Correct ticker: {without['correct_ticker']}/{without['total_results']}")
        print(f"    Precision: {without['precision']:.1%}")
        print(
            f"    Top result: {without['top_result_ticker']} (score: {without['top_result_score']:.3f})"
        )
        print(f"    Tickers in results: {without['unique_tickers']}")

        without_filter_correct += without["correct_ticker"]
        without_filter_total += without["total_results"]

        # Test WITH entity filtering
        with_filter = await test_retrieval(
            pool, embedder, tc.query, tc.ticker, filter_ticker=tc.ticker
        )
        print(f"\n  WITH entity filter (ticker={tc.ticker}):")
        print(f"    Results: {with_filter['total_results']}")
        print(f"    Correct ticker: {with_filter['correct_ticker']}/{with_filter['total_results']}")
        print(f"    Precision: {with_filter['precision']:.1%}")
        print(
            f"    Top result: {with_filter['top_result_ticker']} (score: {with_filter['top_result_score']:.3f})"
        )
        print(f"    Tickers in results: {with_filter['unique_tickers']}")

        with_filter_correct += with_filter["correct_ticker"]
        with_filter_total += with_filter["total_results"]

        # Compare
        improvement = with_filter["precision"] - without["precision"]
        print(f"\n  Improvement: {improvement:+.1%}")
        print("-" * 60)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    without_precision = without_filter_correct / without_filter_total if without_filter_total else 0
    with_precision = with_filter_correct / with_filter_total if with_filter_total else 0

    print("\nWithout entity filter:")
    print(f"  Correct ticker results: {without_filter_correct}/{without_filter_total}")
    print(f"  Overall precision: {without_precision:.1%}")

    print("\nWith entity filter:")
    print(f"  Correct ticker results: {with_filter_correct}/{with_filter_total}")
    print(f"  Overall precision: {with_precision:.1%}")

    improvement = with_precision - without_precision
    lift = (with_precision / without_precision - 1) * 100 if without_precision else float("inf")
    print("\nImprovement:")
    print(f"  Absolute: {improvement:+.1%}")
    print(f"  Relative lift: {lift:+.1f}%")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
