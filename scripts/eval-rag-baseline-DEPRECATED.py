#!/usr/bin/env python3
"""
DEPRECATED: Use batch evaluation framework instead.

This standalone script does NOT integrate with the observability stack (Prometheus/Grafana)
and results are not tracked over time. Use the batch evaluation framework:

    # Load fixtures first (one-time)
    python scripts/load-rag-fixtures.py

    # Run RAG evaluation with proper tracking
    python -m src.evaluation.cli.main batch run --pack rag --tag rag --label your-label

See docs/plans/rag-ingestion-improvements.md for details.

---
Original description (for reference):
Measures retrieval performance on the SEC evaluation dataset:
- Recall@K: Fraction of relevant docs retrieved in top K
- MRR@K: Mean Reciprocal Rank in top K
- NDCG@K: Normalized Discounted Cumulative Gain
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv

from src.evalkit.common.git_info import get_git_info
from src.rag.retriever.embedders import OpenAIEmbedder
from src.rag.retriever.retriever import HybridRAGRetriever

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Recall@K."""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def calculate_mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank @K."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate NDCG@K."""
    relevant_set = set(relevant_ids)

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # +2 because rank starts at 1

    # IDCG: best possible DCG (all relevant docs at top)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Precision@K."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / k


async def run_evaluation(
    pool: asyncpg.Pool,
    retriever: HybridRAGRetriever,
    test_cases: list[dict],
    top_k: int = 10,
) -> dict[str, Any]:
    """Run evaluation on test cases."""
    results = {
        "test_cases": [],
        "metrics": {},
        "by_category": {},
    }

    # Track metrics per category
    category_metrics = {}

    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        relevant_ids = test_case["relevant_chunk_ids"]
        category = test_case["category"]

        try:
            # Run retrieval
            retrieved = await retriever.retrieve(
                query=query,
                document_types=None,  # All types
                limit=top_k,
                threshold=0.0,  # No threshold filtering for evaluation
                use_cache=False,
            )

            retrieved_ids = [str(r.id) for r in retrieved]

            # Calculate metrics
            recall = calculate_recall_at_k(retrieved_ids, relevant_ids, top_k)
            mrr = calculate_mrr_at_k(retrieved_ids, relevant_ids, top_k)
            ndcg = calculate_ndcg_at_k(retrieved_ids, relevant_ids, top_k)
            precision = calculate_precision_at_k(retrieved_ids, relevant_ids, top_k)

            # Store per-case result
            case_result = {
                "id": test_case["id"],
                "query": query,
                "category": category,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "recall": recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "precision": precision,
                "hit": recall > 0,  # At least one relevant doc retrieved
            }
            results["test_cases"].append(case_result)

            # Aggregate by category
            if category not in category_metrics:
                category_metrics[category] = {
                    "recall": [],
                    "mrr": [],
                    "ndcg": [],
                    "precision": [],
                    "hits": 0,
                    "total": 0,
                }
            category_metrics[category]["recall"].append(recall)
            category_metrics[category]["mrr"].append(mrr)
            category_metrics[category]["ndcg"].append(ndcg)
            category_metrics[category]["precision"].append(precision)
            category_metrics[category]["total"] += 1
            if recall > 0:
                category_metrics[category]["hits"] += 1

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(test_cases)} | Last MRR: {mrr:.3f}")

        except Exception as e:
            logger.error(f"Error on case {test_case['id']}: {e}")
            results["test_cases"].append(
                {
                    "id": test_case["id"],
                    "query": query,
                    "category": category,
                    "error": str(e),
                }
            )

    # Calculate overall metrics
    all_recall = [c["recall"] for c in results["test_cases"] if "recall" in c]
    all_mrr = [c["mrr"] for c in results["test_cases"] if "mrr" in c]
    all_ndcg = [c["ndcg"] for c in results["test_cases"] if "ndcg" in c]
    all_precision = [c["precision"] for c in results["test_cases"] if "precision" in c]
    all_hits = sum(1 for c in results["test_cases"] if c.get("hit", False))

    results["metrics"] = {
        f"recall@{top_k}": sum(all_recall) / len(all_recall) if all_recall else 0,
        f"mrr@{top_k}": sum(all_mrr) / len(all_mrr) if all_mrr else 0,
        f"ndcg@{top_k}": sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0,
        f"precision@{top_k}": sum(all_precision) / len(all_precision) if all_precision else 0,
        "hit_rate": all_hits / len(results["test_cases"]) if results["test_cases"] else 0,
        "total_cases": len(results["test_cases"]),
        "successful_cases": len(all_recall),
    }

    # Calculate per-category metrics
    for category, metrics in category_metrics.items():
        results["by_category"][category] = {
            f"recall@{top_k}": sum(metrics["recall"]) / len(metrics["recall"])
            if metrics["recall"]
            else 0,
            f"mrr@{top_k}": sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0,
            f"ndcg@{top_k}": sum(metrics["ndcg"]) / len(metrics["ndcg"]) if metrics["ndcg"] else 0,
            f"precision@{top_k}": sum(metrics["precision"]) / len(metrics["precision"])
            if metrics["precision"]
            else 0,
            "hit_rate": metrics["hits"] / metrics["total"] if metrics["total"] > 0 else 0,
            "count": metrics["total"],
        }

    return results


async def check_embedding_coverage(pool: asyncpg.Pool) -> dict:
    """Check how many documents have embeddings."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embedding,
                100.0 * COUNT(*) FILTER (WHERE embedding IS NOT NULL) / COUNT(*) as pct_complete
            FROM rag_documents
        """)
        return dict(row)


async def main(args: argparse.Namespace):
    """Main entry point."""
    # Load evaluation dataset
    eval_path = Path(args.eval_dataset)
    if not eval_path.exists():
        logger.error(f"Evaluation dataset not found: {eval_path}")
        sys.exit(1)

    with open(eval_path) as f:
        eval_data = json.load(f)

    test_cases = eval_data["test_cases"]
    logger.info(f"Loaded {len(test_cases)} test cases from {eval_path}")

    # Connect to database
    postgres_url = os.getenv(
        "NL2API_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
    )
    pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)

    try:
        # Check embedding coverage
        coverage = await check_embedding_coverage(pool)
        logger.info(
            f"Embedding coverage: {coverage['with_embedding']}/{coverage['total']} "
            f"({coverage['pct_complete']:.1f}%)"
        )

        if coverage["pct_complete"] < 10:
            logger.warning(
                "Less than 10% of documents have embeddings. "
                "Results may not be representative. Consider waiting for more embeddings."
            )
            if not args.force:
                logger.info("Use --force to run anyway.")
                return

        # Create embedder and retriever
        api_key = os.getenv("NL2API_OPENAI_API_KEY")
        if not api_key:
            logger.error("NL2API_OPENAI_API_KEY not set")
            sys.exit(1)

        embedder = OpenAIEmbedder(
            api_key=api_key,
            model="text-embedding-3-small",
        )

        # Create reranker if requested
        reranker = None
        if args.with_reranking:
            from src.rag.retriever.reranker import create_reranker

            logger.info("Loading cross-encoder reranker...")
            reranker = create_reranker()
            logger.info(f"Reranker loaded. First stage limit: {args.first_stage_limit}")

        retriever = HybridRAGRetriever(
            pool=pool,
            embedding_dimension=1536,
            vector_weight=0.7,
            keyword_weight=0.3,
            reranker=reranker,
            first_stage_limit=args.first_stage_limit,
        )
        retriever.set_embedder(embedder)

        # Run evaluation
        logger.info(f"Running evaluation with top_k={args.top_k}...")
        results = await run_evaluation(pool, retriever, test_cases, top_k=args.top_k)

        # Capture git info for experiment tracking
        git_info = get_git_info()

        # Add metadata with run tracking
        results["metadata"] = {
            # Run tracking
            "run_label": args.label,
            "run_description": args.description,
            "git_commit": git_info.commit,
            "git_branch": git_info.branch,
            # Evaluation info
            "evaluation_dataset": str(eval_path),
            "timestamp": datetime.now().isoformat(),
            "top_k": args.top_k,
            "embedding_coverage_pct": float(coverage["pct_complete"]),  # Convert Decimal to float
            "retriever_config": {
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "reranking_enabled": args.with_reranking,
                "first_stage_limit": args.first_stage_limit if args.with_reranking else None,
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
                if args.with_reranking
                else None,
            },
        }

        # Print results
        print("\n" + "=" * 60)
        print("RAG RETRIEVAL EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nRun Label: {args.label}")
        if args.description:
            print(f"Description: {args.description}")
        if git_info.commit:
            print(f"Git: {git_info.commit} ({git_info.branch or 'detached'})")
        print(f"\nOverall Metrics (top-{args.top_k}):")
        for metric, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        print("\nBy Category:")
        for category, metrics in results["by_category"].items():
            print(f"\n  {category} (n={metrics['count']}):")
            print(f"    Recall@{args.top_k}: {metrics[f'recall@{args.top_k}']:.4f}")
            print(f"    MRR@{args.top_k}: {metrics[f'mrr@{args.top_k}']:.4f}")
            print(f"    NDCG@{args.top_k}: {metrics[f'ndcg@{args.top_k}']:.4f}")
            print(f"    Hit Rate: {metrics['hit_rate']:.4f}")

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    finally:
        await pool.close()


if __name__ == "__main__":
    # Deprecation warning
    import warnings

    warnings.warn(
        "\n" + "=" * 70 + "\n"
        "DEPRECATED: This script does not integrate with observability stack.\n"
        "Results will NOT appear in Grafana dashboards or Prometheus metrics.\n"
        "\n"
        "Use the batch evaluation framework instead:\n"
        "  1. python scripts/load_rag_fixtures.py  # Load fixtures (one-time)\n"
        "  2. python -m src.evaluation.cli.main batch run --pack rag --tag rag --label YOUR_LABEL\n"
        "\n"
        "See docs/plans/rag-ingestion-improvements.md for details.\n" + "=" * 70,
        DeprecationWarning,
        stacklevel=1,
    )

    parser = argparse.ArgumentParser(description="[DEPRECATED] Run RAG retrieval evaluation")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label for this run (e.g., 'baseline-v1', 'contextual-chunking'). Required for tracking.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional description for this run",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="tests/fixtures/rag/sec_evaluation_set.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: results/rag_{label}.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K for retrieval metrics (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even with low embedding coverage",
    )
    parser.add_argument(
        "--with-reranking",
        action="store_true",
        help="Enable cross-encoder reranking (two-stage retrieval)",
    )
    parser.add_argument(
        "--first-stage-limit",
        type=int,
        default=50,
        help="Number of candidates for first stage before reranking (default: 50)",
    )

    args = parser.parse_args()

    # Auto-generate output path from label if not specified
    if args.output is None:
        args.output = f"results/rag_{args.label}.json"

    asyncio.run(main(args))
