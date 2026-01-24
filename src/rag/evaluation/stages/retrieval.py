"""
Retrieval Stage

Evaluates retrieval quality using Information Retrieval metrics.
Requires ground truth labels (expected relevant documents).

Metrics:
- Recall@k: What fraction of relevant docs are in top-k?
- Precision@k: What fraction of top-k are relevant?
- MRR (Mean Reciprocal Rank): How high is the first relevant doc?
- NDCG (Normalized DCG): Are relevant docs ranked higher?
- Hit Rate: Is at least one relevant doc retrieved?
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.evalkit.contracts import StageResult, TestCase

if TYPE_CHECKING:
    from src.evalkit.contracts import EvalContext


@dataclass
class RetrievalStage:
    """
    Stage 1: Retrieval Quality Evaluation

    Computes IR metrics comparing retrieved documents against ground truth.
    Reference-based: requires expected['relevant_docs'] in test case.

    Skips with score=1.0 if no ground truth is available.
    """

    name: str = field(default="retrieval", init=False)
    is_gate: bool = field(default=False, init=False)

    # Configurable thresholds
    pass_threshold: float = 0.5
    k_values: tuple[int, ...] = (5, 10)

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None,
    ) -> StageResult:
        """
        Evaluate retrieval quality.

        Args:
            test_case: Test case with expected['relevant_docs']
            system_output: System output with 'retrieved_doc_ids' or 'retrieved_chunks'
            context: Evaluation context

        Returns:
            StageResult with IR metrics
        """
        start_time = time.perf_counter()

        # Get expected relevant documents
        expected_docs = test_case.expected.get("relevant_docs", [])
        if isinstance(expected_docs, str):
            expected_docs = [expected_docs]

        # Get retrieved documents (support multiple field names)
        retrieved_docs = (
            system_output.get("retrieved_doc_ids")
            or system_output.get("retrieved_chunks")
            or system_output.get("sources")
            or []
        )

        # Handle dict format (extract IDs)
        if retrieved_docs and isinstance(retrieved_docs[0], dict):
            retrieved_docs = [
                doc.get("id") or doc.get("doc_id") or doc.get("chunk_id", str(i))
                for i, doc in enumerate(retrieved_docs)
            ]

        # Skip if no ground truth
        if not expected_docs:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=1.0,
                reason="Skipped - no ground truth labels",
                metrics={"skipped": True, "reason": "no_ground_truth"},
                duration_ms=duration_ms,
            )

        # Compute metrics
        metrics: dict[str, Any] = {}

        # Recall@k and Precision@k for each k
        for k in self.k_values:
            metrics[f"recall_at_{k}"] = self._recall_at_k(expected_docs, retrieved_docs, k)
            metrics[f"precision_at_{k}"] = self._precision_at_k(expected_docs, retrieved_docs, k)

        # Other metrics
        metrics["mrr"] = self._mrr(expected_docs, retrieved_docs)
        metrics["ndcg_at_10"] = self._ndcg_at_k(expected_docs, retrieved_docs, k=10)
        metrics["hit_rate"] = self._hit_rate(expected_docs, retrieved_docs)
        metrics["num_expected"] = len(expected_docs)
        metrics["num_retrieved"] = len(retrieved_docs)

        # Compute composite score (weighted average of key metrics)
        score = (
            metrics.get("recall_at_5", 0.0) * 0.3
            + metrics.get("precision_at_5", 0.0) * 0.2
            + metrics["mrr"] * 0.3
            + metrics["hit_rate"] * 0.2
        )

        passed = score >= self.pass_threshold
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        reason = self._build_reason(metrics, passed)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=reason,
            metrics=metrics,
            artifacts={
                "expected_docs": expected_docs[:10],  # Limit for storage
                "retrieved_docs": retrieved_docs[:10],
            },
            duration_ms=duration_ms,
        )

    def _recall_at_k(self, expected: list, retrieved: list, k: int) -> float:
        """Compute recall@k: fraction of relevant docs in top-k."""
        if not expected:
            return 1.0
        retrieved_set = set(retrieved[:k])
        hits = len(set(expected) & retrieved_set)
        return hits / len(expected)

    def _precision_at_k(self, expected: list, retrieved: list, k: int) -> float:
        """Compute precision@k: fraction of top-k that are relevant."""
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        expected_set = set(expected)
        hits = sum(1 for doc in top_k if doc in expected_set)
        return hits / len(top_k)

    def _mrr(self, expected: list, retrieved: list) -> float:
        """Compute Mean Reciprocal Rank: 1/rank of first relevant doc."""
        expected_set = set(expected)
        for i, doc_id in enumerate(retrieved):
            if doc_id in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    def _hit_rate(self, expected: list, retrieved: list) -> float:
        """Compute hit rate: 1 if any relevant doc retrieved, else 0."""
        expected_set = set(expected)
        return 1.0 if any(doc in expected_set for doc in retrieved) else 0.0

    def _ndcg_at_k(self, expected: list, retrieved: list, k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.

        DCG = sum(rel_i / log2(i + 1)) for i in 1..k
        IDCG = DCG for ideal ranking (all relevant docs first)
        NDCG = DCG / IDCG
        """
        expected_set = set(expected)
        top_k = retrieved[:k]

        if not top_k or not expected_set:
            return 0.0

        # DCG: relevance = 1 if doc is relevant, 0 otherwise
        dcg = 0.0
        for i, doc in enumerate(top_k):
            rel = 1.0 if doc in expected_set else 0.0
            dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0

        # IDCG: all relevant docs at top
        num_relevant_in_k = min(len(expected_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant_in_k))

        return dcg / idcg if idcg > 0 else 0.0

    def _build_reason(self, metrics: dict, passed: bool) -> str:
        """Build human-readable explanation."""
        recall = metrics.get("recall_at_5", 0.0)
        mrr = metrics.get("mrr", 0.0)
        hit_rate = metrics.get("hit_rate", 0.0)

        if passed:
            return f"Retrieval quality acceptable: recall@5={recall:.2f}, MRR={mrr:.2f}"
        else:
            issues = []
            if recall < 0.5:
                issues.append(f"low recall@5 ({recall:.2f})")
            if mrr < 0.3:
                issues.append(f"low MRR ({mrr:.2f})")
            if hit_rate == 0:
                issues.append("no relevant docs retrieved")
            return f"Retrieval quality below threshold: {', '.join(issues)}"
