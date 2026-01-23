"""
Unit tests for RetrievalStage.

Tests IR metrics computation: recall@k, precision@k, MRR, NDCG, hit rate.
"""

import pytest

from src.contracts import TestCase
from src.evaluation.packs.rag.stages import RetrievalStage


@pytest.fixture
def stage():
    """Create a RetrievalStage instance."""
    return RetrievalStage()


class TestRetrievalMetrics:
    """Tests for individual metric calculations."""

    def test_recall_at_k_perfect(self, stage):
        """All relevant docs in top-k."""
        expected = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        assert stage._recall_at_k(expected, retrieved, k=5) == 1.0

    def test_recall_at_k_partial(self, stage):
        """Some relevant docs in top-k."""
        expected = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc5"]
        assert stage._recall_at_k(expected, retrieved, k=5) == pytest.approx(1 / 3)

    def test_recall_at_k_none(self, stage):
        """No relevant docs in top-k."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5"]
        assert stage._recall_at_k(expected, retrieved, k=5) == 0.0

    def test_recall_at_k_empty_expected(self, stage):
        """No expected docs (trivially satisfied)."""
        expected = []
        retrieved = ["doc1", "doc2"]
        assert stage._recall_at_k(expected, retrieved, k=5) == 1.0

    def test_precision_at_k_perfect(self, stage):
        """All top-k docs are relevant."""
        expected = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3"]
        assert stage._precision_at_k(expected, retrieved, k=3) == 1.0

    def test_precision_at_k_partial(self, stage):
        """Some top-k docs are relevant."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc1", "doc3", "doc4", "doc5", "doc2"]
        assert stage._precision_at_k(expected, retrieved, k=5) == pytest.approx(2 / 5)

    def test_precision_at_k_none(self, stage):
        """No top-k docs are relevant."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5"]
        assert stage._precision_at_k(expected, retrieved, k=3) == 0.0

    def test_precision_at_k_empty_retrieved(self, stage):
        """No retrieved docs."""
        expected = ["doc1", "doc2"]
        retrieved = []
        assert stage._precision_at_k(expected, retrieved, k=5) == 0.0

    def test_mrr_first_position(self, stage):
        """First relevant doc at position 1."""
        expected = ["doc1"]
        retrieved = ["doc1", "doc2", "doc3"]
        assert stage._mrr(expected, retrieved) == 1.0

    def test_mrr_second_position(self, stage):
        """First relevant doc at position 2."""
        expected = ["doc1"]
        retrieved = ["doc2", "doc1", "doc3"]
        assert stage._mrr(expected, retrieved) == 0.5

    def test_mrr_third_position(self, stage):
        """First relevant doc at position 3."""
        expected = ["doc1"]
        retrieved = ["doc2", "doc3", "doc1"]
        assert stage._mrr(expected, retrieved) == pytest.approx(1 / 3)

    def test_mrr_not_found(self, stage):
        """Relevant doc not in retrieved list."""
        expected = ["doc1"]
        retrieved = ["doc2", "doc3", "doc4"]
        assert stage._mrr(expected, retrieved) == 0.0

    def test_mrr_multiple_relevant(self, stage):
        """Multiple relevant docs - returns first."""
        expected = ["doc1", "doc3"]
        retrieved = ["doc2", "doc3", "doc1"]
        assert stage._mrr(expected, retrieved) == 0.5  # doc3 at position 2

    def test_hit_rate_found(self, stage):
        """At least one relevant doc retrieved."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc3", "doc1", "doc4"]
        assert stage._hit_rate(expected, retrieved) == 1.0

    def test_hit_rate_not_found(self, stage):
        """No relevant doc retrieved."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5"]
        assert stage._hit_rate(expected, retrieved) == 0.0

    def test_ndcg_perfect_ranking(self, stage):
        """All relevant docs ranked first."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        # DCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
        # IDCG = same since perfect ordering
        assert stage._ndcg_at_k(expected, retrieved, k=5) == 1.0

    def test_ndcg_reversed_ranking(self, stage):
        """Relevant docs at bottom."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5", "doc1", "doc2"]
        # DCG = 1/log2(5) + 1/log2(6) < IDCG = 1/log2(2) + 1/log2(3)
        result = stage._ndcg_at_k(expected, retrieved, k=5)
        assert 0.0 < result < 1.0

    def test_ndcg_empty(self, stage):
        """Empty lists."""
        assert stage._ndcg_at_k([], ["doc1"], k=5) == 0.0
        assert stage._ndcg_at_k(["doc1"], [], k=5) == 0.0


class TestRetrievalStageEvaluate:
    """Tests for the evaluate() method."""

    @pytest.mark.asyncio
    async def test_perfect_retrieval(self, stage):
        """All relevant docs retrieved at top."""
        test_case = TestCase(
            id="test-1",
            input={"query": "test query"},
            expected={"relevant_docs": ["doc1", "doc2", "doc3"]},
        )
        system_output = {"retrieved_doc_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"]}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score >= 0.8
        assert result.metrics["recall_at_5"] == 1.0
        assert result.metrics["mrr"] == 1.0
        assert result.metrics["hit_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_no_ground_truth_skips(self, stage):
        """Skip evaluation when no ground truth."""
        test_case = TestCase(
            id="test-2",
            input={"query": "test query"},
            expected={},  # No relevant_docs
        )
        system_output = {"retrieved_doc_ids": ["doc1", "doc2"]}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_poor_retrieval(self, stage):
        """No relevant docs retrieved."""
        test_case = TestCase(
            id="test-3",
            input={"query": "test query"},
            expected={"relevant_docs": ["doc1", "doc2"]},
        )
        system_output = {"retrieved_doc_ids": ["doc3", "doc4", "doc5"]}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.score < 0.5
        assert result.metrics["recall_at_5"] == 0.0
        assert result.metrics["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_single_relevant_doc_string(self, stage):
        """Single relevant doc as string (not list)."""
        test_case = TestCase(
            id="test-4",
            input={"query": "test query"},
            expected={"relevant_docs": "doc1"},  # String, not list
        )
        system_output = {"retrieved_doc_ids": ["doc1", "doc2"]}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.metrics["recall_at_5"] == 1.0

    @pytest.mark.asyncio
    async def test_retrieved_chunks_dict_format(self, stage):
        """Retrieved chunks as list of dicts with id field."""
        test_case = TestCase(
            id="test-5",
            input={"query": "test query"},
            expected={"relevant_docs": ["chunk-1", "chunk-2"]},
        )
        system_output = {
            "retrieved_chunks": [
                {"id": "chunk-1", "text": "content 1"},
                {"id": "chunk-2", "text": "content 2"},
                {"id": "chunk-3", "text": "content 3"},
            ]
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.metrics["recall_at_5"] == 1.0

    @pytest.mark.asyncio
    async def test_sources_field(self, stage):
        """Use 'sources' field for retrieved docs."""
        test_case = TestCase(
            id="test-6",
            input={"query": "test query"},
            expected={"relevant_docs": ["src-1"]},
        )
        system_output = {
            "sources": [
                {"doc_id": "src-1", "text": "content"},
                {"doc_id": "src-2", "text": "content"},
            ]
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_partial_retrieval(self, stage):
        """Some but not all relevant docs retrieved."""
        test_case = TestCase(
            id="test-7",
            input={"query": "test query"},
            expected={"relevant_docs": ["doc1", "doc2", "doc3", "doc4"]},
        )
        system_output = {"retrieved_doc_ids": ["doc1", "doc5", "doc2", "doc6", "doc7"]}

        result = await stage.evaluate(test_case, system_output, None)

        # 2 out of 4 relevant docs = 50% recall
        assert result.metrics["recall_at_5"] == 0.5

    @pytest.mark.asyncio
    async def test_metrics_contain_expected_keys(self, stage):
        """Verify all expected metrics are present."""
        test_case = TestCase(
            id="test-8",
            input={"query": "test query"},
            expected={"relevant_docs": ["doc1"]},
        )
        system_output = {"retrieved_doc_ids": ["doc1", "doc2"]}

        result = await stage.evaluate(test_case, system_output, None)

        expected_keys = [
            "recall_at_5",
            "recall_at_10",
            "precision_at_5",
            "precision_at_10",
            "mrr",
            "ndcg_at_10",
            "hit_rate",
            "num_expected",
            "num_retrieved",
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing metric: {key}"

    @pytest.mark.asyncio
    async def test_empty_retrieved(self, stage):
        """No documents retrieved."""
        test_case = TestCase(
            id="test-9",
            input={"query": "test query"},
            expected={"relevant_docs": ["doc1"]},
        )
        system_output = {"retrieved_doc_ids": []}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.metrics["recall_at_5"] == 0.0
        assert result.metrics["num_retrieved"] == 0


class TestRetrievalStageConfiguration:
    """Tests for stage configuration."""

    @pytest.mark.asyncio
    async def test_custom_pass_threshold(self):
        """Stage respects custom pass threshold."""
        # Low threshold - should pass
        low_stage = RetrievalStage(pass_threshold=0.2)

        test_case = TestCase(
            id="test-10",
            input={"query": "test"},
            expected={"relevant_docs": ["doc1", "doc2", "doc3"]},
        )
        system_output = {"retrieved_doc_ids": ["doc1", "doc4", "doc5"]}  # 1/3 relevant

        result = await low_stage.evaluate(test_case, system_output, None)
        assert result.passed is True  # Low threshold

        # High threshold - should fail
        high_stage = RetrievalStage(pass_threshold=0.9)
        result = await high_stage.evaluate(test_case, system_output, None)
        assert result.passed is False  # High threshold

    def test_stage_properties(self, stage):
        """Verify stage properties."""
        assert stage.name == "retrieval"
        assert stage.is_gate is False
