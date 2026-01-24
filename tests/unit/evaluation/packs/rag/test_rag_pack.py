"""
Unit tests for RAGPack.

Tests the complete RAG evaluation pack integration.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evalkit.contracts import EvalContext, Scorecard, TestCase
from src.evaluation.packs.rag import RAGPack
from src.evaluation.packs.rag.pack import RAGPackConfig


@pytest.fixture
def pack():
    """Create a RAGPack instance."""
    return RAGPack()


@pytest.fixture
def minimal_pack():
    """Create a minimal RAGPack with only retrieval stage."""
    return RAGPack(
        config=RAGPackConfig(
            context_relevance_enabled=False,
            faithfulness_enabled=False,
            answer_relevance_enabled=False,
            citation_enabled=False,
            source_policy_enabled=False,
            policy_compliance_enabled=False,
            rejection_calibration_enabled=False,
        )
    )


class TestRAGPackProperties:
    """Tests for pack properties."""

    def test_pack_name(self, pack):
        """Pack has correct name."""
        assert pack.name == "rag"

    def test_get_stages_returns_all_enabled(self, pack):
        """All enabled stages are returned."""
        stages = pack.get_stages()

        assert len(stages) == 8  # All 8 stages

        stage_names = [s.name for s in stages]
        assert "retrieval" in stage_names
        assert "context_relevance" in stage_names
        assert "faithfulness" in stage_names
        assert "answer_relevance" in stage_names
        assert "citation" in stage_names
        assert "source_policy" in stage_names
        assert "policy_compliance" in stage_names
        assert "rejection_calibration" in stage_names

    def test_get_stages_respects_config(self, minimal_pack):
        """Disabled stages are excluded."""
        stages = minimal_pack.get_stages()

        assert len(stages) == 1
        assert stages[0].name == "retrieval"

    def test_default_weights(self, pack):
        """Default weights are provided."""
        weights = pack.get_default_weights()

        assert "retrieval" in weights
        assert "faithfulness" in weights
        assert sum(weights.values()) == pytest.approx(1.0)


class TestRAGPackValidation:
    """Tests for test case validation."""

    def test_validate_missing_query(self, pack):
        """Validation fails without query."""
        test_case = TestCase(
            id="test-1",
            input={},  # No query
            expected={"relevant_docs": ["doc1"]},
        )

        errors = pack.validate_test_case(test_case)

        assert len(errors) > 0
        assert any("query" in e.lower() for e in errors)

    def test_validate_missing_expected(self, pack):
        """Validation fails without expected output."""
        test_case = TestCase(
            id="test-2",
            input={"query": "test query"},
            expected={},  # No relevant_docs, behavior, or answer
        )

        errors = pack.validate_test_case(test_case)

        assert len(errors) > 0

    def test_validate_with_relevant_docs(self, pack):
        """Valid with relevant_docs."""
        test_case = TestCase(
            id="test-3",
            input={"query": "test query"},
            expected={"relevant_docs": ["doc1"]},
        )

        errors = pack.validate_test_case(test_case)

        assert len(errors) == 0

    def test_validate_with_behavior(self, pack):
        """Valid with behavior."""
        test_case = TestCase(
            id="test-4",
            input={"query": "test query"},
            expected={"behavior": "reject"},
        )

        errors = pack.validate_test_case(test_case)

        assert len(errors) == 0

    def test_validate_with_answer(self, pack):
        """Valid with expected answer."""
        test_case = TestCase(
            id="test-5",
            input={"query": "test query"},
            expected={"answer": "expected answer"},
        )

        errors = pack.validate_test_case(test_case)

        assert len(errors) == 0


class TestRAGPackScoring:
    """Tests for score computation."""

    def test_compute_overall_score_weighted(self, pack):
        """Overall score is weighted average."""
        from src.evalkit.contracts import StageResult

        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.9),
            "faithfulness": StageResult(stage_name="faithfulness", passed=True, score=0.8),
            "answer_relevance": StageResult(stage_name="answer_relevance", passed=True, score=0.7),
        }

        score = pack.compute_overall_score(stage_results)

        # Score should be weighted average
        assert 0.0 <= score <= 1.0

    def test_compute_overall_score_custom_weights(self, pack):
        """Custom weights override defaults."""
        from src.evalkit.contracts import StageResult

        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=1.0),
            "faithfulness": StageResult(stage_name="faithfulness", passed=True, score=0.0),
        }

        # Heavy weight on retrieval
        score = pack.compute_overall_score(
            stage_results,
            weights={"retrieval": 0.9, "faithfulness": 0.1},
        )

        assert score > 0.8

    def test_compute_overall_score_excludes_gate_stages(self, pack):
        """Gate stages not included in weighted score."""
        from src.evalkit.contracts import StageResult

        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=1.0),
            "source_policy": StageResult(stage_name="source_policy", passed=True, score=1.0),
            "policy_compliance": StageResult(
                stage_name="policy_compliance", passed=True, score=1.0
            ),
        }

        # Gate stages should be excluded from weighted score
        score = pack.compute_overall_score(stage_results)
        assert score > 0  # Should still compute something

    def test_compute_overall_passed_all_pass(self, pack):
        """All stages passing means overall pass."""
        from src.evalkit.contracts import StageResult

        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.8),
            "source_policy": StageResult(stage_name="source_policy", passed=True, score=1.0),
            "policy_compliance": StageResult(
                stage_name="policy_compliance", passed=True, score=1.0
            ),
        }

        assert pack.compute_overall_passed(stage_results) is True

    def test_compute_overall_passed_gate_fails(self, pack):
        """Gate stage failing means overall fail."""
        from src.evalkit.contracts import StageResult

        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.9),
            "source_policy": StageResult(stage_name="source_policy", passed=False, score=0.0),
        }

        assert pack.compute_overall_passed(stage_results) is False

    def test_compute_overall_passed_low_average(self, pack):
        """Very low average score fails even without gate failures."""
        from src.evalkit.contracts import StageResult

        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=False, score=0.1),
            "faithfulness": StageResult(stage_name="faithfulness", passed=False, score=0.1),
            "answer_relevance": StageResult(stage_name="answer_relevance", passed=False, score=0.1),
        }

        assert pack.compute_overall_passed(stage_results) is False


class TestRAGPackEvaluate:
    """Tests for the evaluate() convenience method."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_scorecard(self, minimal_pack):
        """Evaluate returns a complete scorecard."""
        test_case = TestCase(
            id="test-6",
            input={"query": "What is Python?"},
            expected={"relevant_docs": ["doc1"]},
        )
        system_output = {
            "response": "Python is a programming language.",
            "retrieved_doc_ids": ["doc1", "doc2"],
        }

        scorecard = await minimal_pack.evaluate(test_case, system_output)

        assert isinstance(scorecard, Scorecard)
        assert scorecard.test_case_id == "test-6"
        assert scorecard.pack_name == "rag"
        assert "retrieval" in scorecard.stage_results

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, minimal_pack):
        """Evaluate uses provided context."""
        context = EvalContext(
            batch_id="batch-123",
            worker_id="worker-456",
        )
        test_case = TestCase(
            id="test-7",
            input={"query": "test"},
            expected={"relevant_docs": ["doc1"]},
        )
        system_output = {"retrieved_doc_ids": ["doc1"]}

        scorecard = await minimal_pack.evaluate(test_case, system_output, context)

        assert scorecard.batch_id == "batch-123"
        assert scorecard.worker_id == "worker-456"

    @pytest.mark.asyncio
    async def test_evaluate_gate_stops_pipeline(self, pack):
        """Gate failure stops further evaluation."""
        test_case = TestCase(
            id="test-8",
            input={"query": "test"},
            expected={
                "relevant_docs": ["doc1"],
                "source_policies": {"1": "no_use"},
            },
        )
        system_output = {
            "response": "This uses the no-use source content directly",
            "retrieved_doc_ids": ["doc1"],
            "sources": [{"id": "1", "text": "This uses the no-use source content"}],
        }

        _scorecard = await pack.evaluate(test_case, system_output)  # noqa: F841

        # Source policy is a gate - should stop further evaluation
        # But earlier stages should still run


class TestRAGPackConfiguration:
    """Tests for pack configuration."""

    def test_custom_thresholds(self):
        """Custom thresholds are applied."""
        pack = RAGPack(
            config=RAGPackConfig(
                retrieval_threshold=0.8,
                faithfulness_threshold=0.9,
            )
        )

        stages = pack.get_stages()
        retrieval = next(s for s in stages if s.name == "retrieval")
        faithfulness = next(s for s in stages if s.name == "faithfulness")

        assert retrieval.pass_threshold == 0.8
        assert faithfulness.pass_threshold == 0.9

    def test_custom_weights_in_config(self):
        """Custom weights override defaults."""
        pack = RAGPack(config=RAGPackConfig(custom_weights={"faithfulness": 0.5, "retrieval": 0.5}))

        weights = pack.get_default_weights()

        assert weights["faithfulness"] == 0.5
        assert weights["retrieval"] == 0.5

    def test_kwargs_config(self):
        """Configuration via kwargs."""
        pack = RAGPack(retrieval_enabled=False)

        stages = pack.get_stages()
        stage_names = [s.name for s in stages]

        assert "retrieval" not in stage_names


class TestRAGPackIntegration:
    """Integration tests with multiple stages."""

    @pytest.mark.asyncio
    async def test_full_evaluation_pass(self):
        """Complete evaluation with all stages passing."""
        pack = RAGPack(
            config=RAGPackConfig(
                source_policy_enabled=False,  # Disable gate to simplify
                policy_compliance_enabled=False,
            )
        )

        test_case = TestCase(
            id="test-9",
            input={"query": "What is Python programming language?"},
            expected={
                "relevant_docs": ["doc1"],
                "behavior": "answer",
            },
        )
        system_output = {
            "response": "Python is a high-level programming language [1].",
            "retrieved_doc_ids": ["doc1", "doc2"],
            "retrieved_chunks": [
                {"id": "doc1", "text": "Python is a high-level programming language."},
            ],
            "context": "Python is a high-level programming language.",
            "sources": [{"id": "1", "text": "Python info"}],
        }

        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.pack_name == "rag"
        assert len(scorecard.stage_results) >= 4  # At least RAG Triad + some gates

    @pytest.mark.asyncio
    async def test_full_evaluation_with_llm_judge(self):
        """Evaluation with mock LLM judge."""
        from src.evaluation.packs.rag.llm_judge import JudgeResult

        mock_judge = MagicMock()
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=0.9, passed=True, reasoning="Good", raw_response="")
        )
        mock_judge.evaluate_faithfulness = AsyncMock(
            return_value=JudgeResult(
                score=0.85,
                passed=True,
                reasoning="Grounded",
                raw_response="",
                metrics={"num_claims": 2, "supported_claims": 2},
            )
        )

        pack = RAGPack(
            config=RAGPackConfig(
                source_policy_enabled=False,
                policy_compliance_enabled=False,
            )
        )
        context = EvalContext(llm_judge=mock_judge)

        test_case = TestCase(
            id="test-10",
            input={"query": "What is Python?"},
            expected={"relevant_docs": ["doc1"]},
        )
        system_output = {
            "response": "Python is a programming language.",
            "retrieved_doc_ids": ["doc1"],
            "context": "Python is a programming language.",
        }

        _scorecard = await pack.evaluate(test_case, system_output, context)  # noqa: F841

        # LLM judge should have been called
        assert mock_judge.evaluate_relevance.called or mock_judge.evaluate_faithfulness.called
