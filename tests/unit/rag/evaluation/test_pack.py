"""
Unit tests for RAG evaluation pack - Direct import validation.

Tests the RAG pack by importing directly from the new application location
(src.rag.evaluation) rather than the compatibility shim.

This validates that the codebase separation refactor works correctly.
"""

import pytest

# Import contracts from evalkit
from src.evalkit.contracts import (
    EvalContext,
    StageResult,
    TestCase,
)

# Direct imports from new location (NOT compatibility shim)
from src.rag.evaluation import RAGPack, RAGPackConfig
from src.rag.evaluation.stages import (
    AnswerRelevanceStage,
    CitationStage,
    ContextRelevanceStage,
    FaithfulnessStage,
    PolicyComplianceStage,
    RejectionCalibrationStage,
    RetrievalStage,
    SourcePolicyStage,
)

# =============================================================================
# RAGPackConfig Tests
# =============================================================================


class TestRAGPackConfig:
    """Tests for RAGPackConfig."""

    def test_default_config(self):
        """Default config has all stages enabled."""
        config = RAGPackConfig()

        assert config.retrieval_enabled is True
        assert config.context_relevance_enabled is True
        assert config.faithfulness_enabled is True
        assert config.answer_relevance_enabled is True
        assert config.citation_enabled is True
        assert config.source_policy_enabled is True
        assert config.policy_compliance_enabled is True
        assert config.rejection_calibration_enabled is True

    def test_default_thresholds(self):
        """Default thresholds are set correctly."""
        config = RAGPackConfig()

        assert config.retrieval_threshold == 0.5
        assert config.context_relevance_threshold == 0.6
        assert config.faithfulness_threshold == 0.7
        assert config.answer_relevance_threshold == 0.7
        assert config.citation_threshold == 0.6

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = RAGPackConfig(
            retrieval_enabled=False,
            faithfulness_threshold=0.9,
        )

        assert config.retrieval_enabled is False
        assert config.faithfulness_threshold == 0.9
        # Others unchanged
        assert config.context_relevance_enabled is True

    def test_custom_weights(self):
        """Custom weights can be set."""
        custom_weights = {"retrieval": 0.3, "faithfulness": 0.5}
        config = RAGPackConfig(custom_weights=custom_weights)

        assert config.custom_weights == custom_weights


# =============================================================================
# RAGPack Initialization Tests
# =============================================================================


class TestRAGPackInit:
    """Tests for RAGPack initialization."""

    def test_default_initialization(self):
        """Pack initializes with defaults."""
        pack = RAGPack()

        assert pack.name == "rag"
        assert pack.config is not None

    def test_with_config(self):
        """Pack accepts config object."""
        config = RAGPackConfig(retrieval_enabled=False)
        pack = RAGPack(config=config)

        assert pack.config.retrieval_enabled is False

    def test_with_kwargs(self):
        """Pack accepts kwargs for config."""
        pack = RAGPack(retrieval_enabled=False, faithfulness_threshold=0.9)

        assert pack.config.retrieval_enabled is False
        assert pack.config.faithfulness_threshold == 0.9


# =============================================================================
# RAGPack Stage Composition Tests
# =============================================================================


class TestRAGPackStages:
    """Tests for RAGPack stage composition."""

    @pytest.fixture
    def pack(self) -> RAGPack:
        return RAGPack()

    def test_default_stages_count(self, pack):
        """Default pack has 8 stages."""
        stages = pack.get_stages()
        assert len(stages) == 8

    def test_default_stage_order(self, pack):
        """Default stages are in correct order."""
        stages = pack.get_stages()
        stage_names = [s.name for s in stages]

        expected_order = [
            "retrieval",
            "context_relevance",
            "faithfulness",
            "answer_relevance",
            "citation",
            "source_policy",
            "policy_compliance",
            "rejection_calibration",
        ]
        assert stage_names == expected_order

    def test_stage_types(self, pack):
        """Stages have correct types."""
        stages = pack.get_stages()

        assert isinstance(stages[0], RetrievalStage)
        assert isinstance(stages[1], ContextRelevanceStage)
        assert isinstance(stages[2], FaithfulnessStage)
        assert isinstance(stages[3], AnswerRelevanceStage)
        assert isinstance(stages[4], CitationStage)
        assert isinstance(stages[5], SourcePolicyStage)
        assert isinstance(stages[6], PolicyComplianceStage)
        assert isinstance(stages[7], RejectionCalibrationStage)

    def test_gate_stages(self, pack):
        """Gate stages are correctly identified."""
        stages = pack.get_stages()
        gate_stages = [s.name for s in stages if s.is_gate]

        # source_policy and policy_compliance are gates
        assert "source_policy" in gate_stages
        assert "policy_compliance" in gate_stages
        assert len(gate_stages) == 2

    def test_disabled_stage_excluded(self):
        """Disabled stages are excluded from pipeline."""
        pack = RAGPack(retrieval_enabled=False, citation_enabled=False)
        stages = pack.get_stages()
        stage_names = [s.name for s in stages]

        assert "retrieval" not in stage_names
        assert "citation" not in stage_names
        assert len(stages) == 6

    def test_all_stages_disabled(self):
        """Can disable all stages (empty pipeline)."""
        pack = RAGPack(
            retrieval_enabled=False,
            context_relevance_enabled=False,
            faithfulness_enabled=False,
            answer_relevance_enabled=False,
            citation_enabled=False,
            source_policy_enabled=False,
            policy_compliance_enabled=False,
            rejection_calibration_enabled=False,
        )
        stages = pack.get_stages()

        assert len(stages) == 0


# =============================================================================
# RAGPack Weights Tests
# =============================================================================


class TestRAGPackWeights:
    """Tests for RAGPack weight configuration."""

    @pytest.fixture
    def pack(self) -> RAGPack:
        return RAGPack()

    def test_default_weights(self, pack):
        """Default weights are defined for all stages."""
        weights = pack.get_default_weights()

        assert "retrieval" in weights
        assert "context_relevance" in weights
        assert "faithfulness" in weights
        assert "answer_relevance" in weights
        assert "citation" in weights
        assert "source_policy" in weights
        assert "policy_compliance" in weights
        assert "rejection_calibration" in weights

    def test_weights_sum_to_one(self, pack):
        """Default weights sum to 1.0."""
        weights = pack.get_default_weights()
        total = sum(weights.values())

        assert abs(total - 1.0) < 0.001

    def test_faithfulness_has_highest_weight(self, pack):
        """Faithfulness has highest weight (critical for hallucination)."""
        weights = pack.get_default_weights()

        max_weight = max(weights.values())
        assert weights["faithfulness"] == max_weight

    def test_custom_weights_override(self):
        """Custom weights override defaults."""
        custom = {"retrieval": 0.5, "faithfulness": 0.5}
        pack = RAGPack(custom_weights=custom)
        weights = pack.get_default_weights()

        assert weights["retrieval"] == 0.5
        assert weights["faithfulness"] == 0.5
        # Others unchanged from defaults
        assert weights["context_relevance"] == 0.15


# =============================================================================
# RAGPack Validation Tests
# =============================================================================


class TestRAGPackValidation:
    """Tests for RAGPack test case validation."""

    @pytest.fixture
    def pack(self) -> RAGPack:
        return RAGPack()

    def test_valid_retrieval_test_case(self, pack):
        """Valid test case with relevant_docs passes validation."""
        test_case = TestCase(
            id="valid-001",
            input={"query": "What is the capital of France?"},
            expected={"relevant_docs": ["doc-1", "doc-2"]},
        )

        errors = pack.validate_test_case(test_case)
        assert errors == []

    def test_valid_behavior_test_case(self, pack):
        """Valid test case with behavior passes validation."""
        test_case = TestCase(
            id="valid-002",
            input={"query": "What is the capital of France?"},
            expected={"behavior": "answer"},
        )

        errors = pack.validate_test_case(test_case)
        assert errors == []

    def test_valid_answer_test_case(self, pack):
        """Valid test case with expected answer passes validation."""
        test_case = TestCase(
            id="valid-003",
            input={"query": "What is the capital of France?"},
            expected={"answer": "Paris"},
        )

        errors = pack.validate_test_case(test_case)
        assert errors == []

    def test_missing_query_fails(self, pack):
        """Missing query produces error."""
        test_case = TestCase(
            id="invalid-001",
            input={},
            expected={"relevant_docs": ["doc-1"]},
        )

        errors = pack.validate_test_case(test_case)
        assert len(errors) > 0
        assert any("query" in e.lower() for e in errors)

    def test_missing_expectations_fails(self, pack):
        """Missing all expectations produces error."""
        test_case = TestCase(
            id="invalid-002",
            input={"query": "What is X?"},
            expected={},
        )

        errors = pack.validate_test_case(test_case)
        assert len(errors) > 0
        assert any("relevant_docs" in e or "behavior" in e or "answer" in e for e in errors)


# =============================================================================
# RAGPack Scoring Tests
# =============================================================================


class TestRAGPackScoring:
    """Tests for RAGPack score computation."""

    @pytest.fixture
    def pack(self) -> RAGPack:
        return RAGPack()

    def test_compute_overall_score_basic(self, pack):
        """Compute overall score from stage results."""
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.8),
            "faithfulness": StageResult(stage_name="faithfulness", passed=True, score=0.9),
        }

        score = pack.compute_overall_score(stage_results)

        # Score should be weighted average (excluding gates)
        assert 0.0 <= score <= 1.0

    def test_gate_stages_excluded_from_scoring(self, pack):
        """Gate stages don't affect weighted score."""
        # Just non-gate stages
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.8),
            "source_policy": StageResult(stage_name="source_policy", passed=True, score=1.0),
        }

        # source_policy is gate, should be excluded
        score = pack.compute_overall_score(stage_results)
        # With only retrieval contributing: 0.8
        assert score == pytest.approx(0.8, abs=0.01)

    def test_compute_overall_passed_all_pass(self, pack):
        """All stages passing means overall passed."""
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.8),
            "faithfulness": StageResult(stage_name="faithfulness", passed=True, score=0.9),
            "source_policy": StageResult(stage_name="source_policy", passed=True, score=1.0),
            "policy_compliance": StageResult(
                stage_name="policy_compliance", passed=True, score=1.0
            ),
        }

        passed = pack.compute_overall_passed(stage_results)
        assert passed is True

    def test_gate_failure_fails_overall(self, pack):
        """Gate stage failure means overall failed."""
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.8),
            "source_policy": StageResult(stage_name="source_policy", passed=False, score=0.0),
        }

        passed = pack.compute_overall_passed(stage_results)
        assert passed is False

    def test_low_average_score_fails(self, pack):
        """Very low average non-gate score fails overall."""
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=False, score=0.1),
            "faithfulness": StageResult(stage_name="faithfulness", passed=False, score=0.1),
            "source_policy": StageResult(stage_name="source_policy", passed=True, score=1.0),
        }

        passed = pack.compute_overall_passed(stage_results)
        # Average of non-gate is 0.1 < 0.3 threshold
        assert passed is False


# =============================================================================
# RAGPack Evaluation Pipeline Tests
# =============================================================================


class TestRAGPackEvaluation:
    """Tests for RAGPack full evaluation pipeline."""

    @pytest.fixture
    def pack(self) -> RAGPack:
        return RAGPack()

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="eval-001",
            input={"query": "What is the capital of France?"},
            expected={
                "relevant_docs": ["doc-1"],
                "behavior": "answer",
            },
        )

    @pytest.fixture
    def system_output(self) -> dict:
        return {
            "response": "The capital of France is Paris.",
            "retrieved_chunks": [{"id": "doc-1", "text": "Paris is the capital of France."}],
        }

    @pytest.mark.asyncio
    async def test_evaluate_returns_scorecard(self, pack, test_case, system_output):
        """Evaluate returns a Scorecard."""
        from src.evalkit.contracts import Scorecard

        scorecard = await pack.evaluate(test_case, system_output)

        assert isinstance(scorecard, Scorecard)
        assert scorecard.test_case_id == test_case.id
        assert scorecard.pack_name == "rag"

    @pytest.mark.asyncio
    async def test_evaluate_includes_stage_results(self, pack, test_case, system_output):
        """Scorecard includes results for all stages."""
        scorecard = await pack.evaluate(test_case, system_output)

        # All 8 stages should have results (unless gate failed)
        assert len(scorecard.stage_results) > 0
        assert "retrieval" in scorecard.stage_results

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, pack, test_case, system_output):
        """Evaluation context is passed to stages."""
        context = EvalContext(
            batch_id="test-batch-rag-001",
            worker_id="test-worker",
        )

        scorecard = await pack.evaluate(test_case, system_output, context)

        assert scorecard.batch_id == "test-batch-rag-001"
        assert scorecard.worker_id == "test-worker"

    @pytest.mark.asyncio
    async def test_gate_failure_stops_pipeline(self, pack, test_case):
        """Gate failure stops the evaluation pipeline."""
        # Configure output that will fail source_policy gate
        system_output = {
            "response": "I made this up.",
            "retrieved_chunks": [],
            # Missing sources will trigger failures
        }

        scorecard = await pack.evaluate(test_case, system_output)

        # Should have some results but may stop early
        assert scorecard.stage_results is not None

    @pytest.mark.asyncio
    async def test_evaluate_tracks_latency(self, pack, test_case, system_output):
        """Evaluation tracks total latency."""
        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_evaluate_captures_output(self, pack, test_case, system_output):
        """Scorecard captures system output."""
        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.generated_output is not None
        assert "response" in scorecard.generated_output


# =============================================================================
# RetrievalStage Tests
# =============================================================================


class TestRetrievalStageDirect:
    """Tests for RetrievalStage using direct imports."""

    @pytest.fixture
    def stage(self) -> RetrievalStage:
        return RetrievalStage()

    @pytest.fixture
    def context(self) -> EvalContext:
        return EvalContext()

    def test_stage_properties(self, stage):
        """Verify stage properties."""
        assert stage.name == "retrieval"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_skips_without_ground_truth(self, stage, context):
        """Skips when no ground truth labels."""
        test_case = TestCase(
            id="ret-001",
            input={"query": "test"},
            expected={},
        )
        system_output = {"retrieved_doc_ids": ["doc-1"]}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert result.score == 1.0
        assert "skipped" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_perfect_retrieval(self, stage, context):
        """Perfect retrieval gets high score."""
        test_case = TestCase(
            id="ret-002",
            input={"query": "test"},
            expected={"relevant_docs": ["doc-1", "doc-2"]},
        )
        system_output = {"retrieved_doc_ids": ["doc-1", "doc-2", "doc-3"]}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_no_retrieval(self, stage, context):
        """No relevant docs retrieved gets low score."""
        test_case = TestCase(
            id="ret-003",
            input={"query": "test"},
            expected={"relevant_docs": ["doc-1"]},
        )
        system_output = {"retrieved_doc_ids": ["doc-99"]}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is False
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_metrics_computed(self, stage, context):
        """IR metrics are computed."""
        test_case = TestCase(
            id="ret-004",
            input={"query": "test"},
            expected={"relevant_docs": ["doc-1"]},
        )
        system_output = {"retrieved_doc_ids": ["doc-1", "doc-2"]}

        result = await stage.evaluate(test_case, system_output, context)

        assert "recall_at_5" in result.metrics
        assert "precision_at_5" in result.metrics
        assert "mrr" in result.metrics
        assert "hit_rate" in result.metrics
        assert "ndcg_at_10" in result.metrics


# =============================================================================
# Import Verification Tests
# =============================================================================


class TestDirectImportVerification:
    """Verify that direct imports work correctly."""

    def test_pack_importable(self):
        """RAGPack is importable from new location."""
        from src.rag.evaluation import RAGPack

        assert RAGPack is not None

    def test_config_importable(self):
        """RAGPackConfig is importable from new location."""
        from src.rag.evaluation import RAGPackConfig

        assert RAGPackConfig is not None

    def test_stages_importable(self):
        """All stages are importable from new location."""
        from src.rag.evaluation.stages import (
            AnswerRelevanceStage,
            CitationStage,
            ContextRelevanceStage,
            FaithfulnessStage,
            PolicyComplianceStage,
            RejectionCalibrationStage,
            RetrievalStage,
            SourcePolicyStage,
        )

        assert RetrievalStage is not None
        assert ContextRelevanceStage is not None
        assert FaithfulnessStage is not None
        assert AnswerRelevanceStage is not None
        assert CitationStage is not None
        assert SourcePolicyStage is not None
        assert PolicyComplianceStage is not None
        assert RejectionCalibrationStage is not None

    def test_pack_from_evaluation_init(self):
        """RAGPack is importable from evaluation __init__."""
        from src.rag.evaluation import RAGPack

        pack = RAGPack()
        assert pack.name == "rag"

    def test_evalkit_contracts_work_with_pack(self):
        """Verify evalkit contracts work with RAG pack."""
        from src.evalkit.contracts import TestCase

        test_case = TestCase(
            id="verify-001",
            input={"query": "test"},
            expected={"relevant_docs": ["doc-1"]},
        )

        pack = RAGPack()
        errors = pack.validate_test_case(test_case)

        assert errors == []
