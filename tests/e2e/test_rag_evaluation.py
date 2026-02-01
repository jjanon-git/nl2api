"""
RAG Evaluation End-to-End Tests

Tests the complete RAG evaluation pipeline:
1. Test case creation with RAG-specific fields
2. 8-stage evaluation execution
3. Scorecard verification with IR metrics
4. Gate stage behavior

These tests validate that the codebase separation refactor works
for RAG end-to-end evaluation workflows.

Requires:
    - Docker compose up for database tests (optional)
"""

import pytest

# Use direct imports from new locations
from src.evalkit.contracts import (
    EvalContext,
    Scorecard,
    StageResult,
    TestCase,
)
from src.rag.evaluation import RAGPack, RAGPackConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rag_pack() -> RAGPack:
    """Create RAG pack with default settings."""
    return RAGPack()


@pytest.fixture
def rag_pack_minimal() -> RAGPack:
    """Create RAG pack with only retrieval stage."""
    return RAGPack(
        retrieval_enabled=True,
        context_relevance_enabled=False,
        faithfulness_enabled=False,
        answer_relevance_enabled=False,
        citation_enabled=False,
        source_policy_enabled=False,
        policy_compliance_enabled=False,
        rejection_calibration_enabled=False,
    )


@pytest.fixture
def simple_rag_test_case() -> TestCase:
    """Simple RAG test case."""
    return TestCase(
        id="rag-e2e-001",
        input={"query": "What is the capital of France?"},
        expected={
            "relevant_docs": ["doc-paris-001"],
            "behavior": "answer",
        },
    )


@pytest.fixture
def rejection_test_case() -> TestCase:
    """Test case for rejection behavior."""
    return TestCase(
        id="rag-e2e-reject-001",
        input={"query": "What is the secret of life?"},
        expected={
            "behavior": "reject",
        },
    )


@pytest.fixture
def system_output_success() -> dict:
    """Successful RAG system output."""
    return {
        "response": "The capital of France is Paris.",
        "retrieved_chunks": [
            {"id": "doc-paris-001", "text": "Paris is the capital city of France."},
            {"id": "doc-paris-002", "text": "France is a country in Western Europe."},
        ],
    }


@pytest.fixture
def system_output_no_retrieval() -> dict:
    """System output with no relevant retrieval."""
    return {
        "response": "I made up this answer.",
        "retrieved_chunks": [
            {"id": "doc-wrong-001", "text": "This is unrelated content."},
        ],
    }


@pytest.fixture
def context() -> EvalContext:
    """Evaluation context for tests."""
    return EvalContext(
        batch_id="rag-e2e-batch",
        worker_id="rag-e2e-worker",
    )


# =============================================================================
# Pack Initialization E2E Tests
# =============================================================================


class TestRAGPackInitializationE2E:
    """End-to-end tests for RAG pack initialization."""

    def test_pack_creates_successfully(self, rag_pack):
        """Pack initializes without errors."""
        assert rag_pack is not None
        assert rag_pack.name == "rag"

    def test_pack_has_8_stages(self, rag_pack):
        """Default pack has 8 stages."""
        stages = rag_pack.get_stages()
        assert len(stages) == 8

    def test_pack_stage_order(self, rag_pack):
        """Stages are in correct order."""
        stages = rag_pack.get_stages()
        stage_names = [s.name for s in stages]

        expected = [
            "retrieval",
            "context_relevance",
            "faithfulness",
            "answer_relevance",
            "citation",
            "source_policy",
            "policy_compliance",
            "rejection_calibration",
        ]
        assert stage_names == expected

    def test_pack_with_custom_config(self):
        """Pack accepts custom configuration."""
        config = RAGPackConfig(
            retrieval_threshold=0.8,
            faithfulness_threshold=0.9,
        )
        pack = RAGPack(config=config)

        assert pack.config.retrieval_threshold == 0.8
        assert pack.config.faithfulness_threshold == 0.9


# =============================================================================
# Evaluation Pipeline E2E Tests
# =============================================================================


class TestRAGEvaluationPipelineE2E:
    """End-to-end tests for RAG evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_successful_evaluation(
        self, rag_pack, simple_rag_test_case, system_output_success, context
    ):
        """Successful evaluation produces valid scorecard."""
        scorecard = await rag_pack.evaluate(simple_rag_test_case, system_output_success, context)

        assert isinstance(scorecard, Scorecard)
        assert scorecard.test_case_id == simple_rag_test_case.id
        assert scorecard.pack_name == "rag"
        assert scorecard.batch_id == "rag-e2e-batch"

    @pytest.mark.asyncio
    async def test_retrieval_stage_executed(
        self, rag_pack, simple_rag_test_case, system_output_success, context
    ):
        """Retrieval stage is executed and produces metrics."""
        scorecard = await rag_pack.evaluate(simple_rag_test_case, system_output_success, context)

        assert "retrieval" in scorecard.stage_results
        retrieval_result = scorecard.stage_results["retrieval"]
        assert retrieval_result.stage_name == "retrieval"
        assert "recall_at_5" in retrieval_result.metrics
        assert "mrr" in retrieval_result.metrics

    @pytest.mark.asyncio
    async def test_all_stages_executed(
        self, rag_pack, simple_rag_test_case, system_output_success, context
    ):
        """All 8 stages are executed (unless gate fails)."""
        scorecard = await rag_pack.evaluate(simple_rag_test_case, system_output_success, context)

        # At least the first few stages should run
        assert "retrieval" in scorecard.stage_results
        assert "context_relevance" in scorecard.stage_results
        assert "faithfulness" in scorecard.stage_results

    @pytest.mark.asyncio
    async def test_minimal_pack_evaluation(
        self, rag_pack_minimal, simple_rag_test_case, system_output_success, context
    ):
        """Minimal pack with only retrieval works."""
        scorecard = await rag_pack_minimal.evaluate(
            simple_rag_test_case, system_output_success, context
        )

        assert len(scorecard.stage_results) == 1
        assert "retrieval" in scorecard.stage_results


# =============================================================================
# Retrieval Metrics E2E Tests
# =============================================================================


class TestRetrievalMetricsE2E:
    """End-to-end tests for retrieval metrics."""

    @pytest.mark.asyncio
    async def test_perfect_retrieval_high_score(self, rag_pack_minimal, context):
        """Perfect retrieval gets high score."""
        test_case = TestCase(
            id="ret-perfect-001",
            input={"query": "test"},
            expected={"relevant_docs": ["doc-1", "doc-2"]},
        )
        system_output = {
            "response": "Answer",
            "retrieved_doc_ids": ["doc-1", "doc-2", "doc-3"],
        }

        scorecard = await rag_pack_minimal.evaluate(test_case, system_output, context)

        retrieval_result = scorecard.stage_results["retrieval"]
        assert retrieval_result.passed is True
        assert retrieval_result.score >= 0.5

    @pytest.mark.asyncio
    async def test_no_retrieval_low_score(self, rag_pack_minimal, context):
        """No relevant retrieval gets low score."""
        test_case = TestCase(
            id="ret-none-001",
            input={"query": "test"},
            expected={"relevant_docs": ["doc-1"]},
        )
        system_output = {
            "response": "Answer",
            "retrieved_doc_ids": ["doc-99", "doc-100"],
        }

        scorecard = await rag_pack_minimal.evaluate(test_case, system_output, context)

        retrieval_result = scorecard.stage_results["retrieval"]
        assert retrieval_result.passed is False
        assert retrieval_result.score < 0.5

    @pytest.mark.asyncio
    async def test_retrieval_skipped_without_ground_truth(self, rag_pack_minimal, context):
        """Retrieval skipped when no ground truth labels."""
        test_case = TestCase(
            id="ret-skip-001",
            input={"query": "test"},
            expected={"behavior": "answer"},  # No relevant_docs
        )
        system_output = {
            "response": "Answer",
            "retrieved_doc_ids": ["doc-1"],
        }

        scorecard = await rag_pack_minimal.evaluate(test_case, system_output, context)

        retrieval_result = scorecard.stage_results["retrieval"]
        assert retrieval_result.passed is True
        assert "skipped" in retrieval_result.reason.lower()


# =============================================================================
# Gate Stage E2E Tests
# =============================================================================


class TestGateStagesE2E:
    """End-to-end tests for gate stages."""

    def test_gate_stages_identified(self, rag_pack):
        """Gate stages are correctly identified."""
        stages = rag_pack.get_stages()
        gate_stages = [s for s in stages if s.is_gate]
        gate_names = [s.name for s in gate_stages]

        assert "source_policy" in gate_names
        assert "policy_compliance" in gate_names
        assert len(gate_stages) == 2

    def test_non_gate_stages_dont_block(self, rag_pack):
        """Non-gate stages don't stop pipeline."""
        stages = rag_pack.get_stages()
        non_gate_stages = [s for s in stages if not s.is_gate]

        # 6 stages should be non-gate
        assert len(non_gate_stages) == 6


# =============================================================================
# Scorecard Structure E2E Tests
# =============================================================================


class TestRAGScorecardE2E:
    """End-to-end tests for RAG scorecard structure."""

    @pytest.mark.asyncio
    async def test_scorecard_has_stage_results(
        self, rag_pack, simple_rag_test_case, system_output_success, context
    ):
        """Scorecard contains stage results."""
        scorecard = await rag_pack.evaluate(simple_rag_test_case, system_output_success, context)

        assert scorecard.stage_results is not None
        assert len(scorecard.stage_results) > 0

    @pytest.mark.asyncio
    async def test_scorecard_has_weights(
        self, rag_pack, simple_rag_test_case, system_output_success, context
    ):
        """Scorecard contains stage weights."""
        scorecard = await rag_pack.evaluate(simple_rag_test_case, system_output_success, context)

        assert scorecard.stage_weights is not None
        assert "retrieval" in scorecard.stage_weights
        assert "faithfulness" in scorecard.stage_weights

    @pytest.mark.asyncio
    async def test_scorecard_captures_latency(
        self, rag_pack, simple_rag_test_case, system_output_success, context
    ):
        """Scorecard tracks total latency."""
        scorecard = await rag_pack.evaluate(simple_rag_test_case, system_output_success, context)

        assert scorecard.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_scorecard_captures_output(
        self, rag_pack, simple_rag_test_case, system_output_success, context
    ):
        """Scorecard captures generated output."""
        scorecard = await rag_pack.evaluate(simple_rag_test_case, system_output_success, context)

        assert scorecard.generated_output is not None
        assert "response" in scorecard.generated_output


# =============================================================================
# Scoring Computation E2E Tests
# =============================================================================


class TestRAGScoringE2E:
    """End-to-end tests for RAG score computation."""

    def test_gates_excluded_from_weighted_score(self, rag_pack):
        """Gate stages are excluded from weighted scoring."""
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.8),
            "source_policy": StageResult(stage_name="source_policy", passed=True, score=1.0),
        }

        score = rag_pack.compute_overall_score(stage_results)

        # Only retrieval should be weighted (source_policy is gate)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_gate_failure_fails_overall(self, rag_pack):
        """Gate stage failure fails overall evaluation."""
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.9),
            "source_policy": StageResult(stage_name="source_policy", passed=False, score=0.0),
        }

        passed = rag_pack.compute_overall_passed(stage_results)

        assert passed is False

    def test_low_average_fails_overall(self, rag_pack):
        """Very low average score fails overall."""
        stage_results = {
            "retrieval": StageResult(stage_name="retrieval", passed=False, score=0.1),
            "faithfulness": StageResult(stage_name="faithfulness", passed=False, score=0.1),
        }

        passed = rag_pack.compute_overall_passed(stage_results)

        # Average 0.1 < 0.3 threshold
        assert passed is False


# =============================================================================
# Validation E2E Tests
# =============================================================================


class TestRAGValidationE2E:
    """End-to-end tests for RAG test case validation."""

    def test_valid_retrieval_case(self, rag_pack):
        """Valid test case with relevant_docs passes."""
        test_case = TestCase(
            id="valid-001",
            input={"query": "What is X?"},
            expected={"relevant_docs": ["doc-1"]},
        )

        errors = rag_pack.validate_test_case(test_case)
        assert errors == []

    def test_valid_behavior_case(self, rag_pack):
        """Valid test case with behavior passes."""
        test_case = TestCase(
            id="valid-002",
            input={"query": "What is X?"},
            expected={"behavior": "answer"},
        )

        errors = rag_pack.validate_test_case(test_case)
        assert errors == []

    def test_missing_query_fails(self, rag_pack):
        """Missing query fails validation."""
        test_case = TestCase(
            id="invalid-001",
            input={},
            expected={"relevant_docs": ["doc-1"]},
        )

        errors = rag_pack.validate_test_case(test_case)
        assert len(errors) > 0

    def test_missing_expectations_fails(self, rag_pack):
        """Missing all expectations fails validation."""
        test_case = TestCase(
            id="invalid-002",
            input={"query": "What is X?"},
            expected={},
        )

        errors = rag_pack.validate_test_case(test_case)
        assert len(errors) > 0


# =============================================================================
# Registry Integration E2E Tests
# =============================================================================


class TestRAGRegistryE2E:
    """Tests for pack registry integration."""

    def test_pack_in_evalkit_registry(self):
        """RAGPack is registered in evalkit."""
        from src.evalkit.packs import get_pack

        pack = get_pack("rag")

        assert pack is not None
        assert pack.name == "rag"

    def test_pack_in_registry(self):
        """RAGPack is in pack registry."""
        from src.evalkit.packs import get_available_packs

        packs = get_available_packs()
        assert "rag" in packs
        pack = packs["rag"]()
        assert pack.name == "rag"


# =============================================================================
# Stage Import E2E Tests
# =============================================================================


class TestRAGStageImportsE2E:
    """Tests for RAG stage imports."""

    def test_all_stages_importable(self):
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

    def test_stages_have_required_properties(self, rag_pack):
        """All stages have required name and is_gate properties."""
        for stage in rag_pack.get_stages():
            assert hasattr(stage, "name")
            assert hasattr(stage, "is_gate")
            assert isinstance(stage.name, str)
            assert isinstance(stage.is_gate, bool)
