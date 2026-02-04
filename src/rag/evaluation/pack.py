"""
RAG Evaluation Pack

Evaluates Retrieval-Augmented Generation systems with 8 stages:

RAG Triad (Core, Reference-Free):
1. Retrieval: IR metrics (recall@k, precision@k, MRR, NDCG)
2. Context Relevance: Is retrieved context relevant to query?
3. Faithfulness: Is response grounded in context?
4. Answer Relevance: Does response answer the question?

Domain Gates:
5. Citation: Citation presence, validity, accuracy, coverage
6. Source Policy: Quote-only vs summarize enforcement (GATE)
7. Policy Compliance: Content policy violations (GATE)
8. Rejection Calibration: False positive/negative detection

Usage:
    from src.rag.evaluation import RAGPack
    from src.evalkit.core.evaluator import Evaluator

    pack = RAGPack()
    evaluator = Evaluator(pack=pack)
    scorecard = await evaluator.evaluate(test_case, system_output)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.evalkit.common.telemetry import get_tracer
from src.evalkit.contracts import Scorecard, StageResult, TestCase

from .stages import (
    AnswerRelevanceStage,
    CitationStage,
    ContextRelevanceStage,
    FaithfulnessStage,
    PolicyComplianceStage,
    RejectionCalibrationStage,
    RetrievalStage,
    SourcePolicyStage,
)

if TYPE_CHECKING:
    from src.evalkit.contracts import EvalContext

tracer = get_tracer(__name__)


def _truncate(text: str, max_length: int = 500) -> str:
    """Truncate text for trace attributes, preserving useful prefix."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... ({len(text) - max_length} more chars)"


@dataclass
class ProviderThresholds:
    """Provider-specific thresholds tuned for different LLM characteristics."""

    retrieval: float = 0.5
    context_relevance: float = 0.35
    faithfulness: float = 0.4
    answer_relevance: float = 0.5
    citation: float = 0.6


# Tuned thresholds per provider (based on evaluation runs)
PROVIDER_THRESHOLDS: dict[str, ProviderThresholds] = {
    # OpenAI stack (gpt-5-nano generation + gpt-5-nano judge w/ reasoning=minimal, 400 samples, 2026-02-01):
    # - context_relevance: p10=0.24, p50=0.60, avg=0.58 → threshold 0.25 gives 89% pass
    # - faithfulness: avg=0.83, 93% pass at 0.4 threshold
    # - answer_relevance: avg=0.78, 91% pass at 0.5 threshold
    "openai": ProviderThresholds(
        retrieval=0.5,
        context_relevance=0.25,  # Lower than anthropic (gpt-5-nano scores stricter on context)
        faithfulness=0.4,
        answer_relevance=0.5,
        citation=0.6,
    ),
    # Anthropic claude-3-5-haiku (baseline, 50 samples, 2026-02-01):
    # - context_relevance: p10=0.42, avg=0.63 → 96% pass at 0.35
    # - faithfulness: avg=0.78, 88-90% pass at 0.4
    "anthropic": ProviderThresholds(
        retrieval=0.5,
        context_relevance=0.35,
        faithfulness=0.4,
        answer_relevance=0.5,
        citation=0.6,
    ),
}


@dataclass
class RAGPackConfig:
    """Configuration for RAGPack."""

    # Stage enablement
    retrieval_enabled: bool = True
    context_relevance_enabled: bool = True
    faithfulness_enabled: bool = True
    answer_relevance_enabled: bool = True
    citation_enabled: bool = True
    source_policy_enabled: bool = True
    policy_compliance_enabled: bool = True
    rejection_calibration_enabled: bool = True

    # LLM provider for threshold selection (openai, anthropic)
    # Set to None to use explicit thresholds below
    llm_provider: str | None = None

    # Stage thresholds - used when llm_provider is None
    # Otherwise, thresholds are loaded from PROVIDER_THRESHOLDS
    retrieval_threshold: float = 0.5
    context_relevance_threshold: float = 0.35
    faithfulness_threshold: float = 0.4
    answer_relevance_threshold: float = 0.5
    citation_threshold: float = 0.6

    # Weight overrides (None = use default)
    custom_weights: dict[str, float] | None = None

    # Stage execution mode: parallel (faster) vs sequential (rate-limit friendly)
    # Default False for Anthropic compatibility; set True for OpenAI which has
    # token-based limits rather than concurrent connection limits
    parallel_stages: bool = False

    def get_thresholds(self) -> ProviderThresholds:
        """Get thresholds based on provider or explicit config."""
        if self.llm_provider and self.llm_provider in PROVIDER_THRESHOLDS:
            return PROVIDER_THRESHOLDS[self.llm_provider]
        return ProviderThresholds(
            retrieval=self.retrieval_threshold,
            context_relevance=self.context_relevance_threshold,
            faithfulness=self.faithfulness_threshold,
            answer_relevance=self.answer_relevance_threshold,
            citation=self.citation_threshold,
        )


class RAGPack:
    """
    Evaluation pack for RAG (Retrieval-Augmented Generation) systems.

    Implements the EvaluationPack protocol with 8 stages organized into:
    - RAG Triad: Core quality metrics (reference-free where possible)
    - Domain Gates: Citation, policy, and rejection handling

    Example:
        pack = RAGPack()
        evaluator = Evaluator(pack=pack)

        test_case = TestCase(
            id="rag-1",
            input={"query": "What is the capital of France?"},
            expected={
                "relevant_docs": ["doc-123"],
                "behavior": "answer",
            },
        )

        system_output = {
            "response": "The capital of France is Paris.",
            "retrieved_chunks": [{"id": "doc-123", "text": "Paris is the capital..."}],
        }

        scorecard = await evaluator.evaluate(test_case, system_output)
    """

    # Default weights per stage (reflects relative importance)
    DEFAULT_WEIGHTS: dict[str, float] = {
        # RAG Triad - core metrics (70% total)
        "retrieval": 0.15,
        "context_relevance": 0.15,
        "faithfulness": 0.25,  # Critical - hallucination detection
        "answer_relevance": 0.15,
        # Domain gates (30% total)
        "citation": 0.10,
        "source_policy": 0.05,  # GATE - binary pass/fail
        "policy_compliance": 0.05,  # GATE - binary pass/fail
        "rejection_calibration": 0.10,
    }

    def __init__(
        self,
        config: RAGPackConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the RAG evaluation pack.

        Args:
            config: Pack configuration
            **kwargs: Individual config overrides
        """
        self.config = config or RAGPackConfig(**kwargs)
        self._stages = self._build_stages()

    @property
    def name(self) -> str:
        """Return the pack name."""
        return "rag"

    def _build_stages(self) -> list[Any]:
        """Build the list of enabled stages."""
        stages: list[Any] = []
        thresholds = self.config.get_thresholds()

        # RAG Triad stages
        if self.config.retrieval_enabled:
            stages.append(RetrievalStage(pass_threshold=thresholds.retrieval))

        if self.config.context_relevance_enabled:
            stages.append(ContextRelevanceStage(pass_threshold=thresholds.context_relevance))

        if self.config.faithfulness_enabled:
            stages.append(FaithfulnessStage(pass_threshold=thresholds.faithfulness))

        if self.config.answer_relevance_enabled:
            stages.append(AnswerRelevanceStage(pass_threshold=thresholds.answer_relevance))

        # Domain gates
        if self.config.citation_enabled:
            stages.append(CitationStage(pass_threshold=thresholds.citation))

        if self.config.source_policy_enabled:
            stages.append(SourcePolicyStage())

        if self.config.policy_compliance_enabled:
            stages.append(PolicyComplianceStage())

        if self.config.rejection_calibration_enabled:
            stages.append(RejectionCalibrationStage())

        return stages

    def get_stages(self) -> list[Any]:
        """Return ordered list of evaluation stages."""
        return list(self._stages)

    def get_default_weights(self) -> dict[str, float]:
        """Return default scoring weights per stage name."""
        if self.config.custom_weights:
            return {**self.DEFAULT_WEIGHTS, **self.config.custom_weights}
        return dict(self.DEFAULT_WEIGHTS)

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        """
        Validate test case has required fields for RAG evaluation.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        # Must have query
        if not test_case.input.get("query"):
            errors.append("Missing input['query']")

        # Should have either relevant_docs (for retrieval eval) or behavior (for rejection eval)
        has_retrieval = bool(test_case.expected.get("relevant_docs"))
        has_behavior = bool(test_case.expected.get("behavior"))
        has_answer = bool(test_case.expected.get("answer"))

        if not has_retrieval and not has_behavior and not has_answer:
            errors.append(
                "Need at least one of: expected['relevant_docs'], "
                "expected['behavior'], or expected['answer']"
            )

        return errors

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Compute weighted overall score from stage results.

        GATE stages are not weighted - they must pass but don't affect score.
        Non-gate stages are weighted according to default or custom weights.

        Args:
            stage_results: Results keyed by stage name.
            weights: Optional custom weights. Uses default if None.

        Returns:
            Weighted average score (0.0 to 1.0).
        """
        weights = weights or self.get_default_weights()

        total_weight = 0.0
        weighted_sum = 0.0

        for stage_name, result in stage_results.items():
            # Skip GATE stages in weighted scoring
            stage = next((s for s in self._stages if s.name == stage_name), None)
            if stage and stage.is_gate:
                continue

            if stage_name in weights and result.score is not None:
                w = weights[stage_name]
                weighted_sum += result.score * w
                total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def compute_overall_passed(
        self,
        stage_results: dict[str, StageResult],
    ) -> bool:
        """
        Determine if overall evaluation passed.

        For RAG:
        - All GATE stages must pass (source_policy, policy_compliance)
        - Other stages can fail without blocking
        """
        for stage in self._stages:
            result = stage_results.get(stage.name)
            if result and stage.is_gate and not result.passed:
                return False

        # Also check that non-gate stages aren't catastrophically failing
        non_gate_results = [
            r
            for name, r in stage_results.items()
            if any(s.name == name and not s.is_gate for s in self._stages)
        ]

        if non_gate_results:
            avg_score = sum(r.score for r in non_gate_results) / len(non_gate_results)
            # Fail if average non-gate score is below 0.3
            if avg_score < 0.3:
                return False

        return True

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None = None,
    ) -> Scorecard:
        """
        Run the complete RAG evaluation pipeline.

        This is a convenience method that runs all stages and creates a Scorecard.
        For more control, use the generic Evaluator class.

        Args:
            test_case: The test case with expected values
            system_output: Output from RAG system. Expected keys:
                - response: Generated response text
                - retrieved_chunks: List of retrieved documents/chunks
                - sources: Source metadata with policies
            context: Optional evaluation context

        Returns:
            Scorecard with all stage results
        """
        from src.evalkit.contracts import EvalContext

        context = context or EvalContext()
        start_time = time.perf_counter()

        stage_results: dict[str, StageResult] = {}

        with tracer.start_as_current_span("rag_pack.evaluate") as span:
            span.set_attribute("test_case.id", test_case.id)
            span.set_attribute("pack.name", self.name)
            span.set_attribute("parallel_stages", self.config.parallel_stages)

            # Add input context to trace for debugging
            query = test_case.input.get("query", "")
            response = system_output.get("response", "")
            retrieved_context = system_output.get("context", "")
            span.set_attribute("input.query", _truncate(query, 500))
            span.set_attribute("input.response_length", len(response))
            span.set_attribute("input.context_length", len(retrieved_context))
            # Add response/context as events to avoid huge attribute values
            span.add_event("input", {"query": _truncate(query, 1000)})
            span.add_event(
                "system_output",
                {
                    "response": _truncate(response, 2000),
                    "context": _truncate(retrieved_context, 2000),
                },
            )

            # Unified stage execution with rich tracing
            async def run_stage_with_tracing(stage: Any) -> tuple[str, StageResult]:
                """Run a stage and record detailed trace information."""
                with tracer.start_as_current_span(f"rag_pack.{stage.name}") as stage_span:
                    stage_span.set_attribute("stage.is_gate", stage.is_gate)
                    result = await stage.evaluate(test_case, system_output, context)

                    # Core result attributes
                    stage_span.set_attribute("result.passed", result.passed)
                    stage_span.set_attribute("result.score", result.score)
                    stage_span.set_attribute("result.duration_ms", result.duration_ms)

                    # Add reason if present (useful for debugging failures)
                    if result.reason:
                        stage_span.set_attribute("result.reason", _truncate(result.reason, 500))

                    # Add key metrics as attributes
                    for key, value in result.metrics.items():
                        if isinstance(value, (int, float, bool, str)):
                            stage_span.set_attribute(f"metrics.{key}", value)

                    # Add artifacts as event (may be larger)
                    if result.artifacts:
                        # Filter to serializable values and truncate strings
                        safe_artifacts = {}
                        for k, v in result.artifacts.items():
                            if isinstance(v, (int, float, bool)):
                                safe_artifacts[k] = v
                            elif isinstance(v, str):
                                safe_artifacts[k] = _truncate(v, 500)
                        if safe_artifacts:
                            stage_span.add_event("artifacts", safe_artifacts)

                    return stage.name, result

            if self.config.parallel_stages:
                # Parallel execution - better for OpenAI (token-based limits)
                results = await asyncio.gather(
                    *[run_stage_with_tracing(stage) for stage in self._stages]
                )
                stage_results = dict(results)

                # Check GATE results after parallel execution
                for stage in self._stages:
                    if stage.is_gate:
                        result = stage_results.get(stage.name)
                        if result and not result.passed:
                            span.set_attribute("gate_failed", stage.name)
                            break
            else:
                # Sequential execution - better for Anthropic (connection-based limits)
                for stage in self._stages:
                    _, result = await run_stage_with_tracing(stage)
                    stage_results[stage.name] = result

                    # Check for gate failure (fail-fast in sequential mode)
                    if stage.is_gate and not result.passed:
                        span.set_attribute("gate_failed", stage.name)
                        break

            total_latency_ms = int((time.perf_counter() - start_time) * 1000)
            overall_passed = self.compute_overall_passed(stage_results)
            overall_score = self.compute_overall_score(stage_results)

            span.set_attribute("result.overall_passed", overall_passed)
            span.set_attribute("result.overall_score", overall_score)
            span.set_attribute("result.total_latency_ms", total_latency_ms)

        return Scorecard(
            test_case_id=test_case.id,
            batch_id=context.batch_id,
            pack_name=self.name,
            stage_results=stage_results,
            stage_weights=self.get_default_weights(),
            generated_output=system_output,
            generated_nl_response=system_output.get("response"),
            worker_id=context.worker_id,
            total_latency_ms=total_latency_ms,
        )


@dataclass
class RAGRetrievalPackConfig:
    """Configuration for RAGRetrievalPack (retrieval-only evaluation)."""

    retrieval_threshold: float = 0.5
    context_relevance_threshold: float = 0.35


class RAGRetrievalPack:
    """
    Retrieval-only evaluation pack for RAG systems.

    Evaluates only retrieval quality without requiring LLM-generated answers.
    Use this for fast iteration on retrieval parameters without generation costs.

    Stages:
    1. Retrieval: IR metrics (recall@k, precision@k, MRR, NDCG)
    2. Context Relevance: Is retrieved context relevant to query?

    Usage:
        pack = RAGRetrievalPack()
        evaluator = Evaluator(pack=pack)

        # Only needs retrieved docs, no generated answer
        system_output = {
            "retrieved_chunks": [{"id": "doc-1", "text": "..."}],
            "retrieved_doc_ids": ["doc-1", "doc-2"],
        }

        scorecard = await evaluator.evaluate(test_case, system_output)
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "retrieval": 0.6,
        "context_relevance": 0.4,
    }

    def __init__(self, config: RAGRetrievalPackConfig | None = None, **kwargs):
        self.config = config or RAGRetrievalPackConfig(**kwargs)
        self._stages = [
            RetrievalStage(pass_threshold=self.config.retrieval_threshold),
            ContextRelevanceStage(pass_threshold=self.config.context_relevance_threshold),
        ]

    @property
    def name(self) -> str:
        return "rag-retrieval"

    def get_stages(self) -> list:
        return list(self._stages)

    def get_default_weights(self) -> dict[str, float]:
        return dict(self.DEFAULT_WEIGHTS)

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        errors = []
        if not test_case.input.get("query"):
            errors.append("Missing input['query']")
        if not test_case.expected.get("relevant_docs"):
            errors.append("Missing expected['relevant_docs'] for retrieval evaluation")
        return errors

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        weights = weights or self.get_default_weights()
        total_weight = 0.0
        weighted_sum = 0.0

        for stage_name, result in stage_results.items():
            if stage_name in weights and result.score is not None:
                w = weights[stage_name]
                weighted_sum += result.score * w
                total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def compute_overall_passed(self, stage_results: dict[str, StageResult]) -> bool:
        # Pass if retrieval meets threshold
        retrieval_result = stage_results.get("retrieval")
        if retrieval_result and not retrieval_result.passed:
            return False
        return True

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None = None,
    ) -> Scorecard:
        """Run retrieval-only evaluation pipeline."""
        from src.evalkit.contracts import EvalContext

        context = context or EvalContext()
        start_time = time.perf_counter()

        stage_results: dict[str, StageResult] = {}

        with tracer.start_as_current_span("rag_retrieval_pack.evaluate") as span:
            span.set_attribute("test_case.id", test_case.id)
            span.set_attribute("pack.name", self.name)

            for stage in self._stages:
                with tracer.start_as_current_span(f"rag_retrieval.{stage.name}"):
                    result = await stage.evaluate(test_case, system_output, context)
                    stage_results[stage.name] = result

            total_latency_ms = int((time.perf_counter() - start_time) * 1000)
            overall_passed = self.compute_overall_passed(stage_results)
            overall_score = self.compute_overall_score(stage_results)

            span.set_attribute("result.overall_passed", overall_passed)
            span.set_attribute("result.overall_score", overall_score)

        return Scorecard(
            test_case_id=test_case.id,
            batch_id=context.batch_id,
            pack_name=self.name,
            stage_results=stage_results,
            stage_weights=self.get_default_weights(),
            generated_output=system_output,
            generated_nl_response=system_output.get("response"),
            worker_id=context.worker_id,
            total_latency_ms=total_latency_ms,
        )


@dataclass
class RAGFaithfulnessPackConfig:
    """Configuration for RAGFaithfulnessPack (faithfulness-only evaluation)."""

    faithfulness_threshold: float = 0.4
    llm_provider: str | None = None


class RAGFaithfulnessPack:
    """
    Faithfulness-only evaluation pack for RAG systems.

    Evaluates only faithfulness (groundedness) - whether the response is
    grounded in the retrieved context. Uses LLM-based refusal detection
    to correctly handle "I cannot determine X" responses.

    Use this for fast iteration on faithfulness evaluation without running
    all 8 stages of the full RAG pack.

    Stages:
    1. Faithfulness: Is response grounded in context? (LLM-based)

    Usage:
        pack = RAGFaithfulnessPack()
        evaluator = Evaluator(pack=pack)

        system_output = {
            "response": "The capital of France is Paris.",
            "context": "Paris is the capital of France.",
        }

        scorecard = await evaluator.evaluate(test_case, system_output)
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "faithfulness": 1.0,
    }

    def __init__(self, config: RAGFaithfulnessPackConfig | None = None, **kwargs):
        self.config = config or RAGFaithfulnessPackConfig(**kwargs)

        # Get threshold from provider-specific settings if available
        threshold = self.config.faithfulness_threshold
        if self.config.llm_provider and self.config.llm_provider in PROVIDER_THRESHOLDS:
            threshold = PROVIDER_THRESHOLDS[self.config.llm_provider].faithfulness

        self._stages = [
            FaithfulnessStage(pass_threshold=threshold),
        ]

    @property
    def name(self) -> str:
        return "rag-faithfulness"

    def get_stages(self) -> list:
        return list(self._stages)

    def get_default_weights(self) -> dict[str, float]:
        return dict(self.DEFAULT_WEIGHTS)

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        errors = []
        if not test_case.input.get("query"):
            errors.append("Missing input['query']")
        return errors

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        result = stage_results.get("faithfulness")
        if result and result.score is not None:
            return result.score
        return 0.0

    def compute_overall_passed(self, stage_results: dict[str, StageResult]) -> bool:
        result = stage_results.get("faithfulness")
        return bool(result and result.passed)

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None = None,
    ) -> Scorecard:
        """Run faithfulness-only evaluation pipeline."""
        from src.evalkit.contracts import EvalContext

        context = context or EvalContext()
        start_time = time.perf_counter()

        stage_results: dict[str, StageResult] = {}

        with tracer.start_as_current_span("rag_faithfulness_pack.evaluate") as span:
            span.set_attribute("test_case.id", test_case.id)
            span.set_attribute("pack.name", self.name)

            for stage in self._stages:
                with tracer.start_as_current_span(f"rag_faithfulness.{stage.name}"):
                    result = await stage.evaluate(test_case, system_output, context)
                    stage_results[stage.name] = result

            total_latency_ms = int((time.perf_counter() - start_time) * 1000)
            overall_passed = self.compute_overall_passed(stage_results)
            overall_score = self.compute_overall_score(stage_results)

            span.set_attribute("result.overall_passed", overall_passed)
            span.set_attribute("result.overall_score", overall_score)

        return Scorecard(
            test_case_id=test_case.id,
            batch_id=context.batch_id,
            pack_name=self.name,
            stage_results=stage_results,
            stage_weights=self.get_default_weights(),
            generated_output=system_output,
            generated_nl_response=system_output.get("response"),
            worker_id=context.worker_id,
            total_latency_ms=total_latency_ms,
        )
