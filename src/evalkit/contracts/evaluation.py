"""
Evaluation Models

Models for the evaluation pipeline: scorecards, stage results, and configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.evalkit.contracts.core import (
    ErrorCode,
    EvaluationStage,
    SystemResponse,
    TemporalValidationMode,
    TestCase,
    ToolCall,
    _generate_id,
    _now_utc,
)

# =============================================================================
# Stage Results
# =============================================================================


class StageResult(BaseModel):
    """Result of a single evaluation stage.

    Supports both:
    - Generic stage names (string) for general-purpose evaluation packs
    - EvaluationStage enum for backwards compatibility with NL2API pack
    """

    model_config = ConfigDict(frozen=True)

    # Generic stage name (string) - preferred for new packs
    stage_name: str = Field(
        default="",
        description="Name of the evaluation stage (e.g., 'syntax', 'retrieval', 'faithfulness')",
    )

    # NL2API-specific stage enum (backwards compatible)
    stage: EvaluationStage | None = Field(
        default=None,
        description="DEPRECATED: Use stage_name instead. Kept for backwards compatibility.",
    )

    passed: bool
    score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Normalized score (0.0 to 1.0)",
    )
    error_code: ErrorCode | None = Field(
        default=None,
        description="Structured error code if failed",
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable explanation",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Debug info: diffs, intermediate results, etc.",
    )
    # Generic metrics dict for pack-specific metrics
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific metrics (e.g., recall@5, precision@5 for retrieval)",
    )
    duration_ms: int = Field(
        default=0,
        ge=0,
        description="Time spent in this stage",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize StageResult, syncing stage_name and stage fields."""
        # If stage is provided but not stage_name, derive stage_name from stage
        if "stage" in data and data["stage"] is not None and not data.get("stage_name"):
            data["stage_name"] = data["stage"].value
        # If stage_name is provided but not stage, try to map to EvaluationStage
        elif "stage_name" in data and data["stage_name"] and data.get("stage") is None:
            try:
                data["stage"] = EvaluationStage(data["stage_name"])
            except ValueError:
                # Not a known NL2API stage, leave stage as None
                pass
        super().__init__(**data)

    def get_stage_name(self) -> str:
        """Get the stage name, preferring stage_name over deprecated stage enum."""
        if self.stage_name:
            return self.stage_name
        if self.stage:
            return self.stage.value
        return "unknown"


# =============================================================================
# Scorecard
# =============================================================================


class Scorecard(BaseModel):
    """
    Complete evaluation result for a single test case execution.

    Designed for storage in Azure Table Storage with partition/row keys.

    GENERIC FIELDS (for general-purpose evaluation):
    - stage_results: dict[str, StageResult] - arbitrary stages keyed by name
    - pack_name: which evaluation pack was used

    NL2API-SPECIFIC FIELDS (backwards compatible):
    - syntax_result, logic_result, execution_result, semantics_result

    Generic stage_results is populated automatically from NL2API fields.
    """

    model_config = ConfigDict(frozen=True)

    # Identity & Keys
    test_case_id: str = Field(..., description="References TestCase.id")
    batch_id: str | None = Field(
        default=None,
        description="Batch this test belongs to (used as PartitionKey)",
    )
    scorecard_id: str = Field(
        default_factory=_generate_id,
        description="Unique scorecard ID",
    )

    # ==========================================================================
    # GENERIC FIELDS (for general-purpose evaluation framework)
    # ==========================================================================
    pack_name: str = Field(
        default="nl2api",
        description="Name of the evaluation pack used (e.g., 'nl2api', 'rag')",
    )
    stage_results: dict[str, StageResult] = Field(
        default_factory=dict,
        description="Generic stage results keyed by stage name",
    )
    stage_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights used for overall score calculation",
    )

    # Client tracking (multi-client evaluation)
    client_type: str | None = Field(
        default=None,
        description="Type of client (internal, mcp_claude, mcp_chatgpt, mcp_custom)",
    )
    client_version: str | None = Field(
        default=None,
        description="Client version identifier (e.g., claude-opus-4.5-20251101)",
    )
    eval_mode: str | None = Field(
        default=None,
        description="Evaluation mode (orchestrator, tool_only, routing, resolver)",
    )

    # Cost tracking
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Input/prompt tokens used",
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Output/completion tokens used",
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated cost in USD",
    )

    # Timestamps
    timestamp: datetime = Field(default_factory=_now_utc)
    completed_at: datetime | None = Field(default=None)

    # ==========================================================================
    # NL2API-SPECIFIC FIELDS (backwards compatible)
    # ==========================================================================
    syntax_result: StageResult | None = Field(
        default=None,
        description="Stage 1 result (NL2API-specific)",
    )
    logic_result: StageResult | None = Field(
        default=None,
        description="Stage 2 result (NL2API-specific)",
    )
    execution_result: StageResult | None = Field(
        default=None,
        description="Stage 3 result (NL2API-specific)",
    )
    semantics_result: StageResult | None = Field(
        default=None,
        description="Stage 4 result (NL2API-specific)",
    )

    # Captured Outputs
    generated_tool_calls: tuple[ToolCall, ...] | None = Field(
        default=None,
        description="Actual tool calls from target system",
    )
    generated_nl_response: str | None = Field(
        default=None,
        description="Actual NL response from target system",
    )
    # Generic captured output for non-NL2API packs
    generated_output: dict[str, Any] = Field(
        default_factory=dict,
        description="Generic captured output from target system",
    )

    # Execution Context
    worker_id: str = Field(
        default="local",
        description="ID of worker that processed this",
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        description="Which attempt this is (for retries)",
    )
    message_id: str | None = Field(
        default=None,
        description="Service Bus message ID for correlation",
    )

    # Metrics
    total_latency_ms: int = Field(
        default=0,
        ge=0,
        description="Total processing time",
    )

    # Temporal evaluation context
    evaluation_date: date | None = Field(
        default=None,
        description="Date used for temporal normalization (for reproducibility)",
    )
    temporal_validation_mode: str | None = Field(
        default=None,
        description="Temporal validation mode used (behavioral, structural, data)",
    )
    date_normalization_applied: bool = Field(
        default=False,
        description="Whether date normalization was applied during evaluation",
    )

    def get_all_stage_results(self) -> dict[str, StageResult]:
        """Get all stage results as a dict, merging NL2API fields with generic stage_results."""
        results = dict(self.stage_results)

        # Add NL2API-specific results if present and not already in dict
        if self.syntax_result and "syntax" not in results:
            results["syntax"] = self.syntax_result
        if self.logic_result and "logic" not in results:
            results["logic"] = self.logic_result
        if self.execution_result and "execution" not in results:
            results["execution"] = self.execution_result
        if self.semantics_result and "semantics" not in results:
            results["semantics"] = self.semantics_result

        return results

    @computed_field
    @property
    def overall_passed(self) -> bool:
        """Test passes if all executed stages pass."""
        all_results = self.get_all_stage_results()
        if not all_results:
            return False
        return all(r.passed for r in all_results.values())

    @computed_field
    @property
    def overall_score(self) -> float:
        """Weighted average of all stage scores."""
        all_results = self.get_all_stage_results()
        if not all_results:
            return 0.0

        # Use provided weights or default NL2API weights
        weights = (
            self.stage_weights
            if self.stage_weights
            else {
                "syntax": 0.1,
                "logic": 0.3,
                "execution": 0.5,  # CRITICAL - most important
                "semantics": 0.1,  # Optional polish
            }
        )

        total_weight = 0.0
        weighted_score = 0.0

        for stage_name, result in all_results.items():
            w = weights.get(stage_name, 0.25)  # Default weight for unknown stages
            weighted_score += result.score * w
            total_weight += w

        return weighted_score / total_weight if total_weight > 0 else 0.0

    @computed_field
    @property
    def partition_key(self) -> str:
        """Azure Table Storage partition key."""
        return self.batch_id or "UNBATCHED"

    @computed_field
    @property
    def row_key(self) -> str:
        """Azure Table Storage row key."""
        return f"{self.test_case_id}_{self.scorecard_id}"

    @classmethod
    def from_stage_results(
        cls,
        test_case_id: str,
        stage_results: dict[str, StageResult],
        pack_name: str = "generic",
        stage_weights: dict[str, float] | None = None,
        worker_id: str = "local",
        **kwargs: Any,
    ) -> Scorecard:
        """Create a Scorecard from generic stage results dict.

        This is the preferred way to create scorecards for non-NL2API packs.
        """
        return cls(
            test_case_id=test_case_id,
            pack_name=pack_name,
            stage_results=stage_results,
            stage_weights=stage_weights or {},
            worker_id=worker_id,
            **kwargs,
        )


# =============================================================================
# Configuration
# =============================================================================


class EvaluationConfig(BaseModel):
    """
    Configuration for the evaluation pipeline.

    Controls which stages run and their parameters.

    Stage Criticality:
    - Stage 1 (Syntax): Gate - always runs
    - Stage 2 (Logic): High - always runs
    - Stage 3 (Execution): CRITICAL - the ultimate correctness test
    - Stage 4 (Semantics): Low - evaluates presentation, not correctness
    """

    model_config = ConfigDict(frozen=True)

    # Stage toggles
    execution_stage_enabled: bool = Field(
        default=True,
        description=(
            "Stage 3: CRITICAL - the most important test. "
            "Disable only for CI/cost savings; enable for nightly/release."
        ),
    )
    semantics_stage_enabled: bool = Field(
        default=False,
        description=(
            "Stage 4: OPTIONAL - evaluates NL response quality, not correctness. "
            "Enable when assessing presentation/UX."
        ),
    )

    # Temporal evaluation
    temporal_mode: TemporalValidationMode = Field(
        default=TemporalValidationMode.STRUCTURAL,
        description="Temporal validation mode for date comparison",
    )
    evaluation_date: date | None = Field(
        default=None,
        description="Reference date for temporal normalization (defaults to today)",
    )
    relative_date_fields: tuple[str, ...] = Field(
        default=("start", "end", "SDate", "EDate", "Period"),
        description="Field names that may contain relative date expressions",
    )
    fiscal_year_end_month: int = Field(
        default=12,
        ge=1,
        le=12,
        description="Month when fiscal year ends (for FY/FQ resolution)",
    )

    # Tolerances
    numeric_tolerance: float = Field(
        default=0.0001,
        ge=0.0,
        le=1.0,
        description="Relative tolerance for numeric comparisons (0.01% = 0.0001)",
    )
    string_similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for fuzzy string matching",
    )

    # Timeouts (milliseconds)
    syntax_timeout_ms: int = Field(default=5000, ge=100)
    logic_timeout_ms: int = Field(default=10000, ge=100)
    execution_timeout_ms: int = Field(default=30000, ge=100)
    semantics_timeout_ms: int = Field(default=60000, ge=100)

    # Score weights (reflect criticality: execution > logic > semantics > syntax)
    syntax_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    logic_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    execution_weight: float = Field(default=0.5, ge=0.0, le=1.0)  # CRITICAL
    semantics_weight: float = Field(default=0.1, ge=0.0, le=1.0)  # Optional polish


class LLMJudgeConfig(BaseModel):
    """Configuration for the LLM-as-Judge semantic evaluator."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Model for semantic evaluation (Claude 3.5 Haiku by default)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 for deterministic)",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        description="Max tokens in judge response",
    )
    pass_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum score to pass semantic evaluation",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        description="Timeout for LLM judge call in milliseconds",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts on transient errors",
    )
    # Evaluation weights
    meaning_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for meaning match criterion",
    )
    completeness_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for completeness criterion",
    )
    accuracy_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for accuracy criterion",
    )


# =============================================================================
# Evaluation Pack Protocol (General-Purpose Framework)
# =============================================================================


@dataclass
class EvalContext:
    """
    Context passed to evaluation stages.

    Contains configuration, LLM judge client, and other shared resources.
    """

    config: dict[str, Any] = field(default_factory=dict)
    llm_judge: Any | None = None  # LLM client for LLM-as-judge evaluations
    evaluation_date: date | None = None
    worker_id: str = "local"
    batch_id: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self.config.get(key, default)


class Stage(Protocol):
    """
    Protocol for a single evaluation stage.

    Stages are the building blocks of evaluation packs. Each stage
    evaluates one aspect of the system output and returns a StageResult.

    Example stages:
    - NL2API: SyntaxStage, LogicStage, ExecutionStage, SemanticsStage
    - RAG: RetrievalStage, FaithfulnessStage, AnswerRelevanceStage
    """

    @property
    def name(self) -> str:
        """Unique name for this stage (e.g., 'syntax', 'retrieval')."""
        ...

    @property
    def is_gate(self) -> bool:
        """If True, pipeline stops on failure. If False, continues."""
        ...

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext,
    ) -> StageResult:
        """
        Evaluate the system output against the test case.

        Args:
            test_case: The test case with expected values.
            system_output: Output from the target system (pack-specific schema).
            context: Evaluation context with config and shared resources.

        Returns:
            StageResult with pass/fail, score, and stage-specific metrics.
        """
        ...


class EvaluationPack(Protocol):
    """
    Protocol for domain-specific evaluation logic.

    An EvaluationPack defines:
    - Which stages to run and in what order
    - Default scoring weights per stage
    - Test case validation rules
    - Overall score computation

    Example packs:
    - NL2APIPack: Tool-calling LLM evaluation (syntax, logic, execution, semantics)
    - RAGPack: RAG system evaluation (retrieval, faithfulness, relevance)

    Usage:
        pack = NL2APIPack()
        evaluator = Evaluator(pack=pack)
        results = await evaluator.evaluate(test_cases, system)
    """

    @property
    def name(self) -> str:
        """Unique name for this pack (e.g., 'nl2api', 'rag')."""
        ...

    def get_stages(self) -> list[Stage]:
        """Return ordered list of evaluation stages."""
        ...

    def get_default_weights(self) -> dict[str, float]:
        """Return default scoring weights per stage name."""
        ...

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        """
        Validate test case has required fields for this pack.

        Returns:
            List of validation error messages. Empty if valid.
        """
        ...

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Compute weighted overall score from stage results.

        Args:
            stage_results: Results keyed by stage name.
            weights: Optional custom weights. Uses default if None.

        Returns:
            Weighted average score (0.0 to 1.0).
        """
        ...

    def compute_overall_passed(
        self,
        stage_results: dict[str, StageResult],
    ) -> bool:
        """
        Determine if overall evaluation passed.

        Default: All stages must pass.
        Override for custom logic (e.g., only gate stages must pass).
        """
        ...


# =============================================================================
# Evaluator ABC (NL2API-specific, kept for backwards compatibility)
# =============================================================================


class Evaluator(ABC):
    """
    Abstract Base Class for the Waterfall Evaluation Pipeline.

    Implementations must handle all four stages, respecting the
    hard/soft stop rules defined in the architecture.
    """

    # Class-level default config
    DEFAULT_CONFIG: ClassVar[EvaluationConfig] = EvaluationConfig()

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or self.DEFAULT_CONFIG

    @abstractmethod
    async def evaluate(
        self,
        test_case: TestCase,
        system_response: SystemResponse,
        worker_id: str,
    ) -> Scorecard:
        """
        Run the complete evaluation pipeline.

        Args:
            test_case: The gold standard test definition.
            system_response: Captured response from target system.
            worker_id: ID of the processing worker.

        Returns:
            Scorecard with results from all applicable stages.
        """
        pass

    @abstractmethod
    def evaluate_syntax(self, raw_output: str) -> StageResult:
        """
        Stage 1: Validate JSON structure and schema.

        Args:
            raw_output: Raw string from target system.

        Returns:
            StageResult with parsed output in artifacts if successful.
        """
        pass

    @abstractmethod
    def evaluate_logic(
        self,
        expected: tuple[ToolCall, ...],
        actual: tuple[ToolCall, ...],
    ) -> StageResult:
        """
        Stage 2: AST-based comparison of tool calls.

        Args:
            expected: Expected tool calls from test case.
            actual: Actual tool calls from system response.

        Returns:
            StageResult with diff details in artifacts.
        """
        pass

    @abstractmethod
    async def evaluate_execution(
        self,
        expected_data: dict[str, Any] | None,
        actual_data: dict[str, Any] | None,
    ) -> StageResult:
        """
        Stage 3: Compare execution results with tolerance.

        Args:
            expected_data: Expected return data from test case.
            actual_data: Actual return data from execution.

        Returns:
            StageResult with value comparison details.
        """
        pass

    @abstractmethod
    async def evaluate_semantics(
        self,
        expected_text: str,
        actual_text: str,
    ) -> StageResult:
        """
        Stage 4: LLM-as-Judge semantic comparison.

        Args:
            expected_text: Expected NL response.
            actual_text: Actual NL response.

        Returns:
            StageResult with LLM reasoning in artifacts.
        """
        pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Stage Results & Scorecard
    "StageResult",
    "Scorecard",
    # Configuration
    "EvaluationConfig",
    "LLMJudgeConfig",
    # Evaluation Pack Protocol (general-purpose framework)
    "EvalContext",
    "Stage",
    "EvaluationPack",
    # Evaluator ABC (NL2API-specific, backwards compatible)
    "Evaluator",
]
