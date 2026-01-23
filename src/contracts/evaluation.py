"""
Evaluation Models

Models for the evaluation pipeline: scorecards, stage results, and configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.contracts.core import (
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
    """Result of a single evaluation stage."""

    model_config = ConfigDict(frozen=True)

    stage: EvaluationStage
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
    duration_ms: int = Field(
        default=0,
        ge=0,
        description="Time spent in this stage",
    )


# =============================================================================
# Scorecard
# =============================================================================


class Scorecard(BaseModel):
    """
    Complete evaluation result for a single test case execution.

    Designed for storage in Azure Table Storage with partition/row keys.
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

    # Stage Results
    syntax_result: StageResult = Field(..., description="Stage 1 result (always present)")
    logic_result: StageResult | None = Field(
        default=None,
        description="Stage 2 result (None if syntax failed)",
    )
    execution_result: StageResult | None = Field(
        default=None,
        description="Stage 3 result (None if disabled or skipped)",
    )
    semantics_result: StageResult | None = Field(
        default=None,
        description="Stage 4 result (None if not reached)",
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

    # Execution Context
    worker_id: str = Field(..., description="ID of worker that processed this")
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

    @computed_field
    @property
    def overall_passed(self) -> bool:
        """Test passes if all executed stages pass."""
        results = [self.syntax_result]
        if self.logic_result:
            results.append(self.logic_result)
        if self.execution_result:
            results.append(self.execution_result)
        if self.semantics_result:
            results.append(self.semantics_result)
        return all(r.passed for r in results)

    @computed_field
    @property
    def overall_score(self) -> float:
        """Weighted average of all stage scores (reflects criticality)."""
        # Weights: execution (critical) > logic (high) > semantics (low) > syntax (gate)
        weights = {
            EvaluationStage.SYNTAX: 0.1,
            EvaluationStage.LOGIC: 0.3,
            EvaluationStage.EXECUTION: 0.5,  # CRITICAL - most important
            EvaluationStage.SEMANTICS: 0.1,  # Optional polish
        }
        total_weight = 0.0
        weighted_score = 0.0

        for result in [
            self.syntax_result,
            self.logic_result,
            self.execution_result,
            self.semantics_result,
        ]:
            if result is not None:
                w = weights[result.stage]
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
# Evaluator ABC
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
    "StageResult",
    "Scorecard",
    "EvaluationConfig",
    "LLMJudgeConfig",
    "Evaluator",
]
