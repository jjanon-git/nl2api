"""
EvalPlatform Contracts - Pydantic v2 Schemas

This module defines all data contracts for the distributed evaluation framework.
All models are designed for:
- Azure Table Storage compatibility (partition/row keys)
- Azure AI Search compatibility (embeddings, filterable fields)
- Service Bus message serialization
- Idempotent processing
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# =============================================================================
# Utility Types
# =============================================================================


class FrozenDict(dict):
    """
    Immutable dictionary that supports hashing for set-based comparisons.
    Used for ToolCall.arguments to enable order-independent comparison.
    """

    def __hash__(self) -> int:
        return hash(self._make_hashable(self))

    def _make_hashable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        if isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        if isinstance(obj, set):
            return frozenset(self._make_hashable(item) for item in obj)
        return obj

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("FrozenDict does not support item assignment")

    def __delitem__(self, key: Any) -> None:
        raise TypeError("FrozenDict does not support item deletion")

    def clear(self) -> None:
        raise TypeError("FrozenDict does not support clear")

    def pop(self, *args: Any) -> Any:
        raise TypeError("FrozenDict does not support pop")

    def popitem(self) -> tuple:
        raise TypeError("FrozenDict does not support popitem")

    def setdefault(self, *args: Any) -> Any:
        raise TypeError("FrozenDict does not support setdefault")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict does not support update")


def _now_utc() -> datetime:
    """Return current UTC datetime (Python 3.11+ compatible)."""
    return datetime.now(timezone.utc)


def _generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid4())


# =============================================================================
# Enumerations
# =============================================================================


class EvaluationStage(str, Enum):
    """Evaluation pipeline stages."""

    SYNTAX = "syntax"
    LOGIC = "logic"
    EXECUTION = "execution"
    SEMANTICS = "semantics"


class TaskStatus(str, Enum):
    """Status of a worker task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class TaskPriority(str, Enum):
    """Priority levels for task processing."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ErrorCode(str, Enum):
    """Structured error codes for debugging."""

    # Syntax errors
    SYNTAX_INVALID_JSON = "SYNTAX_INVALID_JSON"
    SYNTAX_SCHEMA_VIOLATION = "SYNTAX_SCHEMA_VIOLATION"

    # Logic errors
    LOGIC_TOOL_MISMATCH = "LOGIC_TOOL_MISMATCH"
    LOGIC_ARG_MISMATCH = "LOGIC_ARG_MISMATCH"
    LOGIC_MISSING_CALL = "LOGIC_MISSING_CALL"
    LOGIC_EXTRA_CALL = "LOGIC_EXTRA_CALL"

    # Execution errors
    EXEC_TIMEOUT = "EXEC_TIMEOUT"
    EXEC_API_ERROR = "EXEC_API_ERROR"
    EXEC_VALUE_MISMATCH = "EXEC_VALUE_MISMATCH"

    # Semantic errors
    SEMANTIC_LLM_ERROR = "SEMANTIC_LLM_ERROR"
    SEMANTIC_LOW_SCORE = "SEMANTIC_LOW_SCORE"

    # System errors
    SYSTEM_INTERNAL = "SYSTEM_INTERNAL"
    SYSTEM_CIRCUIT_OPEN = "SYSTEM_CIRCUIT_OPEN"
    SYSTEM_TIMEOUT = "SYSTEM_TIMEOUT"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# 1. The "Gold Standard" 4-ple Schema
# =============================================================================


class ToolCall(BaseModel):
    """
    Represents a single API/tool invocation.

    Designed for set-based comparison (order-independent).
    Arguments are stored as FrozenDict for hashability.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tool_name: str = Field(
        ...,
        min_length=1,
        description="Name of the tool/function being called",
        examples=["search_products", "get_user_profile"],
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool (key-value pairs)",
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def freeze_arguments(cls, v: dict[str, Any]) -> FrozenDict:
        """Convert arguments to FrozenDict for immutability."""
        if isinstance(v, FrozenDict):
            return v
        return FrozenDict(v)

    def __hash__(self) -> int:
        return hash((self.tool_name, FrozenDict(self.arguments)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return False
        return self.tool_name == other.tool_name and dict(self.arguments) == dict(
            other.arguments
        )

    def to_canonical_string(self) -> str:
        """Return canonical string for comparison (sorted keys)."""
        return json.dumps(
            {"tool_name": self.tool_name, "arguments": dict(self.arguments)},
            sort_keys=True,
        )


class TestCaseMetadata(BaseModel):
    """Metadata for a test case."""

    model_config = ConfigDict(frozen=True)

    api_version: str = Field(
        ...,
        pattern=r"^v?\d+\.\d+(\.\d+)?$",
        description="Semantic version of the API being tested",
        examples=["v2.1.0", "1.0.0"],
    )
    complexity_level: int = Field(
        ...,
        ge=1,
        le=5,
        description="Complexity: 1 (trivial) to 5 (complex multi-step)",
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Categorization tags for filtering",
        examples=[("search", "multi-tool", "edge-case")],
    )
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)
    author: str | None = Field(default=None, description="Creator of the test case")
    source: str | None = Field(
        default=None, description="Origin of test case (e.g., 'manual', 'generated')"
    )


class TestCase(BaseModel):
    """
    The 'Gold Standard' test case definition.

    Represents a single evaluation unit with expected inputs and outputs.
    Designed for storage in Azure AI Search with vector embeddings.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(
        default_factory=_generate_id,
        description="Unique identifier (UUID4)",
    )

    # The 4-tuple
    nl_query: str = Field(
        ...,
        min_length=1,
        description="Natural language input query",
        examples=["Find all products under $50 with free shipping"],
    )
    expected_tool_calls: tuple[ToolCall, ...] = Field(
        ...,
        min_length=1,
        description="Expected tool calls (order-independent comparison)",
    )
    expected_raw_data: dict[str, Any] | None = Field(
        default=None,
        description="Mock return data from tool execution (for execution stage)",
    )
    expected_nl_response: str = Field(
        ...,
        min_length=1,
        description="Expected natural language response/summary",
    )

    # Metadata
    metadata: TestCaseMetadata

    # Vector embedding for similarity search (Azure AI Search)
    embedding: tuple[float, ...] | None = Field(
        default=None,
        description="Vector embedding of nl_query (dimension: 1536 for ada-002)",
    )

    @field_validator("expected_tool_calls", mode="before")
    @classmethod
    def convert_to_tuple(cls, v: Any) -> tuple:
        """Ensure tool calls are stored as tuple for immutability."""
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

    def get_tool_calls_set(self) -> frozenset[ToolCall]:
        """Return tool calls as frozenset for order-independent comparison."""
        return frozenset(self.expected_tool_calls)

    @computed_field
    @property
    def content_hash(self) -> str:
        """Content hash for deduplication (excludes id and timestamps)."""
        content = {
            "nl_query": self.nl_query,
            "expected_tool_calls": [tc.to_canonical_string() for tc in self.expected_tool_calls],
            "expected_nl_response": self.expected_nl_response,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


# =============================================================================
# 2. System Response Models
# =============================================================================


class SystemResponse(BaseModel):
    """
    Captured response from the target system (MCP Server or Direct API).

    This is what the worker receives after invoking the system under test.
    """

    model_config = ConfigDict(frozen=True)

    raw_output: str = Field(
        ...,
        description="Raw string output from the target system",
    )
    parsed_tool_calls: tuple[ToolCall, ...] | None = Field(
        default=None,
        description="Parsed tool calls (None if parsing failed)",
    )
    nl_response: str | None = Field(
        default=None,
        description="Natural language response from the system",
    )
    execution_data: dict[str, Any] | None = Field(
        default=None,
        description="Data returned from actual tool execution (if enabled)",
    )
    latency_ms: int = Field(
        ...,
        ge=0,
        description="Response time in milliseconds",
    )
    error: str | None = Field(
        default=None,
        description="Error message if invocation failed",
    )


# =============================================================================
# 3. The "Scorecard" Interface & Evaluation Models
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
        """Weighted average of all stage scores."""
        weights = {
            EvaluationStage.SYNTAX: 0.1,
            EvaluationStage.LOGIC: 0.4,
            EvaluationStage.EXECUTION: 0.2,
            EvaluationStage.SEMANTICS: 0.3,
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
# 4. Worker Task Models (Service Bus Messages)
# =============================================================================


class WorkerTask(BaseModel):
    """
    Message sent to workers via Azure Service Bus.

    Contains everything needed for idempotent processing.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    task_id: str = Field(
        default_factory=_generate_id,
        description="Unique task identifier",
    )
    test_case_id: str = Field(..., description="Test case to evaluate")
    batch_id: str | None = Field(default=None, description="Parent batch")

    # Processing hints
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    created_at: datetime = Field(default_factory=_now_utc)

    # Idempotency
    idempotency_key: str | None = Field(
        default=None,
        description="Key for deduplication (auto-generated if not provided)",
    )

    # Retry tracking (updated by worker, not in original message)
    attempt_count: int = Field(default=0, ge=0)
    last_error: str | None = Field(default=None)

    @model_validator(mode="after")
    def ensure_idempotency_key(self) -> "WorkerTask":
        """Generate idempotency key if not provided."""
        if self.idempotency_key is None:
            object.__setattr__(
                self,
                "idempotency_key",
                f"{self.test_case_id}:{self.task_id}",
            )
        return self


class BatchJob(BaseModel):
    """
    Tracks a batch of test cases submitted together.

    Stored in Table Storage for status tracking.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    batch_id: str = Field(
        default_factory=_generate_id,
        description="Unique batch identifier",
    )

    # Tracking
    total_tests: int = Field(..., ge=1, description="Number of tests in batch")
    completed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    status: TaskStatus = Field(default=TaskStatus.PENDING)

    # Timestamps
    created_at: datetime = Field(default_factory=_now_utc)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    # Metadata
    submitted_by: str | None = Field(default=None, description="User/system that submitted")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    tags: tuple[str, ...] = Field(default_factory=tuple)

    @computed_field
    @property
    def progress_pct(self) -> float:
        """Completion percentage."""
        if self.total_tests == 0:
            return 0.0
        return ((self.completed_count + self.failed_count) / self.total_tests) * 100

    @computed_field
    @property
    def partition_key(self) -> str:
        """Azure Table Storage partition key (by date for efficient queries)."""
        return self.created_at.strftime("%Y-%m-%d")

    @computed_field
    @property
    def row_key(self) -> str:
        """Azure Table Storage row key."""
        return self.batch_id


# =============================================================================
# 5. Configuration Models
# =============================================================================


class EvaluationConfig(BaseModel):
    """
    Configuration for the evaluation pipeline.

    Controls which stages run and their parameters.
    """

    model_config = ConfigDict(frozen=True)

    # Stage toggles
    execution_stage_enabled: bool = Field(
        default=False,
        description="Whether to run Stage 3 (live execution)",
    )
    semantics_stage_enabled: bool = Field(
        default=True,
        description="Whether to run Stage 4 (LLM judge)",
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

    # Score weights
    syntax_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    logic_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    execution_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    semantics_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class LLMJudgeConfig(BaseModel):
    """Configuration for the LLM-as-Judge semantic evaluator."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="gpt-4o",
        description="Azure OpenAI deployment name",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 for deterministic)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Max tokens in judge response",
    )
    system_prompt: str = Field(
        default=(
            "You are an expert evaluator comparing AI-generated responses. "
            "Score the semantic similarity between the expected and actual responses. "
            "Consider meaning, completeness, and accuracy. Ignore minor wording differences."
        ),
        description="System prompt for the judge",
    )


class WorkerConfig(BaseModel):
    """Configuration for worker instances."""

    model_config = ConfigDict(frozen=True)

    # Concurrency
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    prefetch_count: int = Field(default=10, ge=1, le=100)

    # Timeouts
    task_timeout_seconds: int = Field(default=300, ge=30)
    visibility_timeout_seconds: int = Field(default=300, ge=60)

    # Circuit breaker
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_recovery_timeout_seconds: int = Field(default=30, ge=5)

    # Retry
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_backoff_base_seconds: float = Field(default=1.0, ge=0.1)


# =============================================================================
# 6. Abstract Evaluator Interface
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
# 7. Table Storage Entity Helpers
# =============================================================================


class TableStorageEntity(BaseModel):
    """
    Mixin for Azure Table Storage compatibility.

    Provides partition/row key computation.
    """

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def partition_key(self) -> str:
        """Override in subclass."""
        raise NotImplementedError

    @computed_field
    @property
    def row_key(self) -> str:
        """Override in subclass."""
        raise NotImplementedError

    def to_table_entity(self) -> dict[str, Any]:
        """Convert to Azure Table Storage entity format."""
        data = self.model_dump(mode="json")
        data["PartitionKey"] = self.partition_key
        data["RowKey"] = self.row_key
        return data


class IdempotencyRecord(BaseModel):
    """
    Record for tracking processed tasks (idempotency).

    Stored in Azure Table Storage.
    """

    model_config = ConfigDict(frozen=True)

    idempotency_key: str = Field(..., description="Unique processing key")
    scorecard_id: str = Field(..., description="Resulting scorecard ID")
    processed_at: datetime = Field(default_factory=_now_utc)
    worker_id: str = Field(..., description="Worker that processed this")

    @computed_field
    @property
    def partition_key(self) -> str:
        """Partition by first 2 chars of key for distribution."""
        return self.idempotency_key[:2] if len(self.idempotency_key) >= 2 else "00"

    @computed_field
    @property
    def row_key(self) -> str:
        """Full idempotency key as row key."""
        return self.idempotency_key


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Utility
    "FrozenDict",
    # Enums
    "EvaluationStage",
    "TaskStatus",
    "TaskPriority",
    "ErrorCode",
    "CircuitState",
    # Core Models
    "ToolCall",
    "TestCaseMetadata",
    "TestCase",
    "SystemResponse",
    # Evaluation
    "StageResult",
    "Scorecard",
    "Evaluator",
    # Worker
    "WorkerTask",
    "BatchJob",
    # Configuration
    "EvaluationConfig",
    "LLMJudgeConfig",
    "WorkerConfig",
    # Storage
    "TableStorageEntity",
    "IdempotencyRecord",
]
