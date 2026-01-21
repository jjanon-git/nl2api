"""
NL2API Contracts - Pydantic v2 Schemas

This module defines all data contracts for the NL2API system and evaluation framework.
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


# =============================================================================
# Tool Registry - Single Source of Truth for Tool Names
# =============================================================================


class ToolRegistry:
    """
    Central registry of all tool names used by agents and expected in fixtures.

    This ensures agents and test fixtures stay in sync. All tool names must:
    - Match pattern: ^[a-zA-Z0-9_-]{1,128}$ (Anthropic API requirement)
    - Be defined here as the single source of truth

    Usage:
        # In agents:
        from CONTRACTS import ToolRegistry
        name=ToolRegistry.DATASTREAM_GET_DATA

        # In generators:
        from CONTRACTS import ToolRegistry
        "function": ToolRegistry.DATASTREAM_GET_DATA

        # For comparison (handles legacy names):
        canonical = ToolRegistry.normalize("datastream.get_data")  # -> "get_data"
    """

    # Canonical tool names (used in fixtures for portability)
    GET_DATA = "get_data"
    SCREEN = "screen"

    # Agent-specific tool names (prefixed for API clarity)
    DATASTREAM_GET_DATA = "datastream_get_data"
    ESTIMATES_GET_DATA = "estimates_get_data"
    FUNDAMENTALS_GET_DATA = "refinitiv_get_data"
    OFFICERS_GET_DATA = "refinitiv_get_data"
    SCREENING_GET_DATA = "refinitiv_get_data"

    # Mapping from agent tool names to canonical names
    _TO_CANONICAL: ClassVar[dict[str, str]] = {
        "datastream_get_data": "get_data",
        "datastream.get_data": "get_data",  # Legacy
        "refinitiv_get_data": "get_data",
        "refinitiv.get_data": "get_data",  # Legacy
        "estimates_get_data": "get_data",
        "get_data": "get_data",
    }

    @classmethod
    def normalize(cls, tool_name: str) -> str:
        """
        Normalize a tool name to its canonical form for comparison.

        Args:
            tool_name: Any tool name (agent-specific or canonical)

        Returns:
            Canonical tool name for comparison
        """
        return cls._TO_CANONICAL.get(tool_name, tool_name)

    @classmethod
    def is_valid(cls, tool_name: str) -> bool:
        """Check if a tool name is valid (known to the registry)."""
        return tool_name in cls._TO_CANONICAL

    @classmethod
    def is_api_compliant(cls, tool_name: str) -> bool:
        """Check if tool name matches Anthropic API requirements."""
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]{1,128}$', tool_name))


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


class TestCaseStatus(str, Enum):
    """Lifecycle status of a test case."""

    ACTIVE = "active"  # Current, valid, running in suites
    STALE = "stale"  # Detected drift, needs review
    DEPRECATED = "deprecated"  # Old API version, kept for regression
    ARCHIVED = "archived"  # No longer relevant, excluded from runs


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
    expected_response: dict[str, Any] | None = Field(
        default=None,
        description="Expected structured data response from API execution",
        examples=[{"AAPL.O": {"P": 246.02, "MV": 3850000000000}}],
    )
    expected_nl_response: str | None = Field(
        default=None,
        description="Expected natural language summary of the response",
        examples=["Apple's stock price is $246.02 with a market cap of $3.85 trillion."],
    )

    # Metadata
    metadata: TestCaseMetadata

    # Lifecycle status
    status: TestCaseStatus = Field(
        default=TestCaseStatus.ACTIVE,
        description="Lifecycle status for staleness tracking",
    )
    stale_reason: str | None = Field(
        default=None,
        description="Reason for staleness (if status=STALE)",
    )

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
            "expected_nl_response": self.expected_nl_response or "",
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


class TestCaseSetConfig(BaseModel):
    """
    Configuration for a test case set defining required fields and metadata.

    Embedded in fixture files as the '_meta' key. The fixture loader validates
    each test case against this configuration.

    Different evaluation capabilities have different field requirements:
    - nl2api: requires expected_nl_response
    - entity_extraction: no NL response needed
    - tool_generation: no NL response needed
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str = Field(
        ...,
        description="Human-readable name for this test case set",
        examples=["lookups", "entity_extraction_us_equities"],
    )
    capability: str = Field(
        ...,
        description="The capability this set evaluates",
        examples=["nl2api", "entity_extraction", "tool_generation"],
    )
    description: str | None = Field(
        default=None,
        description="Optional description of what this test set covers",
    )

    # Field requirements - which fields must be present
    requires_nl_response: bool = Field(
        default=True,
        description="Whether expected_nl_response is required for test cases in this set",
    )
    requires_expected_response: bool = Field(
        default=False,
        description="Whether expected_response is required for test cases in this set",
    )

    # Generation metadata
    schema_version: str = Field(
        default="1.0",
        description="Schema version for forward compatibility",
    )
    generated_at: datetime | None = Field(
        default=None,
        description="When this fixture set was generated",
    )
    generator: str | None = Field(
        default=None,
        description="Script/tool that generated this set",
        examples=["scripts/generate_test_cases.py"],
    )


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
# 5. Test Suite & Client Management
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers for the orchestrator."""

    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"  # For self-hosted or other providers


class TargetSystemConfig(BaseModel):
    """
    Configuration for the LLM orchestrator being evaluated.

    This is the "system under test" - the LLM that receives nl_query,
    decides which tools to call, and generates the NL response.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(default_factory=_generate_id)
    name: str = Field(..., min_length=1, description="Human-readable name")

    # Provider configuration
    provider: LLMProvider = Field(default=LLMProvider.AZURE_OPENAI)
    model: str = Field(
        ...,
        description="Model name or deployment ID",
        examples=["gpt-4o", "gpt-4-turbo", "claude-3-opus"],
    )
    endpoint: str | None = Field(
        default=None,
        description="API endpoint (required for Azure, optional for others)",
    )
    api_version: str | None = Field(
        default=None,
        description="API version (for Azure OpenAI)",
    )

    # Model parameters
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the orchestrator",
    )

    # Tool configuration
    available_tools: tuple[str, ...] = Field(
        default_factory=tuple,
        description="List of tool names the LLM can call",
    )
    tool_schema_version: str = Field(
        default="v1",
        description="Version of the tool schema definitions",
    )

    # Metadata
    created_at: datetime = Field(default_factory=_now_utc)
    is_active: bool = Field(default=True)


class TestSuite(BaseModel):
    """
    A named collection of test cases.

    Allows organizing tests by API, feature, complexity, or any criteria.
    Clients are assigned one or more test suites.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(default_factory=_generate_id)
    name: str = Field(..., min_length=1, description="Suite name", examples=["search-api-v2", "checkout-flow"])
    description: str | None = Field(default=None)

    # Test case references (IDs, not full objects for efficiency)
    test_case_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of test cases in this suite",
    )

    # Filtering criteria (alternative to explicit IDs)
    # If set, dynamically includes test cases matching these filters
    filter_tags: tuple[str, ...] | None = Field(
        default=None,
        description="Include test cases with ANY of these tags",
    )
    filter_api_version: str | None = Field(
        default=None,
        description="Include test cases for this API version",
    )
    filter_min_complexity: int | None = Field(default=None, ge=1, le=5)
    filter_max_complexity: int | None = Field(default=None, ge=1, le=5)

    # Metadata
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)
    owner: str | None = Field(default=None, description="Team or person responsible")
    is_active: bool = Field(default=True)

    @computed_field
    @property
    def is_dynamic(self) -> bool:
        """True if suite uses filters instead of explicit IDs."""
        return any([
            self.filter_tags,
            self.filter_api_version,
            self.filter_min_complexity is not None,
            self.filter_max_complexity is not None,
        ])


class Client(BaseModel):
    """
    A tenant/consumer of the evaluation platform.

    Each client has assigned test suites and a target system configuration.
    Enables multi-tenant operation with different test sets per client.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(default_factory=_generate_id)
    name: str = Field(..., min_length=1, description="Client/tenant name")
    description: str | None = Field(default=None)

    # Assigned resources
    test_suite_ids: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Test suites this client runs",
    )
    target_system_id: str = Field(
        ...,
        description="The LLM orchestrator config to evaluate",
    )
    evaluation_config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Override default EvaluationConfig settings",
    )

    # Scheduling
    schedule_cron: str | None = Field(
        default=None,
        description="Cron expression for scheduled runs",
        examples=["0 2 * * *"],  # Daily at 2 AM
    )
    notify_emails: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Email addresses for run notifications",
    )
    notify_on_failure_only: bool = Field(default=True)

    # Metadata
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)
    is_active: bool = Field(default=True)


class RunStatus(str, Enum):
    """Status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationRun(BaseModel):
    """
    A specific execution of a test suite against a target system.

    Tracks the full context: what was tested, how, and results summary.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(default_factory=_generate_id)

    # What is being evaluated
    client_id: str = Field(..., description="Client that triggered this run")
    test_suite_id: str = Field(..., description="Suite being executed")
    target_system_id: str = Field(..., description="LLM orchestrator being tested")

    # Configuration snapshot (captured at run start for reproducibility)
    evaluation_config: dict[str, Any] = Field(
        default_factory=dict,
        description="EvaluationConfig as dict (snapshot)",
    )
    target_system_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="TargetSystemConfig as dict (snapshot)",
    )

    # Execution tracking
    status: RunStatus = Field(default=RunStatus.PENDING)
    total_tests: int = Field(default=0, ge=0)
    completed_tests: int = Field(default=0, ge=0)
    passed_tests: int = Field(default=0, ge=0)
    failed_tests: int = Field(default=0, ge=0)

    # Timestamps
    created_at: datetime = Field(default_factory=_now_utc)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    # Trigger context
    triggered_by: str = Field(
        default="manual",
        description="How run was triggered",
        examples=["manual", "scheduled", "api", "ci-pipeline"],
    )
    trigger_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (CI job ID, commit SHA, etc.)",
    )

    # Results summary (computed after completion)
    overall_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    average_score: float | None = Field(default=None, ge=0.0, le=1.0)
    stage_pass_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Pass rate per stage: {syntax: 0.99, logic: 0.95, ...}",
    )

    @computed_field
    @property
    def progress_pct(self) -> float:
        """Completion percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.completed_tests / self.total_tests) * 100

    @computed_field
    @property
    def partition_key(self) -> str:
        """Azure Table Storage partition key (by client for efficient queries)."""
        return self.client_id

    @computed_field
    @property
    def row_key(self) -> str:
        """Azure Table Storage row key (run ID)."""
        return self.id


# =============================================================================
# 6. Configuration Models
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
    "TestCaseStatus",
    "LLMProvider",
    "RunStatus",
    # Core Models
    "ToolCall",
    "TestCaseMetadata",
    "TestCase",
    "TestCaseSetConfig",
    "SystemResponse",
    # Evaluation
    "StageResult",
    "Scorecard",
    "Evaluator",
    # Worker
    "WorkerTask",
    "BatchJob",
    # Management
    "TargetSystemConfig",
    "TestSuite",
    "Client",
    "EvaluationRun",
    # Configuration
    "EvaluationConfig",
    "LLMJudgeConfig",
    "WorkerConfig",
    # Storage
    "TableStorageEntity",
    "IdempotencyRecord",
]
