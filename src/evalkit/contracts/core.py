"""
Core Data Models

Fundamental types used throughout the NL2API system.
Includes utility types, enumerations, and the "Gold Standard" test case models.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
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
    return datetime.now(UTC)


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


class TestCaseStatus(str, Enum):
    """Lifecycle status of a test case."""

    ACTIVE = "active"  # Current, valid, running in suites
    STALE = "stale"  # Detected drift, needs review
    DEPRECATED = "deprecated"  # Old API version, kept for regression
    ARCHIVED = "archived"  # No longer relevant, excluded from runs


class ClientType(str, Enum):
    """Type of client being evaluated."""

    INTERNAL_ORCHESTRATOR = "internal"  # Internal NL2API orchestrator
    MCP_CLAUDE = "mcp_claude"  # Claude via MCP server
    MCP_CHATGPT = "mcp_chatgpt"  # ChatGPT via MCP server
    MCP_CUSTOM = "mcp_custom"  # Custom MCP client


class EvalMode(str, Enum):
    """Evaluation mode - what component is being tested."""

    ORCHESTRATOR = "orchestrator"  # Full end-to-end orchestrator
    TOOL_ONLY = "tool_only"  # Single agent/tool evaluation
    ROUTING_ONLY = "routing"  # Router decision evaluation
    RESOLVER_ONLY = "resolver"  # Entity resolution evaluation
    MCP_PASSTHROUGH = "mcp_passthrough"  # MCP server passthrough


class TemporalStability(str, Enum):
    """How stable are expected values over time."""

    EVERGREEN = "evergreen"  # Never changes (sector, industry, static data)
    ABSOLUTE = "absolute"  # Fixed historical (FY2024, 2024-01-01)
    RELATIVE = "relative"  # Shifts with eval date (-1D, -1M, FQ0)
    POINT_IN_TIME = "point_in_time"  # Requires specific data snapshot


class TemporalValidationMode(str, Enum):
    """What aspect of temporal handling to validate."""

    BEHAVIORAL = "behavioral"  # Only validate both are valid temporal expressions
    STRUCTURAL = "structural"  # Normalize dates to absolute, then compare
    DATA = "data"  # Compare actual data values (snapshot required)


# =============================================================================
# Tool Registry
# =============================================================================


class ToolRegistry:
    """
    Central registry of all tool names used by agents and expected in fixtures.

    This ensures agents and test fixtures stay in sync. All tool names must:
    - Match pattern: ^[a-zA-Z0-9_-]{1,128}$ (Anthropic API requirement)
    - Be defined here as the single source of truth

    Usage:
        from src.evalkit.contracts.core import ToolRegistry
        name=ToolRegistry.DATASTREAM_GET_DATA
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
        """Normalize a tool name to its canonical form for comparison."""
        return cls._TO_CANONICAL.get(tool_name, tool_name)

    @classmethod
    def is_valid(cls, tool_name: str) -> bool:
        """Check if a tool name is valid (known to the registry)."""
        return tool_name in cls._TO_CANONICAL

    @classmethod
    def is_api_compliant(cls, tool_name: str) -> bool:
        """Check if tool name matches Anthropic API requirements."""
        import re

        return bool(re.match(r"^[a-zA-Z0-9_-]{1,128}$", tool_name))


# =============================================================================
# Core Models
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
        return self.tool_name == other.tool_name and dict(self.arguments) == dict(other.arguments)

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


class TemporalContext(BaseModel):
    """
    Temporal context for test case evaluation.

    Enables temporally-aware evaluation where relative date expressions
    (like "-1D", "FQ0") are normalized before comparison.
    """

    model_config = ConfigDict(frozen=True)

    stability: TemporalStability = Field(
        default=TemporalStability.EVERGREEN,
        description="How stable are expected values over time",
    )
    validation_mode: TemporalValidationMode = Field(
        default=TemporalValidationMode.STRUCTURAL,
        description="What aspect of temporal handling to validate",
    )
    assertion_date: date | None = Field(
        default=None,
        description="When expected values were valid (for POINT_IN_TIME stability)",
    )
    snapshot_id: str | None = Field(
        default=None,
        description="Reference to data snapshot (for DATA validation mode)",
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


class TestCase(BaseModel):
    """
    The 'Gold Standard' test case definition.

    Represents a single evaluation unit with expected inputs and outputs.
    Designed for storage in Azure AI Search with vector embeddings.

    GENERIC FIELDS (for general-purpose evaluation):
    - input: Arbitrary input data (dict)
    - expected: Arbitrary expected output data (dict)

    NL2API-SPECIFIC FIELDS (backwards compatible):
    - nl_query: Natural language input
    - expected_tool_calls: Expected tool calls
    - expected_response: Expected API response data
    - expected_nl_response: Expected NL summary

    Use generic fields for new evaluation packs. NL2API fields are
    automatically populated from generic fields when using NL2APIPack.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(
        default_factory=_generate_id,
        description="Unique identifier (UUID4)",
    )

    # ==========================================================================
    # GENERIC FIELDS (for general-purpose evaluation framework)
    # ==========================================================================
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Generic input data. Pack-specific schema.",
        examples=[{"query": "What is Apple's stock price?"}],
    )
    expected: dict[str, Any] = Field(
        default_factory=dict,
        description="Generic expected output. Pack-specific schema.",
        examples=[{"tool_calls": [], "nl_response": "Apple's price is $150"}],
    )

    # ==========================================================================
    # NL2API-SPECIFIC FIELDS (backwards compatible, use generic fields for new packs)
    # ==========================================================================
    # DEPRECATION NOTICE: For new evaluation packs (RAG, code-gen, etc.), use
    # the generic `input` and `expected` fields instead. These NL2API-specific
    # fields are retained for backwards compatibility with existing NL2API code.
    # See docs/evaluation-test-case-patterns.md for recommended patterns.
    nl_query: str | None = Field(
        default=None,
        description="Natural language input query. DEPRECATED for new packs - use input['nl_query'] instead.",
        examples=["Find all products under $50 with free shipping"],
        deprecated=True,
    )
    expected_tool_calls: tuple[ToolCall, ...] = Field(
        default_factory=tuple,
        description="Expected tool calls. DEPRECATED for new packs - use expected['tool_calls'] instead.",
        deprecated=True,
    )
    expected_response: dict[str, Any] | None = Field(
        default=None,
        description="Expected structured API response. DEPRECATED for new packs - use expected['response'] instead.",
        examples=[{"AAPL.O": {"P": 246.02, "MV": 3850000000000}}],
        deprecated=True,
    )
    expected_nl_response: str | None = Field(
        default=None,
        description="Expected natural language summary. DEPRECATED for new packs - use expected['nl_response'] instead.",
        examples=["Apple's stock price is $246.02 with a market cap of $3.85 trillion."],
        deprecated=True,
    )

    # Metadata
    metadata: TestCaseMetadata | None = Field(
        default=None,
        description="Test case metadata (optional for generic test cases)",
    )

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

    # Temporal context (optional, backward compatible)
    temporal_context: TemporalContext | None = Field(
        default=None,
        description="Temporal context for evaluation (None = evergreen, exact match)",
    )

    @field_validator("expected_tool_calls", mode="before")
    @classmethod
    def convert_to_tuple(cls, v: Any) -> tuple:
        """Ensure tool calls are stored as tuple for immutability."""
        if v is None:
            return ()
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
        # Use generic fields if populated, otherwise fall back to NL2API fields
        if self.input or self.expected:
            content = {
                "input": self.input,
                "expected": self.expected,
            }
        else:
            content = {
                "nl_query": self.nl_query or "",
                "expected_tool_calls": [
                    tc.to_canonical_string() for tc in self.expected_tool_calls
                ],
                "expected_nl_response": self.expected_nl_response or "",
            }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]

    def to_generic(self) -> TestCase:
        """Convert NL2API-specific fields to generic format.

        Returns a new TestCase with populated generic fields.
        Useful for migration or pack-agnostic processing.
        """
        if self.input and self.expected:
            # Already in generic format
            return self

        generic_input = dict(self.input) if self.input else {}
        generic_expected = dict(self.expected) if self.expected else {}

        # Populate from NL2API fields if not already set
        if self.nl_query and "nl_query" not in generic_input:
            generic_input["nl_query"] = self.nl_query

        if self.expected_tool_calls and "tool_calls" not in generic_expected:
            generic_expected["tool_calls"] = [
                {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                for tc in self.expected_tool_calls
            ]

        if self.expected_response and "response" not in generic_expected:
            generic_expected["response"] = self.expected_response

        if self.expected_nl_response and "nl_response" not in generic_expected:
            generic_expected["nl_response"] = self.expected_nl_response

        return self.model_copy(update={"input": generic_input, "expected": generic_expected})

    @classmethod
    def from_generic(
        cls,
        id: str,
        input: dict[str, Any],
        expected: dict[str, Any],
        metadata: TestCaseMetadata | None = None,
        **kwargs: Any,
    ) -> TestCase:
        """Create a TestCase from generic input/expected dicts.

        Automatically extracts NL2API-specific fields if present in the dicts.
        """
        # Extract NL2API fields from generic dicts if present
        nl_query = input.get("nl_query")
        expected_tool_calls = ()
        if "tool_calls" in expected:
            expected_tool_calls = tuple(
                ToolCall(tool_name=tc["tool_name"], arguments=tc.get("arguments", {}))
                for tc in expected["tool_calls"]
            )
        expected_response = expected.get("response")
        expected_nl_response = expected.get("nl_response")

        return cls(
            id=id,
            input=input,
            expected=expected,
            nl_query=nl_query,
            expected_tool_calls=expected_tool_calls,
            expected_response=expected_response,
            expected_nl_response=expected_nl_response,
            metadata=metadata,
            **kwargs,
        )


class TestCaseSetConfig(BaseModel):
    """
    Configuration for a test case set defining required fields and metadata.

    Embedded in fixture files as the '_meta' key. The fixture loader validates
    each test case against this configuration.
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

    # Token usage tracking (for cost calculation)
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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Utility
    "FrozenDict",
    "_now_utc",
    "_generate_id",
    # Enums
    "EvaluationStage",
    "TaskStatus",
    "TaskPriority",
    "ErrorCode",
    "CircuitState",
    "TestCaseStatus",
    "ClientType",
    "EvalMode",
    "TemporalStability",
    "TemporalValidationMode",
    # Registry
    "ToolRegistry",
    # Core Models
    "ToolCall",
    "TestCaseMetadata",
    "TemporalContext",
    "TestCase",
    "TestCaseSetConfig",
    "SystemResponse",
]
