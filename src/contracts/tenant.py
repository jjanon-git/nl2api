"""
Tenant Models

Models for multi-tenant operation: clients, test suites, target systems, and runs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.contracts.core import _generate_id, _now_utc

# =============================================================================
# Enumerations
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers for the orchestrator."""

    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"  # For self-hosted or other providers


class RunStatus(str, Enum):
    """Status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Target System Configuration
# =============================================================================


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


# =============================================================================
# Test Suite
# =============================================================================


class TestSuite(BaseModel):
    """
    A named collection of test cases.

    Allows organizing tests by API, feature, complexity, or any criteria.
    Clients are assigned one or more test suites.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(default_factory=_generate_id)
    name: str = Field(
        ..., min_length=1, description="Suite name", examples=["search-api-v2", "checkout-flow"]
    )
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
        return any(
            [
                self.filter_tags,
                self.filter_api_version,
                self.filter_min_complexity is not None,
                self.filter_max_complexity is not None,
            ]
        )


# =============================================================================
# Client
# =============================================================================


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


# =============================================================================
# Evaluation Run
# =============================================================================


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
# Exports
# =============================================================================

__all__ = [
    "LLMProvider",
    "RunStatus",
    "TargetSystemConfig",
    "TestSuite",
    "Client",
    "EvaluationRun",
]
