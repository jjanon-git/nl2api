"""
Batch Runner Configuration

Configuration settings for batch evaluation runs.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class BatchRunnerConfig(BaseModel):
    """Configuration for batch evaluation runs."""

    model_config = ConfigDict(frozen=True)

    # Pack selection (REQUIRED - no default)
    pack_name: str = Field(
        ...,  # Required - no default
        description="Evaluation pack to use (nl2api, rag)",
    )

    max_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent evaluations",
    )
    show_progress: bool = Field(
        default=True,
        description="Show progress bar during execution",
    )
    verbose: bool = Field(
        default=False,
        description="Show detailed output for each test",
    )
    checkpoint_interval: int = Field(
        default=10,
        ge=0,
        description="Save progress every N evaluations (0 to disable checkpointing)",
    )

    # Client tracking (multi-client evaluation)
    client_type: str = Field(
        default="internal",
        description="Type of client (internal, mcp_claude, mcp_chatgpt, mcp_custom)",
    )
    client_version: str | None = Field(
        default=None,
        description="Client version identifier (e.g., claude-opus-4.5-20251101)",
    )
    eval_mode: str = Field(
        default="orchestrator",
        description="Evaluation mode (orchestrator, tool_only, routing, resolver)",
    )

    # Semantics stage configuration
    semantics_enabled: bool = Field(
        default=False,
        description="Enable LLM-as-Judge semantic evaluation (Stage 4)",
    )
    semantics_model: str | None = Field(
        default=None,
        description="Override model for semantic evaluation (default: claude-3-5-haiku-20241022)",
    )
    semantics_pass_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum score to pass semantic evaluation",
    )

    # Temporal evaluation configuration
    evaluation_date: date | None = Field(
        default=None,
        description="Reference date for temporal normalization (defaults to today)",
    )
    temporal_mode: str = Field(
        default="structural",
        description="Temporal validation mode: behavioral, structural, or data",
    )

    # Run tracking (for experiment management)
    run_label: str = Field(
        default="untracked",
        description="Label for this evaluation run (e.g., 'baseline', 'new-embedder-v2')",
    )
    run_description: str | None = Field(
        default=None,
        description="Optional longer description of what change is being tested",
    )
    git_commit: str | None = Field(
        default=None,
        description="Git commit hash at time of run (auto-captured)",
    )
    git_branch: str | None = Field(
        default=None,
        description="Git branch at time of run (auto-captured)",
    )
