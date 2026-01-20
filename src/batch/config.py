"""
Batch Runner Configuration

Configuration settings for batch evaluation runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BatchRunnerConfig(BaseModel):
    """Configuration for batch evaluation runs."""

    model_config = ConfigDict(frozen=True)

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
