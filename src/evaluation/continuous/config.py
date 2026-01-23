"""
Continuous Evaluation Configuration

Defines schedule configuration for continuous evaluation runs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScheduleConfig(BaseModel):
    """Configuration for a scheduled evaluation run."""

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str = Field(
        ...,
        min_length=1,
        description="Unique name for this schedule",
    )
    description: str | None = Field(
        default=None,
        description="Optional description of what this schedule evaluates",
    )

    # Target configuration
    pack_name: str = Field(
        default="nl2api",
        description="Evaluation pack (nl2api, rag)",
    )
    client_type: str = Field(
        default="internal",
        description="Client type to evaluate (internal, mcp_claude, etc.)",
    )
    client_version: str | None = Field(
        default=None,
        description="Optional client version",
    )
    eval_mode: str = Field(
        default="orchestrator",
        description="Evaluation mode (orchestrator, routing, resolver, tool_only)",
    )

    # Schedule
    cron_expression: str = Field(
        ...,
        description="Cron expression (e.g., '0 2 * * *' = daily at 2 AM)",
    )

    # Test selection
    test_suite_tags: list[str] = Field(
        default_factory=list,
        description="Tags to filter test cases",
    )
    test_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tests per run (None for all)",
    )

    # State
    enabled: bool = Field(
        default=True,
        description="Whether this schedule is active",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduleConfig:
        """Create ScheduleConfig from a dictionary."""
        return cls(**data)


class AlertConfig(BaseModel):
    """Configuration for alerting on regressions."""

    model_config = ConfigDict(frozen=True)

    # Webhook configuration
    webhook_url: str | None = Field(
        default=None,
        description="Webhook URL for alerts (Slack/Discord)",
    )

    # Email configuration
    email_enabled: bool = Field(
        default=False,
        description="Whether to send email alerts",
    )
    email_recipients: list[str] = Field(
        default_factory=list,
        description="Email addresses to notify",
    )
    email_smtp_host: str | None = Field(
        default=None,
        description="SMTP server hostname",
    )
    email_smtp_port: int = Field(
        default=587,
        description="SMTP server port",
    )

    # Thresholds
    pass_rate_warning: float = Field(
        default=-0.02,
        description="Pass rate drop threshold for warning (negative = regression)",
    )
    pass_rate_critical: float = Field(
        default=-0.05,
        description="Pass rate drop threshold for critical alert",
    )
    latency_warning_multiplier: float = Field(
        default=1.5,
        description="Latency increase multiplier for warning",
    )
    latency_critical_multiplier: float = Field(
        default=2.0,
        description="Latency increase multiplier for critical",
    )


class ContinuousConfig(BaseModel):
    """Main configuration for continuous evaluation."""

    model_config = ConfigDict(frozen=True)

    # Schedules
    schedules: list[ScheduleConfig] = Field(
        default_factory=list,
        description="List of evaluation schedules",
    )

    # Alerting
    alerts: AlertConfig = Field(
        default_factory=AlertConfig,
        description="Alerting configuration",
    )

    # Scheduler settings
    check_interval_seconds: int = Field(
        default=60,
        ge=10,
        description="How often to check for due schedules",
    )
    max_concurrent_runs: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent evaluation runs",
    )

    @classmethod
    def from_yaml_file(cls, path: str) -> ContinuousConfig:
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)
