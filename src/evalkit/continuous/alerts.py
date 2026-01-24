"""
Alert Handler

Handles sending alerts for regression detection results.
Supports webhook (Slack/Discord) and email notifications.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from src.common.telemetry.metrics import get_regression_alert_metrics
from src.evaluation.continuous.regression import RegressionResult, Severity

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """Alert for a detected regression."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schedule_id: str | None = None
    batch_id: str = ""
    previous_batch_id: str | None = None
    severity: Severity = Severity.WARNING
    metric_name: str = ""
    previous_value: float | None = None
    current_value: float = 0.0
    threshold_value: float = 0.0
    delta_pct: float | None = None
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    notes: str | None = None
    webhook_sent: bool = False
    email_sent: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "schedule_id": self.schedule_id,
            "batch_id": self.batch_id,
            "previous_batch_id": self.previous_batch_id,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "delta_pct": self.delta_pct,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "notes": self.notes,
            "webhook_sent": self.webhook_sent,
            "email_sent": self.email_sent,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_regression_result(
        cls,
        result: RegressionResult,
        batch_id: str,
        previous_batch_id: str | None = None,
        schedule_id: str | None = None,
    ) -> RegressionAlert:
        """Create an alert from a regression detection result."""
        return cls(
            schedule_id=schedule_id,
            batch_id=batch_id,
            previous_batch_id=previous_batch_id,
            severity=result.severity or Severity.WARNING,
            metric_name=result.metric_name,
            previous_value=result.previous_value,
            current_value=result.current_value,
            threshold_value=result.threshold,
            delta_pct=result.delta_pct,
        )


class AlertHandler:
    """
    Handles sending and persisting regression alerts.

    Supports:
    - Database persistence
    - Webhook notifications (Slack/Discord)
    - Email notifications (optional)
    """

    def __init__(
        self,
        pool: asyncpg.Pool | None = None,
        webhook_url: str | None = None,
        email_config: dict[str, Any] | None = None,
    ):
        """
        Initialize alert handler.

        Args:
            pool: Database connection pool for persistence
            webhook_url: Webhook URL for Slack/Discord notifications
            email_config: Email configuration dict with smtp settings
        """
        self.pool = pool
        self.webhook_url = webhook_url
        self.email_config = email_config or {}

    async def send(self, alert: RegressionAlert) -> bool:
        """
        Send an alert through all configured channels.

        Args:
            alert: The alert to send

        Returns:
            True if at least one channel succeeded
        """
        success = False

        # Record metrics for the alert
        metrics = get_regression_alert_metrics()
        metrics.record_alert_created(
            severity=alert.severity.value,
            metric_name=alert.metric_name,
            delta_pct=alert.delta_pct,
        )

        # 1. Persist to database
        if self.pool:
            try:
                await self._persist_alert(alert)
                success = True
            except Exception as e:
                logger.error(f"Failed to persist alert: {e}")

        # 2. Send webhook
        if self.webhook_url:
            try:
                await self._send_webhook(alert)
                alert.webhook_sent = True
                success = True
            except Exception as e:
                logger.error(f"Failed to send webhook: {e}")

        # 3. Send email (if configured)
        if self.email_config.get("enabled"):
            try:
                await self._send_email(alert)
                alert.email_sent = True
                success = True
            except Exception as e:
                logger.error(f"Failed to send email: {e}")

        return success

    async def _persist_alert(self, alert: RegressionAlert) -> None:
        """Save alert to database."""
        if not self.pool:
            return

        await self.pool.execute(
            """
            INSERT INTO regression_alerts (
                id, schedule_id, batch_id, previous_batch_id,
                severity, metric_name, previous_value, current_value,
                threshold_value, delta_pct,
                acknowledged, acknowledged_by, acknowledged_at,
                notes, webhook_sent, email_sent, created_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10,
                $11, $12, $13,
                $14, $15, $16, $17
            )
            """,
            uuid.UUID(alert.id),
            uuid.UUID(alert.schedule_id) if alert.schedule_id else None,
            alert.batch_id,
            alert.previous_batch_id,
            alert.severity.value,
            alert.metric_name,
            alert.previous_value,
            alert.current_value,
            alert.threshold_value,
            alert.delta_pct,
            alert.acknowledged,
            alert.acknowledged_by,
            alert.acknowledged_at,
            alert.notes,
            alert.webhook_sent,
            alert.email_sent,
            alert.created_at,
        )

    async def _send_webhook(self, alert: RegressionAlert) -> None:
        """Send webhook notification (Slack/Discord compatible)."""
        if not self.webhook_url:
            return

        # Format message for Slack/Discord
        severity_emoji = "🔴" if alert.severity == Severity.CRITICAL else "🟡"
        severity_text = alert.severity.value.upper()

        message = {
            "text": f"{severity_emoji} *Regression Alert: {severity_text}*",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{severity_emoji} Regression Alert: {severity_text}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Metric:*\n{alert.metric_name}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Batch ID:*\n{alert.batch_id[:12]}...",
                        },
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"*Previous:*\n{alert.previous_value:.4f}"
                                if alert.previous_value
                                else "*Previous:*\nN/A"
                            ),
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Current:*\n{alert.current_value:.4f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"*Change:*\n{alert.delta_pct:+.2f}%"
                                if alert.delta_pct
                                else "*Change:*\nN/A"
                            ),
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Threshold:*\n{alert.threshold_value:.4f}",
                        },
                    ],
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"Alert ID: `{alert.id[:12]}` | "
                                f"{alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                            ),
                        },
                    ],
                },
            ],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
            response.raise_for_status()

    async def _send_email(self, alert: RegressionAlert) -> None:
        """Send email notification."""
        # Email implementation would go here
        # For now, just log that email would be sent
        recipients = self.email_config.get("recipients", [])
        if recipients:
            logger.info(
                f"Would send email for alert {alert.id} to {recipients} "
                f"(email sending not implemented)"
            )

    async def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: str | None = None,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: User/system acknowledging the alert
            notes: Optional notes about the acknowledgment

        Returns:
            True if acknowledgment succeeded
        """
        if not self.pool:
            return False

        try:
            # First get the alert details for metrics
            row = await self.pool.fetchrow(
                "SELECT severity, metric_name FROM regression_alerts WHERE id = $1",
                uuid.UUID(alert_id),
            )

            result = await self.pool.execute(
                """
                UPDATE regression_alerts
                SET acknowledged = true,
                    acknowledged_by = $2,
                    acknowledged_at = $3,
                    notes = $4
                WHERE id = $1
                """,
                uuid.UUID(alert_id),
                acknowledged_by,
                datetime.now(UTC),
                notes,
            )

            if result == "UPDATE 1" and row:
                # Record acknowledgment metric
                metrics = get_regression_alert_metrics()
                metrics.record_alert_acknowledged(
                    severity=row["severity"],
                    metric_name=row["metric_name"],
                )
                return True

            return result == "UPDATE 1"
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False

    async def get_unacknowledged(
        self,
        days: int = 7,
        limit: int = 100,
    ) -> list[RegressionAlert]:
        """
        Get unacknowledged alerts from the last N days.

        Args:
            days: Number of days to look back
            limit: Maximum alerts to return

        Returns:
            List of unacknowledged alerts
        """
        if not self.pool:
            return []

        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(days=days)

        rows = await self.pool.fetch(
            """
            SELECT * FROM regression_alerts
            WHERE acknowledged = false
              AND created_at >= $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            cutoff,
            limit,
        )

        alerts = []
        for row in rows:
            alerts.append(
                RegressionAlert(
                    id=str(row["id"]),
                    schedule_id=str(row["schedule_id"]) if row["schedule_id"] else None,
                    batch_id=row["batch_id"],
                    previous_batch_id=row["previous_batch_id"],
                    severity=Severity(row["severity"]),
                    metric_name=row["metric_name"],
                    previous_value=float(row["previous_value"]) if row["previous_value"] else None,
                    current_value=float(row["current_value"]),
                    threshold_value=float(row["threshold_value"]),
                    delta_pct=float(row["delta_pct"]) if row["delta_pct"] else None,
                    acknowledged=row["acknowledged"],
                    acknowledged_by=row["acknowledged_by"],
                    acknowledged_at=row["acknowledged_at"],
                    notes=row["notes"],
                    webhook_sent=row["webhook_sent"],
                    email_sent=row["email_sent"],
                    created_at=row["created_at"],
                )
            )

        return alerts
