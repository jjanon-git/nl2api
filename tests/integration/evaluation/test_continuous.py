"""
Integration tests for continuous evaluation module.

Tests alert persistence and regression detection against a real database.
Requires: docker compose up -d
"""

import uuid
from datetime import UTC, datetime

import pytest

import pytest_asyncio

from src.evaluation.continuous.alerts import AlertHandler, RegressionAlert
from src.evaluation.continuous.regression import RegressionResult, Severity


@pytest_asyncio.fixture(loop_scope="session")
async def alert_handler(db_pool):
    """Create an alert handler with database connection."""
    return AlertHandler(pool=db_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def cleanup_alerts(db_pool):
    """Clean up test alerts after each test."""
    yield
    async with db_pool.acquire() as conn:
        # Clean up test alerts (those with 'test-' prefix in batch_id)
        await conn.execute(
            "DELETE FROM regression_alerts WHERE batch_id LIKE 'test-%'"
        )


@pytest.mark.asyncio(loop_scope="session")
class TestAlertHandlerIntegration:
    """Integration tests for AlertHandler with real database."""

    async def test_persist_and_retrieve_alert(self, alert_handler, cleanup_alerts):
        """Test saving and retrieving an alert."""
        alert = RegressionAlert(
            batch_id=f"test-{uuid.uuid4()}",
            severity=Severity.WARNING,
            metric_name="pass_rate",
            previous_value=0.95,
            current_value=0.90,
            threshold_value=-0.02,
            delta_pct=-5.26,
        )

        # Save alert
        success = await alert_handler.send(alert)
        assert success is True

        # Retrieve unacknowledged alerts
        alerts = await alert_handler.get_unacknowledged(days=1)
        matching = [a for a in alerts if a.batch_id == alert.batch_id]

        assert len(matching) == 1
        retrieved = matching[0]
        assert retrieved.severity == Severity.WARNING
        assert retrieved.metric_name == "pass_rate"
        assert retrieved.previous_value == pytest.approx(0.95, abs=0.001)
        assert retrieved.current_value == pytest.approx(0.90, abs=0.001)

    async def test_acknowledge_alert(self, alert_handler, cleanup_alerts):
        """Test acknowledging an alert."""
        alert = RegressionAlert(
            batch_id=f"test-{uuid.uuid4()}",
            severity=Severity.CRITICAL,
            metric_name="avg_latency_ms",
            current_value=500.0,
            threshold_value=1.5,
        )

        await alert_handler.send(alert)

        # Acknowledge
        success = await alert_handler.acknowledge(
            alert.id,
            acknowledged_by="test-user",
            notes="Investigated, was expected spike",
        )
        assert success is True

        # Should not appear in unacknowledged
        alerts = await alert_handler.get_unacknowledged(days=1)
        matching = [a for a in alerts if a.id == alert.id]
        assert len(matching) == 0

    async def test_alert_from_regression_result(self, alert_handler, cleanup_alerts):
        """Test creating alert from regression result."""
        result = RegressionResult(
            metric_name="pass_rate",
            previous_value=0.95,
            current_value=0.88,
            threshold=-0.05,
            delta=-0.07,
            delta_pct=-7.37,
            is_regression=True,
            severity=Severity.CRITICAL,
            p_value=0.01,
        )

        batch_id = f"test-{uuid.uuid4()}"
        alert = RegressionAlert.from_regression_result(
            result,
            batch_id=batch_id,
            previous_batch_id=f"test-prev-{uuid.uuid4()}",
        )

        await alert_handler.send(alert)

        alerts = await alert_handler.get_unacknowledged(days=1)
        matching = [a for a in alerts if a.batch_id == batch_id]

        assert len(matching) == 1
        assert matching[0].severity == Severity.CRITICAL
        assert matching[0].delta_pct == pytest.approx(-7.37, abs=0.01)

    async def test_multiple_alerts_ordering(self, alert_handler, cleanup_alerts):
        """Test that alerts are returned in order by creation time."""
        alerts_created = []

        for i in range(3):
            alert = RegressionAlert(
                batch_id=f"test-order-{i}-{uuid.uuid4()}",
                severity=Severity.WARNING,
                metric_name="pass_rate",
                current_value=0.9 - i * 0.05,
                threshold_value=-0.02,
            )
            await alert_handler.send(alert)
            alerts_created.append(alert.batch_id)

        # Retrieve and check ordering (most recent first)
        alerts = await alert_handler.get_unacknowledged(days=1, limit=10)
        test_alerts = [a for a in alerts if a.batch_id.startswith("test-order-")]

        assert len(test_alerts) >= 3
        # Most recent should be first
        batch_ids = [a.batch_id for a in test_alerts[:3]]
        assert batch_ids == list(reversed(alerts_created))


class TestAlertHandlerWithoutDatabase:
    """Tests for AlertHandler without database connection."""

    @pytest.mark.asyncio
    async def test_handler_without_pool(self):
        """Test that handler works gracefully without database."""
        handler = AlertHandler(pool=None)

        alert = RegressionAlert(
            batch_id="test-no-db",
            severity=Severity.WARNING,
            metric_name="pass_rate",
            current_value=0.9,
            threshold_value=-0.02,
        )

        # Should not crash, but return False (no channels succeeded)
        success = await handler.send(alert)
        assert success is False

        # Should return empty list
        alerts = await handler.get_unacknowledged()
        assert alerts == []
