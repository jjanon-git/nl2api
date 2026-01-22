"""
Unit tests for evaluation scheduler.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.continuous.config import ContinuousConfig, ScheduleConfig
from src.evaluation.continuous.scheduler import (
    EvalScheduler,
    parse_cron_expression,
    should_run_now,
)


class TestCronParsing:
    """Tests for cron expression parsing."""

    def test_parse_simple_cron(self):
        """Test parsing a simple cron expression."""
        result = parse_cron_expression("0 2 * * *")

        assert result["minute"] == {0}
        assert result["hour"] == {2}
        assert result["day_of_month"] == set(range(1, 32))
        assert result["month"] == set(range(1, 13))
        assert result["day_of_week"] == set(range(0, 7))

    def test_parse_cron_with_range(self):
        """Test parsing cron with range."""
        result = parse_cron_expression("0 9-17 * * *")

        assert result["minute"] == {0}
        assert result["hour"] == set(range(9, 18))

    def test_parse_cron_with_list(self):
        """Test parsing cron with list."""
        result = parse_cron_expression("0,30 * * * *")

        assert result["minute"] == {0, 30}

    def test_parse_cron_with_step(self):
        """Test parsing cron with step."""
        result = parse_cron_expression("*/15 * * * *")

        assert result["minute"] == {0, 15, 30, 45}

    def test_parse_invalid_cron(self):
        """Test parsing invalid cron expression."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            parse_cron_expression("invalid")

    def test_parse_cron_too_few_fields(self):
        """Test parsing cron with too few fields."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            parse_cron_expression("0 2 *")


class TestShouldRunNow:
    """Tests for should_run_now function."""

    def test_should_run_at_matching_time(self):
        """Test that schedule runs at matching time."""
        # 2 AM on any day
        now = datetime(2024, 1, 15, 2, 0, tzinfo=timezone.utc)
        assert should_run_now("0 2 * * *", now) is True

    def test_should_not_run_wrong_minute(self):
        """Test schedule doesn't run at wrong minute."""
        now = datetime(2024, 1, 15, 2, 1, tzinfo=timezone.utc)
        assert should_run_now("0 2 * * *", now) is False

    def test_should_not_run_wrong_hour(self):
        """Test schedule doesn't run at wrong hour."""
        now = datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc)
        assert should_run_now("0 2 * * *", now) is False

    def test_should_run_every_30_minutes(self):
        """Test schedule runs every 30 minutes."""
        cron = "0,30 * * * *"

        assert should_run_now(cron, datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)) is True
        assert should_run_now(cron, datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)) is True
        assert should_run_now(cron, datetime(2024, 1, 15, 10, 15, tzinfo=timezone.utc)) is False


class TestScheduleConfig:
    """Tests for ScheduleConfig model."""

    def test_schedule_config_creation(self):
        """Test creating a schedule config."""
        config = ScheduleConfig(
            name="daily-internal",
            cron_expression="0 2 * * *",
            client_type="internal",
            test_suite_tags=["routing"],
        )

        assert config.name == "daily-internal"
        assert config.cron_expression == "0 2 * * *"
        assert config.client_type == "internal"
        assert config.enabled is True

    def test_schedule_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test-schedule",
            "cron_expression": "0 * * * *",
            "client_type": "mcp_claude",
            "enabled": False,
        }

        config = ScheduleConfig.from_dict(data)

        assert config.name == "test-schedule"
        assert config.enabled is False


class TestEvalScheduler:
    """Tests for EvalScheduler class."""

    @pytest.fixture
    def mock_repos(self):
        """Create mock repositories."""
        test_case_repo = MagicMock()
        scorecard_repo = MagicMock()
        scorecard_repo.get_batch_summary = AsyncMock(return_value={"total": 0})
        scorecard_repo.get_by_batch = AsyncMock(return_value=[])
        batch_repo = MagicMock()
        return test_case_repo, scorecard_repo, batch_repo

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ContinuousConfig(
            schedules=[
                ScheduleConfig(
                    name="test-schedule",
                    cron_expression="0 * * * *",
                    client_type="internal",
                ),
            ],
            check_interval_seconds=60,
        )

    @pytest.fixture
    def scheduler(self, config, mock_repos):
        """Create a scheduler instance."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos
        return EvalScheduler(
            config=config,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
        )

    def test_scheduler_creation(self, scheduler):
        """Test creating a scheduler."""
        assert scheduler._running is False
        assert len(scheduler.config.schedules) == 1

    def test_get_status(self, scheduler):
        """Test getting scheduler status."""
        status = scheduler.get_status()

        assert status["running"] is False
        assert len(status["schedules"]) == 1
        assert status["schedules"][0]["name"] == "test-schedule"
        assert status["active_runs"] == 0

    @pytest.mark.asyncio
    async def test_trigger_unknown_schedule(self, scheduler):
        """Test triggering an unknown schedule."""
        result = await scheduler.trigger("unknown-schedule")
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_known_schedule(self, scheduler, mock_repos):
        """Test triggering a known schedule."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        # Mock the batch runner to return a batch job
        with patch("src.evaluation.batch.BatchRunner") as MockRunner:
            mock_batch_job = MagicMock()
            mock_batch_job.batch_id = "test-batch-123"
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_batch_job)
            MockRunner.return_value = mock_instance

            result = await scheduler.trigger("test-schedule")

        assert result == "test-batch-123"

    def test_stop(self, scheduler):
        """Test stopping the scheduler."""
        scheduler._running = True
        scheduler.stop()
        assert scheduler._running is False


class TestContinuousConfig:
    """Tests for ContinuousConfig model."""

    def test_continuous_config_defaults(self):
        """Test default configuration values."""
        config = ContinuousConfig()

        assert config.schedules == []
        assert config.check_interval_seconds == 60
        assert config.max_concurrent_runs == 2

    def test_continuous_config_with_schedules(self):
        """Test configuration with schedules."""
        config = ContinuousConfig(
            schedules=[
                ScheduleConfig(
                    name="schedule-1",
                    cron_expression="0 2 * * *",
                ),
                ScheduleConfig(
                    name="schedule-2",
                    cron_expression="0 14 * * *",
                ),
            ],
            max_concurrent_runs=4,
        )

        assert len(config.schedules) == 2
        assert config.max_concurrent_runs == 4


class TestCronParsingEdgeCases:
    """Edge case tests for cron expression parsing."""

    def test_parse_cron_midnight(self):
        """Test parsing cron for midnight."""
        result = parse_cron_expression("0 0 * * *")
        assert result["minute"] == {0}
        assert result["hour"] == {0}

    def test_parse_cron_last_minute_of_hour(self):
        """Test parsing cron for last minute of hour."""
        result = parse_cron_expression("59 * * * *")
        assert result["minute"] == {59}

    def test_parse_cron_all_weekdays(self):
        """Test parsing cron for weekdays (Mon-Fri)."""
        result = parse_cron_expression("0 9 * * 1-5")
        assert result["day_of_week"] == {1, 2, 3, 4, 5}

    def test_parse_cron_complex_expression(self):
        """Test parsing complex cron with multiple ranges and steps."""
        result = parse_cron_expression("0,30 9-17 1,15 * 1-5")
        assert result["minute"] == {0, 30}
        assert result["hour"] == set(range(9, 18))
        assert result["day_of_month"] == {1, 15}
        assert result["day_of_week"] == {1, 2, 3, 4, 5}

    def test_parse_cron_step_from_specific_value(self):
        """Test parsing cron step starting from specific value."""
        result = parse_cron_expression("5/10 * * * *")
        # Starting from 5, every 10 minutes: 5, 15, 25, 35, 45, 55
        assert result["minute"] == {5, 15, 25, 35, 45, 55}

    def test_parse_cron_last_day_of_month(self):
        """Test parsing cron for 31st (last day of month)."""
        result = parse_cron_expression("0 0 31 * *")
        assert result["day_of_month"] == {31}

    def test_parse_cron_december(self):
        """Test parsing cron for December only."""
        result = parse_cron_expression("0 0 * 12 *")
        assert result["month"] == {12}

    def test_parse_cron_sunday(self):
        """Test parsing cron for Sunday (0)."""
        result = parse_cron_expression("0 0 * * 0")
        assert result["day_of_week"] == {0}


class TestShouldRunNowEdgeCases:
    """Edge case tests for should_run_now function."""

    def test_should_run_at_midnight_new_year(self):
        """Test schedule runs at midnight on New Year's."""
        now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert should_run_now("0 0 1 1 *", now) is True

    def test_should_run_on_leap_year_feb_29(self):
        """Test schedule on leap year February 29."""
        now = datetime(2024, 2, 29, 0, 0, tzinfo=timezone.utc)  # 2024 is a leap year
        assert should_run_now("0 0 29 2 *", now) is True

    def test_should_not_run_on_non_leap_year_feb_29(self):
        """Test schedule doesn't exist for Feb 29 in non-leap year."""
        # February 29 doesn't exist in 2025, so this just tests the pattern
        now = datetime(2025, 2, 28, 0, 0, tzinfo=timezone.utc)
        assert should_run_now("0 0 29 2 *", now) is False

    def test_should_run_last_minute_of_day(self):
        """Test schedule at last minute of day."""
        now = datetime(2024, 1, 15, 23, 59, tzinfo=timezone.utc)
        assert should_run_now("59 23 * * *", now) is True

    def test_should_run_with_complex_day_pattern(self):
        """Test schedule with day-of-week constraint."""
        # Monday (weekday 0 in Python)
        monday = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
        # Check if it's actually a Monday
        assert monday.weekday() == 0

        # The scheduler uses Python weekday values directly (0=Monday)
        # So 0 in cron expression = Monday in this scheduler
        assert should_run_now("0 9 * * 0", monday) is True

    def test_should_not_run_on_weekend_with_weekday_schedule(self):
        """Test weekday schedule doesn't run on weekend."""
        # Saturday (weekday 5 in Python)
        saturday = datetime(2024, 1, 13, 9, 0, tzinfo=timezone.utc)
        assert saturday.weekday() == 5

        # Schedule for Monday (0 in scheduler which uses Python weekday)
        assert should_run_now("0 9 * * 0", saturday) is False


class TestEvalSchedulerEdgeCases:
    """Edge case tests for EvalScheduler."""

    @pytest.fixture
    def mock_repos(self):
        """Create mock repositories."""
        test_case_repo = MagicMock()
        scorecard_repo = MagicMock()
        scorecard_repo.get_batch_summary = AsyncMock(return_value={"total": 0})
        scorecard_repo.get_by_batch = AsyncMock(return_value=[])
        batch_repo = MagicMock()
        return test_case_repo, scorecard_repo, batch_repo

    def test_scheduler_with_no_schedules(self, mock_repos):
        """Test scheduler with empty schedule list."""
        config = ContinuousConfig(schedules=[])
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        scheduler = EvalScheduler(
            config=config,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
        )

        status = scheduler.get_status()
        assert status["schedules"] == []
        assert status["active_runs"] == 0

    def test_scheduler_with_disabled_schedules(self, mock_repos):
        """Test scheduler status shows disabled schedules."""
        config = ContinuousConfig(
            schedules=[
                ScheduleConfig(
                    name="disabled-schedule",
                    cron_expression="0 2 * * *",
                    enabled=False,
                ),
            ],
        )
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        scheduler = EvalScheduler(
            config=config,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
        )

        status = scheduler.get_status()
        assert len(status["schedules"]) == 1
        assert status["schedules"][0]["enabled"] is False

    @pytest.mark.asyncio
    async def test_trigger_disabled_schedule(self, mock_repos):
        """Test triggering a disabled schedule still works."""
        config = ContinuousConfig(
            schedules=[
                ScheduleConfig(
                    name="disabled-schedule",
                    cron_expression="0 2 * * *",
                    enabled=False,
                ),
            ],
        )
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        scheduler = EvalScheduler(
            config=config,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
        )

        # Manual trigger should still work even if disabled
        with patch("src.evaluation.batch.BatchRunner") as MockRunner:
            mock_batch_job = MagicMock()
            mock_batch_job.batch_id = "test-batch-456"
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_batch_job)
            MockRunner.return_value = mock_instance

            result = await scheduler.trigger("disabled-schedule")

        assert result == "test-batch-456"
