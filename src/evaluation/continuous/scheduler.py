"""
Evaluation Scheduler

Cron-based scheduler for continuous evaluation runs.
Manages scheduled evaluations, triggers batch runs, and integrates
with regression detection and alerting.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.evaluation.continuous.alerts import AlertHandler, RegressionAlert
from src.evaluation.continuous.config import ContinuousConfig, ScheduleConfig
from src.evaluation.continuous.regression import RegressionDetector

if TYPE_CHECKING:
    import asyncpg

    from src.common.storage.protocols import (
        BatchJobRepository,
        ScorecardRepository,
        TestCaseRepository,
    )

logger = logging.getLogger(__name__)


def parse_cron_expression(cron_expr: str) -> dict[str, set[int]]:
    """
    Parse a cron expression into its components.

    Supports: minute, hour, day_of_month, month, day_of_week
    Supports: numbers, ranges (1-5), lists (1,3,5), and wildcards (*)

    Args:
        cron_expr: Cron expression (e.g., "0 2 * * *")

    Returns:
        Dictionary with sets of valid values for each field
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Invalid cron expression: {cron_expr}")

    field_names = ["minute", "hour", "day_of_month", "month", "day_of_week"]
    field_ranges = [
        (0, 59),  # minute
        (0, 23),  # hour
        (1, 31),  # day of month
        (1, 12),  # month
        (0, 6),  # day of week (0 = Sunday)
    ]

    result = {}
    for i, (field, (min_val, max_val)) in enumerate(zip(fields, field_ranges)):
        result[field_names[i]] = _parse_cron_field(field, min_val, max_val)

    return result


def _parse_cron_field(field: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field."""
    if field == "*":
        return set(range(min_val, max_val + 1))

    values = set()
    for part in field.split(","):
        if "-" in part:
            # Range
            start, end = part.split("-")
            values.update(range(int(start), int(end) + 1))
        elif "/" in part:
            # Step
            base, step = part.split("/")
            if base == "*":
                start = min_val
            else:
                start = int(base)
            values.update(range(start, max_val + 1, int(step)))
        else:
            values.add(int(part))

    return values


def should_run_now(cron_expr: str, now: datetime | None = None) -> bool:
    """
    Check if a cron schedule should run at the given time.

    Args:
        cron_expr: Cron expression
        now: Time to check (defaults to current UTC time)

    Returns:
        True if the schedule should run now
    """
    if now is None:
        now = datetime.now(UTC)

    parsed = parse_cron_expression(cron_expr)

    return (
        now.minute in parsed["minute"]
        and now.hour in parsed["hour"]
        and now.day in parsed["day_of_month"]
        and now.month in parsed["month"]
        and now.weekday() in parsed["day_of_week"]  # Python weekday: 0=Monday
    )


class EvalScheduler:
    """
    Manages scheduled evaluation runs.

    Checks schedules periodically, triggers batch evaluations,
    runs regression detection, and sends alerts.
    """

    def __init__(
        self,
        config: ContinuousConfig,
        test_case_repo: TestCaseRepository,
        scorecard_repo: ScorecardRepository,
        batch_repo: BatchJobRepository,
        pool: asyncpg.Pool | None = None,
    ):
        """
        Initialize scheduler.

        Args:
            config: Continuous evaluation configuration
            test_case_repo: Repository for test cases
            scorecard_repo: Repository for scorecards
            batch_repo: Repository for batch jobs
            pool: Database connection pool
        """
        self.config = config
        self.test_case_repo = test_case_repo
        self.scorecard_repo = scorecard_repo
        self.batch_repo = batch_repo
        self.pool = pool

        # Initialize components
        self.regression_detector = RegressionDetector(scorecard_repo)
        self.alert_handler = AlertHandler(
            pool=pool,
            webhook_url=config.alerts.webhook_url,
            email_config={
                "enabled": config.alerts.email_enabled,
                "recipients": config.alerts.email_recipients,
            },
        )

        # State
        self._running = False
        self._current_runs: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """
        Start the scheduler loop.

        Runs until stop() is called.
        """
        self._running = True
        logger.info("Starting evaluation scheduler")

        while self._running:
            try:
                await self._check_schedules()
            except Exception as e:
                logger.error(f"Error checking schedules: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        logger.info("Stopping evaluation scheduler")
        self._running = False

    async def trigger(self, schedule_name: str) -> str | None:
        """
        Manually trigger a schedule.

        Args:
            schedule_name: Name of the schedule to trigger

        Returns:
            Batch ID if triggered, None if schedule not found
        """
        for schedule in self.config.schedules:
            if schedule.name == schedule_name:
                return await self._run_schedule(schedule)

        logger.warning(f"Schedule not found: {schedule_name}")
        return None

    def get_status(self) -> dict[str, Any]:
        """
        Get current scheduler status.

        Returns:
            Status dictionary with schedule info
        """
        return {
            "running": self._running,
            "check_interval_seconds": self.config.check_interval_seconds,
            "schedules": [
                {
                    "name": s.name,
                    "enabled": s.enabled,
                    "cron": s.cron_expression,
                    "client_type": s.client_type,
                    "running": s.name in self._current_runs,
                }
                for s in self.config.schedules
            ],
            "active_runs": len(self._current_runs),
        }

    async def _check_schedules(self) -> None:
        """Check all schedules and trigger due ones."""
        now = datetime.now(UTC)

        for schedule in self.config.schedules:
            if not schedule.enabled:
                continue

            if schedule.name in self._current_runs:
                # Already running
                continue

            if len(self._current_runs) >= self.config.max_concurrent_runs:
                # At capacity
                continue

            if should_run_now(schedule.cron_expression, now):
                # Trigger the schedule
                task = asyncio.create_task(self._run_schedule_with_cleanup(schedule))
                self._current_runs[schedule.name] = task

    async def _run_schedule_with_cleanup(self, schedule: ScheduleConfig) -> None:
        """Run a schedule and clean up when done."""
        try:
            await self._run_schedule(schedule)
        finally:
            self._current_runs.pop(schedule.name, None)

    async def _run_schedule(self, schedule: ScheduleConfig) -> str | None:
        """
        Run a scheduled evaluation.

        Args:
            schedule: Schedule configuration

        Returns:
            Batch ID of the completed run
        """
        logger.info(f"Running scheduled evaluation: {schedule.name}")

        try:
            # Import here to avoid circular imports
            from src.evaluation.batch import BatchRunner, BatchRunnerConfig

            # Create runner config
            runner_config = BatchRunnerConfig(
                max_concurrency=10,
                show_progress=False,
                verbose=False,
                client_type=schedule.client_type,
                client_version=schedule.client_version,
                eval_mode=schedule.eval_mode,
            )

            # Create runner
            runner = BatchRunner(
                test_case_repo=self.test_case_repo,
                scorecard_repo=self.scorecard_repo,
                batch_repo=self.batch_repo,
                config=runner_config,
            )

            # Run evaluation
            batch_job = await runner.run(
                tags=schedule.test_suite_tags if schedule.test_suite_tags else None,
                limit=schedule.test_limit,
            )

            if batch_job is None:
                logger.warning(f"No test cases found for schedule: {schedule.name}")
                return None

            batch_id = batch_job.batch_id
            logger.info(f"Completed scheduled evaluation: {schedule.name} (batch: {batch_id})")

            # Run regression detection
            await self._check_for_regressions(batch_id, schedule)

            return batch_id

        except Exception as e:
            logger.error(f"Error running schedule {schedule.name}: {e}")
            return None

    async def _check_for_regressions(
        self,
        batch_id: str,
        schedule: ScheduleConfig,
    ) -> None:
        """
        Check for regressions and send alerts if found.

        Args:
            batch_id: ID of the completed batch
            schedule: Schedule that was run
        """
        try:
            results = await self.regression_detector.detect_regressions(
                current_batch_id=batch_id,
                client_type=schedule.client_type,
            )

            for result in results:
                if result.is_regression:
                    alert = RegressionAlert.from_regression_result(
                        result,
                        batch_id=batch_id,
                    )
                    await self.alert_handler.send(alert)
                    logger.warning(
                        f"Regression detected: {result.metric_name} "
                        f"({result.severity.value if result.severity else 'unknown'})"
                    )

        except Exception as e:
            logger.error(f"Error checking for regressions: {e}")
