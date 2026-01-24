"""
Distributed Batch Coordinator.

Coordinates distributed batch evaluation by enqueuing tasks and monitoring progress.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from src.common.storage.protocols import (
    BatchJobRepository,
    ScorecardRepository,
    TestCaseRepository,
)
from src.contracts.core import TestCase
from src.contracts.worker import WorkerTask
from src.evaluation.distributed.config import CoordinatorConfig
from src.evaluation.distributed.models import BatchProgress
from src.evaluation.distributed.queue.protocol import TaskQueue

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a distributed batch evaluation."""

    batch_id: str
    total: int
    passed: int
    failed: int
    in_dlq: int
    duration_seconds: float
    completed: bool


class BatchCoordinator:
    """
    Coordinates distributed batch evaluation.

    Responsibilities:
    - Enqueue test cases as WorkerTasks to the queue
    - Monitor batch progress via scorecard counts
    - Provide progress updates to callers
    - Handle completion detection and timeouts

    The coordinator does NOT run workers - workers are separate processes
    managed by LocalWorkerManager or external orchestration (K8s, etc.).
    """

    def __init__(
        self,
        queue: TaskQueue,
        batch_repo: BatchJobRepository,
        test_case_repo: TestCaseRepository,
        scorecard_repo: ScorecardRepository,
        config: CoordinatorConfig | None = None,
    ):
        """
        Initialize the batch coordinator.

        Args:
            queue: Task queue for distributing work
            batch_repo: Repository for batch job tracking
            test_case_repo: Repository for test case lookups
            scorecard_repo: Repository for scorecard queries
            config: Optional coordinator configuration
        """
        self._queue = queue
        self._batch_repo = batch_repo
        self._test_case_repo = test_case_repo
        self._scorecard_repo = scorecard_repo
        self._config = config or CoordinatorConfig()

    async def start_batch(
        self,
        test_cases: list[TestCase],
        batch_id: str,
        eval_mode: str = "resolver",
    ) -> int:
        """
        Enqueue test cases as WorkerTasks.

        Creates WorkerTask for each test case and adds them to the queue.
        Does NOT create BatchJob - caller is responsible for that.

        Args:
            test_cases: List of test cases to evaluate
            batch_id: Batch identifier (must already exist in DB)
            eval_mode: Evaluation mode for workers

        Returns:
            Number of tasks enqueued
        """
        if not test_cases:
            logger.warning(f"No test cases to enqueue for batch {batch_id}")
            return 0

        # Ensure stream exists
        await self._queue.ensure_stream(batch_id)

        # Create WorkerTasks
        tasks = [
            WorkerTask(
                test_case_id=tc.id,
                batch_id=batch_id,
            )
            for tc in test_cases
        ]

        # Enqueue in batches
        batch_size = 100
        total_enqueued = 0
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i : i + batch_size]
            message_ids = await self._queue.enqueue_batch(chunk, batch_id)
            total_enqueued += len(message_ids)
            logger.debug(f"Enqueued {len(message_ids)} tasks (batch chunk {i // batch_size + 1})")

        logger.info(f"Enqueued {total_enqueued} tasks for batch {batch_id}")
        return total_enqueued

    async def wait_for_completion(
        self,
        batch_id: str,
        total_tasks: int,
        timeout_seconds: float | None = None,
        poll_interval: float | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """
        Poll until all tasks are complete or timeout.

        Monitors scorecard count to determine completion. Workers save
        scorecards as they complete tasks.

        Args:
            batch_id: Batch to monitor
            total_tasks: Total number of tasks expected
            timeout_seconds: Maximum time to wait (default from config)
            poll_interval: Seconds between progress checks (default from config)
            on_progress: Optional callback for progress updates (completed, total)

        Returns:
            BatchResult with final counts

        Raises:
            TimeoutError: If batch doesn't complete within timeout
        """
        timeout = timeout_seconds or self._config.batch_timeout_seconds
        interval = poll_interval or self._config.progress_poll_interval_seconds
        start_time = datetime.now(UTC)

        last_completed = 0
        while True:
            progress = await self.get_progress(batch_id, total_tasks)

            # Report progress if changed
            if progress.completed != last_completed and on_progress:
                on_progress(progress.completed, progress.total)
                last_completed = progress.completed

            # Check completion
            if progress.is_complete:
                duration = (datetime.now(UTC) - start_time).total_seconds()
                logger.info(
                    f"Batch {batch_id} completed: {progress.passed} passed, "
                    f"{progress.failed} failed, {progress.in_dlq} in DLQ"
                )
                return BatchResult(
                    batch_id=batch_id,
                    total=progress.total,
                    passed=progress.passed,
                    failed=progress.failed,
                    in_dlq=progress.in_dlq,
                    duration_seconds=duration,
                    completed=True,
                )

            # Check timeout
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed >= timeout:
                logger.warning(
                    f"Batch {batch_id} timed out after {elapsed:.1f}s "
                    f"({progress.completed}/{progress.total} completed)"
                )
                return BatchResult(
                    batch_id=batch_id,
                    total=progress.total,
                    passed=progress.passed,
                    failed=progress.failed,
                    in_dlq=progress.in_dlq,
                    duration_seconds=elapsed,
                    completed=False,
                )

            # Wait before next poll
            await asyncio.sleep(interval)

    async def get_progress(self, batch_id: str, total_tasks: int | None = None) -> BatchProgress:
        """
        Get current batch progress.

        Queries scorecard repository for completion counts and queue for
        pending/DLQ counts.

        Args:
            batch_id: Batch to check
            total_tasks: Override total (if known), otherwise queries batch job

        Returns:
            BatchProgress with current state
        """
        # Get scorecard summary
        summary = await self._scorecard_repo.get_batch_summary(batch_id)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        completed = passed + failed

        # Get queue status
        pending = await self._queue.get_pending_count(batch_id)
        processing = await self._queue.get_processing_count(batch_id)
        in_dlq = await self._queue.get_dlq_count(batch_id)

        # Determine total
        if total_tasks is not None:
            total = total_tasks
        else:
            # Query batch job for total
            batch_job = await self._batch_repo.get(batch_id)
            total = batch_job.total_tests if batch_job else completed + pending + processing

        return BatchProgress(
            batch_id=batch_id,
            total=total,
            completed=completed,
            passed=passed,
            failed=failed,
            pending=pending + processing,
            in_dlq=in_dlq,
        )

    async def retry_failed(self, batch_id: str) -> int:
        """
        Re-enqueue failed tasks from DLQ.

        Moves all DLQ messages back to the main queue with reset attempt count.

        Args:
            batch_id: Batch to retry

        Returns:
            Number of tasks re-enqueued
        """
        dlq_messages = await self._queue.get_dlq_messages(batch_id, limit=1000)
        if not dlq_messages:
            logger.info(f"No DLQ messages to retry for batch {batch_id}")
            return 0

        retried = 0
        for message in dlq_messages:
            try:
                await self._queue.retry_from_dlq(message)
                retried += 1
            except Exception as e:
                logger.warning(f"Failed to retry message {message.message_id}: {e}")

        logger.info(f"Retried {retried}/{len(dlq_messages)} DLQ messages for batch {batch_id}")
        return retried

    async def cleanup(self, batch_id: str) -> None:
        """
        Clean up queue resources for a batch.

        Deletes the stream and DLQ. Only call after batch is complete.

        Args:
            batch_id: Batch to clean up
        """
        await self._queue.delete_stream(batch_id)
        logger.info(f"Cleaned up queue resources for batch {batch_id}")


__all__ = ["BatchCoordinator", "BatchResult"]
