"""
Distributed Evaluation Worker.

The EvalWorker processes tasks from the queue, running evaluations and
saving results. Designed for fault tolerance:
- Graceful shutdown on SIGTERM/SIGINT
- Automatic retry of failed tasks
- Heartbeat reporting for health monitoring
- OTEL telemetry integration
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from src.evalkit.contracts.core import _now_utc
from src.evalkit.contracts.worker import WorkerTask
from src.evalkit.distributed.config import WorkerConfig
from src.evalkit.distributed.models import QueueMessage, WorkerStatus
from src.evalkit.distributed.queue.protocol import TaskQueue

# Telemetry imports (optional)
try:
    from src.evalkit.common.telemetry import get_meter, get_tracer, trace_span_safe

    tracer = get_tracer(__name__)
    meter = get_meter(__name__)
    TELEMETRY_AVAILABLE = True

    # Worker metrics
    _tasks_processed_counter = meter.create_counter(
        "eval_worker_tasks_processed",
        description="Number of tasks processed by workers",
    )
    _tasks_failed_counter = meter.create_counter(
        "eval_worker_tasks_failed",
        description="Number of tasks that failed processing",
    )
    _task_duration_histogram = meter.create_histogram(
        "eval_worker_task_duration_ms",
        description="Time to process a single task in milliseconds",
    )
    _worker_active_gauge = meter.create_up_down_counter(
        "eval_worker_active",
        description="Number of active workers",
    )
except ImportError:
    tracer = None
    meter = None
    TELEMETRY_AVAILABLE = False
    _tasks_processed_counter = None
    _tasks_failed_counter = None
    _task_duration_histogram = None
    _worker_active_gauge = None

    def trace_span_safe(*args, **kwargs):
        """No-op decorator when telemetry not available."""

        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


# Type aliases
ResponseGenerator = Callable[[WorkerTask, Any], Awaitable[Any]]
Evaluator = Callable[[Any, Any], Awaitable[Any]]
ScorecardSaver = Callable[[Any], Awaitable[None]]
TestCaseFetcher = Callable[[str], Awaitable[Any]]


class EvalWorker:
    """
    Distributed evaluation worker.

    Pulls tasks from a queue, generates responses, evaluates them,
    and saves scorecards. Handles graceful shutdown and retries.

    Example:
        async def main():
            queue = await create_queue(config)
            worker = EvalWorker(
                worker_id="worker-0",
                queue=queue,
                batch_id="batch-001",
                response_generator=my_generator,
                evaluator=my_evaluator,
                scorecard_saver=my_saver,
                test_case_fetcher=my_fetcher,
            )
            await worker.run()
    """

    def __init__(
        self,
        worker_id: str,
        queue: TaskQueue,
        batch_id: str,
        response_generator: ResponseGenerator,
        evaluator: Evaluator,
        scorecard_saver: ScorecardSaver,
        test_case_fetcher: TestCaseFetcher,
        config: WorkerConfig | None = None,
    ):
        """
        Initialize the worker.

        Args:
            worker_id: Unique identifier for this worker
            queue: Task queue to pull from
            batch_id: Batch to process
            response_generator: Function to generate response for a test case
            evaluator: Function to evaluate response against test case
            scorecard_saver: Function to save scorecard to storage
            test_case_fetcher: Function to fetch test case by ID
            config: Worker configuration
        """
        self.worker_id = worker_id
        self.queue = queue
        self.batch_id = batch_id
        self.response_generator = response_generator
        self.evaluator = evaluator
        self.scorecard_saver = scorecard_saver
        self.test_case_fetcher = test_case_fetcher
        self.config = config or WorkerConfig(worker_id=worker_id)

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._current_task: QueueMessage | None = None
        self._status = WorkerStatus(worker_id=worker_id, status="idle")

        # Stats
        self._tasks_processed = 0
        self._tasks_failed = 0
        self._started_at: Any = None

    @property
    def status(self) -> WorkerStatus:
        """Get current worker status."""
        return WorkerStatus(
            worker_id=self.worker_id,
            status="running" if self._running else "stopped",
            tasks_processed=self._tasks_processed,
            tasks_failed=self._tasks_failed,
            last_heartbeat=_now_utc(),
            current_task_id=self._current_task.task_id if self._current_task else None,
            current_batch_id=self.batch_id if self._running else None,
        )

    async def run(self) -> None:
        """
        Main worker loop.

        Processes tasks until shutdown signal or queue is empty.
        Handles SIGTERM/SIGINT for graceful shutdown.
        """
        self._running = True
        self._started_at = _now_utc()

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown_signal)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        logger.info(f"Worker {self.worker_id} starting for batch {self.batch_id}")

        # Track worker as active
        if _worker_active_gauge:
            _worker_active_gauge.add(1, {"worker_id": self.worker_id, "batch_id": self.batch_id})

        try:
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Main processing loop
            await self._process_loop()

        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} cancelled")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
            raise
        finally:
            self._running = False
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            # Track worker as inactive
            if _worker_active_gauge:
                _worker_active_gauge.add(
                    -1, {"worker_id": self.worker_id, "batch_id": self.batch_id}
                )

            logger.info(
                f"Worker {self.worker_id} stopped. "
                f"Processed: {self._tasks_processed}, Failed: {self._tasks_failed}"
            )

    async def _process_loop(self) -> None:
        """Main loop that processes tasks from the queue."""
        consumer = self.queue.consume(
            consumer_id=self.worker_id,
            batch_id=self.batch_id,
            block_ms=5000,
        )

        async for message in consumer:
            if self._shutdown_event.is_set():
                # Graceful shutdown - nack current message for reprocessing
                logger.info(f"Shutdown requested, requeueing message {message.message_id}")
                await self.queue.nack(message, requeue=True)
                break

            self._current_task = message
            task_start = _now_utc()
            labels = {
                "worker_id": self.worker_id,
                "batch_id": self.batch_id,
                "eval_mode": self.config.eval_mode.value,
            }

            try:
                await self._process_task(message)
                await self.queue.ack(message)
                self._tasks_processed += 1

                # Record success metrics
                if _tasks_processed_counter:
                    _tasks_processed_counter.add(1, {**labels, "status": "success"})
                if _task_duration_histogram:
                    duration_ms = (_now_utc() - task_start).total_seconds() * 1000
                    _task_duration_histogram.record(duration_ms, labels)

            except Exception as e:
                self._tasks_failed += 1
                logger.error(
                    f"Task {message.task_id} failed: {e}",
                    exc_info=True,
                )

                # Record failure metrics
                if _tasks_failed_counter:
                    _tasks_failed_counter.add(1, {**labels, "error_type": type(e).__name__})
                await self.queue.nack(
                    message,
                    requeue=True,
                    error=str(e)[:500],  # Truncate error message
                )
            finally:
                self._current_task = None

    async def _process_task(self, message: QueueMessage) -> None:
        """
        Process a single task.

        1. Fetch the test case
        2. Generate response
        3. Evaluate
        4. Save scorecard
        """
        test_case_id = message.test_case_id
        if not test_case_id:
            raise ValueError(f"Message {message.message_id} has no test_case_id")

        async with self._span("eval.worker.process_task") as span:
            if span:
                span.set_attribute("worker_id", self.worker_id)
                span.set_attribute("task_id", message.task_id or "")
                span.set_attribute("test_case_id", test_case_id)
                span.set_attribute("attempt", message.attempt)

            # 1. Fetch test case
            async with self._span("eval.worker.fetch_test_case"):
                test_case = await self.test_case_fetcher(test_case_id)
                if not test_case:
                    raise ValueError(f"Test case {test_case_id} not found")

            # 2. Generate response (this may call LLM)
            async with self._span("eval.worker.generate_response"):
                # Convert message payload back to WorkerTask if needed
                worker_task = WorkerTask.model_validate(message.payload)
                response = await self.response_generator(worker_task, test_case)

            # 3. Evaluate
            async with self._span("eval.worker.evaluate") as eval_span:
                scorecard = await self.evaluator(test_case, response)
                if eval_span and scorecard:
                    eval_span.set_attribute(
                        "result.passed", getattr(scorecard, "overall_passed", False)
                    )
                    eval_span.set_attribute(
                        "result.score", getattr(scorecard, "overall_score", 0.0)
                    )

            # 4. Save scorecard
            async with self._span("eval.worker.save_scorecard"):
                await self.scorecard_saver(scorecard)

            logger.debug(
                f"Processed task {message.task_id}: "
                f"passed={getattr(scorecard, 'overall_passed', 'N/A')}"
            )

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat for health monitoring."""
        interval = self.config.heartbeat_interval_seconds

        while self._running and not self._shutdown_event.is_set():
            try:
                # Log heartbeat (could also publish to Redis/metrics)
                logger.debug(
                    f"Worker {self.worker_id} heartbeat: "
                    f"processed={self._tasks_processed}, failed={self._tasks_failed}"
                )

                # TODO: Publish to Redis for coordinator monitoring
                # await self._publish_heartbeat()

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    def _handle_shutdown_signal(self) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        logger.info(f"Worker {self.worker_id} received shutdown signal")
        self._shutdown_event.set()

    async def shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info(f"Worker {self.worker_id} shutdown requested")
        self._shutdown_event.set()

        # Wait for current task to complete (with timeout)
        timeout = self.config.shutdown_timeout_seconds
        start = _now_utc()

        while self._current_task is not None:
            elapsed = (_now_utc() - start).total_seconds()
            if elapsed > timeout:
                logger.warning(
                    f"Worker {self.worker_id} shutdown timeout, "
                    f"task {self._current_task.task_id} may be requeued"
                )
                break
            await asyncio.sleep(0.5)

    @asynccontextmanager
    async def _span(self, name: str):
        """Create a tracing span if telemetry is available."""
        if tracer and TELEMETRY_AVAILABLE:
            with tracer.start_as_current_span(name) as span:
                yield span
        else:
            yield None


async def run_worker(
    worker_id: str,
    queue: TaskQueue,
    batch_id: str,
    response_generator: ResponseGenerator,
    evaluator: Evaluator,
    scorecard_saver: ScorecardSaver,
    test_case_fetcher: TestCaseFetcher,
    config: WorkerConfig | None = None,
) -> None:
    """
    Convenience function to create and run a worker.

    Args:
        worker_id: Unique identifier for this worker
        queue: Task queue to pull from
        batch_id: Batch to process
        response_generator: Function to generate response for a test case
        evaluator: Function to evaluate response against test case
        scorecard_saver: Function to save scorecard to storage
        test_case_fetcher: Function to fetch test case by ID
        config: Worker configuration
    """
    worker = EvalWorker(
        worker_id=worker_id,
        queue=queue,
        batch_id=batch_id,
        response_generator=response_generator,
        evaluator=evaluator,
        scorecard_saver=scorecard_saver,
        test_case_fetcher=test_case_fetcher,
        config=config,
    )
    await worker.run()


__all__ = [
    "EvalWorker",
    "run_worker",
    "ResponseGenerator",
    "Evaluator",
    "ScorecardSaver",
    "TestCaseFetcher",
]
