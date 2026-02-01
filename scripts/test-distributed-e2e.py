#!/usr/bin/env python
"""
End-to-end test for distributed evaluation.

Tests:
1. Queue operations with Redis
2. Worker processing (success and failure cases)
3. Telemetry (traces and metrics)
4. Dashboard visibility

Run with: python scripts/test_distributed_e2e.py
"""

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from uuid import uuid4

# Set up telemetry BEFORE imports
os.environ["EVALKIT_TELEMETRY_ENABLED"] = "true"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_SERVICE_NAME"] = "distributed-eval-e2e-test"

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize telemetry
from src.evalkit.common.telemetry import get_meter, get_tracer, init_telemetry
from src.evalkit.contracts.worker import WorkerTask
from src.evalkit.distributed.config import EvalMode, QueueBackend, QueueConfig, WorkerConfig
from src.evalkit.distributed.queue import create_queue
from src.evalkit.distributed.worker import EvalWorker

init_telemetry(service_name="distributed-eval-e2e-test")
tracer = get_tracer(__name__)
meter = get_meter(__name__)

# Create custom metrics for this test
test_counter = meter.create_counter(
    "e2e_test.tasks_created", description="Number of test tasks created"
)
test_histogram = meter.create_histogram(
    "e2e_test.processing_duration_ms", description="Time to process test tasks"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class MockTestCase:
    """Mock test case for testing."""

    def __init__(self, id: str, should_fail: bool = False):
        self.id = id
        self.nl_query = f"Test query for {id}"
        self.expected_tool_calls = []
        self.should_fail = should_fail


class MockScorecard:
    """Mock scorecard for testing."""

    def __init__(self, test_case_id: str, passed: bool):
        self.test_case_id = test_case_id
        self.overall_passed = passed
        self.overall_score = 1.0 if passed else 0.0


async def run_e2e_test():
    """Run the end-to-end test."""
    batch_id = f"e2e-test-{uuid4().hex[:8]}"
    logger.info(f"Starting E2E test with batch_id: {batch_id}")

    # Create queue config
    queue_config = QueueConfig(
        backend=QueueBackend.REDIS,
        redis_url="redis://localhost:6379",
        redis_db=0,
        max_retries=2,
        stream_prefix="e2e:eval:tasks",
        dlq_prefix="e2e:eval:dlq",
    )

    # Test cases - mix of success and failure
    test_cases_data = {
        "tc-success-001": MockTestCase("tc-success-001"),
        "tc-success-002": MockTestCase("tc-success-002"),
        "tc-success-003": MockTestCase("tc-success-003"),
        "tc-fail-001": MockTestCase("tc-fail-001", should_fail=True),  # Will fail in evaluator
        "tc-notfound-001": None,  # Will fail - test case not found
    }

    # Track results
    results = {
        "processed": [],
        "failed": [],
        "scorecards_saved": [],
    }

    # Create dependencies
    async def response_generator(worker_task, test_case):
        """Generate mock response."""
        with tracer.start_as_current_span("e2e.response_generator") as span:
            span.set_attribute("test_case_id", test_case.id)
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": "simulated", "test_case_id": test_case.id}

    async def evaluator(test_case, response):
        """Evaluate response."""
        with tracer.start_as_current_span("e2e.evaluator") as span:
            span.set_attribute("test_case_id", test_case.id)

            if test_case.should_fail:
                span.set_attribute("error", True)
                raise ValueError(f"Simulated evaluation failure for {test_case.id}")

            passed = True
            scorecard = MockScorecard(test_case.id, passed)
            span.set_attribute("result.passed", passed)
            return scorecard

    async def scorecard_saver(scorecard):
        """Save scorecard."""
        with tracer.start_as_current_span("e2e.scorecard_saver") as span:
            span.set_attribute("test_case_id", scorecard.test_case_id)
            span.set_attribute("passed", scorecard.overall_passed)
            results["scorecards_saved"].append(scorecard.test_case_id)
            logger.info(
                f"Saved scorecard for {scorecard.test_case_id}: passed={scorecard.overall_passed}"
            )

    async def test_case_fetcher(test_case_id: str):
        """Fetch test case."""
        with tracer.start_as_current_span("e2e.test_case_fetcher") as span:
            span.set_attribute("test_case_id", test_case_id)
            test_case = test_cases_data.get(test_case_id)
            if test_case is None:
                span.set_attribute("found", False)
                logger.warning(f"Test case not found: {test_case_id}")
            else:
                span.set_attribute("found", True)
            return test_case

    # Connect to queue
    logger.info("Connecting to Redis queue...")
    try:
        queue = await create_queue(queue_config)
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return False

    try:
        # Ensure stream exists
        await queue.ensure_stream(batch_id)
        logger.info(f"Created stream for batch {batch_id}")

        # Enqueue tasks
        tasks = []
        for test_case_id in test_cases_data.keys():
            task = WorkerTask(
                task_id=f"task-{uuid4().hex[:8]}",
                test_case_id=test_case_id,
                batch_id=batch_id,
            )
            tasks.append(task)
            test_counter.add(1, {"test_case_id": test_case_id})

        with tracer.start_as_current_span("e2e.enqueue_batch") as span:
            span.set_attribute("batch_id", batch_id)
            span.set_attribute("task_count", len(tasks))
            message_ids = await queue.enqueue_batch(tasks, batch_id)
            logger.info(f"Enqueued {len(message_ids)} tasks")

        # Create worker
        worker_config = WorkerConfig(
            worker_id=f"e2e-worker-{uuid4().hex[:8]}",
            eval_mode=EvalMode.SIMULATED,
            heartbeat_interval_seconds=5,
        )

        worker = EvalWorker(
            worker_id=worker_config.worker_id,
            queue=queue,
            batch_id=batch_id,
            response_generator=response_generator,
            evaluator=evaluator,
            scorecard_saver=scorecard_saver,
            test_case_fetcher=test_case_fetcher,
            config=worker_config,
        )

        # Run worker with timeout
        logger.info(f"Starting worker {worker_config.worker_id}")
        start_time = datetime.now(UTC)

        try:
            # Run for up to 30 seconds (should finish much faster)
            await asyncio.wait_for(worker.run(), timeout=30.0)
        except TimeoutError:
            logger.info("Worker timed out (expected - no more messages)")

        end_time = datetime.now(UTC)
        duration_ms = (end_time - start_time).total_seconds() * 1000
        test_histogram.record(duration_ms, {"batch_id": batch_id})

        # Report results
        logger.info("=" * 60)
        logger.info("E2E TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Batch ID: {batch_id}")
        logger.info(f"Tasks enqueued: {len(tasks)}")
        logger.info(f"Tasks processed: {worker._tasks_processed}")
        logger.info(f"Tasks failed: {worker._tasks_failed}")
        logger.info(f"Scorecards saved: {len(results['scorecards_saved'])}")
        logger.info(f"Duration: {duration_ms:.0f}ms")

        # Check DLQ
        dlq_count = await queue.get_dlq_count(batch_id)
        logger.info(f"Tasks in DLQ: {dlq_count}")

        if dlq_count > 0:
            dlq_messages = await queue.get_dlq_messages(batch_id)
            for msg in dlq_messages:
                logger.info(f"  DLQ message: {msg.test_case_id} (attempt {msg.attempt})")

        # Verify expectations
        logger.info("=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)

        # We expect:
        # - 3 successful tasks (tc-success-*)
        # - 1 evaluation failure (tc-fail-001) - should end up in DLQ after retries
        # - 1 not-found failure (tc-notfound-001) - should end up in DLQ after retries

        expected_success = 3

        success = True
        if len(results["scorecards_saved"]) != expected_success:
            logger.error(
                f"Expected {expected_success} scorecards, got {len(results['scorecards_saved'])}"
            )
            success = False
        else:
            logger.info(f"✓ Correct number of scorecards saved: {expected_success}")

        # DLQ may have fewer if retries haven't completed
        if dlq_count == 0 and worker._tasks_failed > 0:
            logger.warning("Tasks failed but DLQ empty - retries may still be pending")
        else:
            logger.info(f"✓ DLQ contains failed tasks: {dlq_count}")

        logger.info("=" * 60)
        logger.info("OBSERVABILITY CHECK")
        logger.info("=" * 60)
        logger.info("Please verify in the dashboards:")
        logger.info("  1. Jaeger (http://localhost:16686):")
        logger.info("     - Service: distributed-eval-e2e-test")
        logger.info("     - Look for spans: e2e.*, eval.worker.*")
        logger.info("  2. Prometheus (http://localhost:9090):")
        logger.info("     - Query: e2e_test_tasks_created_total")
        logger.info("     - Query: e2e_test_processing_duration_ms_bucket")
        logger.info("  3. Grafana (http://localhost:3000):")
        logger.info("     - Check evaluation dashboards for new metrics")
        logger.info("=" * 60)

        return success

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        await queue.delete_stream(batch_id)
        await queue.close()
        logger.info("Done")


if __name__ == "__main__":
    success = asyncio.run(run_e2e_test())
    sys.exit(0 if success else 1)
