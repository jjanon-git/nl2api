"""
Distributed Evaluation Worker Entry Point.

Run a worker process:
    python -m src.evaluation.distributed --worker-id worker-0 --batch-id batch-001

Or with environment variables:
    EVAL_WORKER_ID=worker-0 EVAL_BATCH_ID=batch-001 python -m src.evaluation.distributed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from src.evaluation.distributed.config import (
    QueueConfig,
    WorkerConfig,
    QueueBackend,
    EvalMode,
)
from src.evaluation.distributed.queue import create_queue
from src.evaluation.distributed.worker import EvalWorker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a distributed evaluation worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--worker-id",
        type=str,
        default=os.environ.get("EVAL_WORKER_ID", "worker-0"),
        help="Unique identifier for this worker",
    )

    parser.add_argument(
        "--batch-id",
        type=str,
        default=os.environ.get("EVAL_BATCH_ID"),
        help="Batch ID to process (required)",
    )

    parser.add_argument(
        "--redis-url",
        type=str,
        default=os.environ.get("EVAL_REDIS_URL", "redis://localhost:6379"),
        help="Redis URL for queue backend",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in EvalMode],
        default=os.environ.get("EVAL_MODE", EvalMode.ORCHESTRATOR.value),
        help="Evaluation mode",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("EVAL_MAX_RETRIES", "3")),
        help="Maximum retry attempts per task",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def create_worker_dependencies(args: argparse.Namespace):
    """
    Create worker dependencies based on evaluation mode.

    Returns tuple of (response_generator, evaluator, scorecard_saver, test_case_fetcher)
    """
    # Import here to avoid circular imports and allow mode-specific imports
    from src.evaluation.batch.response_generators import simulate_correct_response

    eval_mode = EvalMode(args.mode)

    # Default implementations (simulated mode)
    async def default_response_generator(task, test_case):
        """Generate a simulated response (always correct)."""
        return simulate_correct_response(test_case)

    async def default_evaluator(test_case, response):
        """Evaluate the response against test case."""
        # Import evaluator
        from src.evaluation.core.evaluators import create_evaluator_pipeline
        from src.evaluation.core.config import EvaluationConfig

        config = EvaluationConfig()
        pipeline = create_evaluator_pipeline(config)
        return await pipeline.evaluate(test_case, response)

    async def default_scorecard_saver(scorecard):
        """Save scorecard (no-op for now, will use repo in production)."""
        logger.debug(f"Scorecard saved: {scorecard.test_case_id}")

    async def default_test_case_fetcher(test_case_id: str):
        """Fetch test case by ID (placeholder)."""
        # In production, this would use TestCaseRepository
        logger.warning(f"Test case fetcher not configured, returning None for {test_case_id}")
        return None

    # Mode-specific implementations would go here
    if eval_mode == EvalMode.SIMULATED:
        pass  # Use defaults
    elif eval_mode == EvalMode.RESOLVER:
        # TODO: Configure entity resolver mode
        pass
    elif eval_mode == EvalMode.ORCHESTRATOR:
        # TODO: Configure full orchestrator mode
        pass

    return (
        default_response_generator,
        default_evaluator,
        default_scorecard_saver,
        default_test_case_fetcher,
    )


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.batch_id:
        logger.error("--batch-id is required")
        return 1

    logger.info(f"Starting worker {args.worker_id} for batch {args.batch_id}")
    logger.info(f"Mode: {args.mode}, Redis: {args.redis_url}")

    # Create queue
    queue_config = QueueConfig(
        backend=QueueBackend.REDIS,
        redis_url=args.redis_url,
        max_retries=args.max_retries,
    )

    try:
        queue = await create_queue(queue_config)
    except Exception as e:
        logger.error(f"Failed to connect to queue: {e}")
        return 1

    # Create worker dependencies
    try:
        (
            response_generator,
            evaluator,
            scorecard_saver,
            test_case_fetcher,
        ) = await create_worker_dependencies(args)
    except Exception as e:
        logger.error(f"Failed to create worker dependencies: {e}")
        await queue.close()
        return 1

    # Create and run worker
    worker_config = WorkerConfig(
        worker_id=args.worker_id,
        eval_mode=EvalMode(args.mode),
    )

    worker = EvalWorker(
        worker_id=args.worker_id,
        queue=queue,
        batch_id=args.batch_id,
        response_generator=response_generator,
        evaluator=evaluator,
        scorecard_saver=scorecard_saver,
        test_case_fetcher=test_case_fetcher,
        config=worker_config,
    )

    try:
        await worker.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
        await worker.shutdown()
        return 0
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        return 1
    finally:
        await queue.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
