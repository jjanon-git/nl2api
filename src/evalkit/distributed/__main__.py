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
from typing import Any

from src.evalkit.distributed.config import (
    EvalMode,
    QueueBackend,
    QueueConfig,
    WorkerConfig,
)
from src.evalkit.distributed.queue import create_queue
from src.evalkit.distributed.worker import EvalWorker

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
        default=os.environ.get("EVAL_MODE", EvalMode.RESOLVER.value),
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


class WorkerDependencies:
    """Holds worker dependencies and handles cleanup."""

    def __init__(self):
        self.test_case_repo = None
        self.scorecard_repo = None
        self.batch_repo = None
        self.entity_resolver = None
        self.orchestrator = None
        self._cleanup_funcs = []

    def add_cleanup(self, func):
        """Add a cleanup function to be called on close."""
        self._cleanup_funcs.append(func)

    async def close(self):
        """Clean up all resources."""
        for func in reversed(self._cleanup_funcs):
            try:
                result = func()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")


async def create_worker_dependencies(args: argparse.Namespace) -> tuple[Any, ...]:
    """
    Create worker dependencies based on evaluation mode.

    Returns tuple of (response_generator, evaluator, scorecard_saver, test_case_fetcher, deps)
    """
    from src.evalkit.batch.response_generators import (
        create_entity_resolver_generator,
        simulate_correct_response,
    )
    from src.evalkit.common.storage import StorageConfig, close_repositories, create_repositories
    from src.evalkit.contracts.evaluation import EvalContext, EvaluationConfig
    from src.nl2api.evaluation import NL2APIPack

    eval_mode = EvalMode(args.mode)
    deps = WorkerDependencies()

    # Create repositories
    config = StorageConfig()
    test_case_repo, scorecard_repo, batch_repo = await create_repositories(config)
    deps.test_case_repo = test_case_repo
    deps.scorecard_repo = scorecard_repo
    deps.batch_repo = batch_repo
    deps.add_cleanup(close_repositories)

    # Create test case fetcher
    async def test_case_fetcher(test_case_id: str):
        """Fetch test case by ID from database."""
        test_case = await test_case_repo.get(test_case_id)
        if test_case is None:
            logger.warning(f"Test case not found: {test_case_id}")
        return test_case

    # Create scorecard saver
    async def scorecard_saver(scorecard):
        """Save scorecard to database."""
        # Add batch_id to scorecard
        scorecard_with_batch = scorecard.model_copy(update={"batch_id": args.batch_id})
        await scorecard_repo.save(scorecard_with_batch)
        logger.debug(f"Saved scorecard for test case: {scorecard.test_case_id}")

    # Create evaluator (same for all modes)
    eval_config = EvaluationConfig()
    pack = NL2APIPack(
        execution_enabled=eval_config.execution_stage_enabled,
        semantics_enabled=eval_config.semantics_stage_enabled,
        numeric_tolerance=eval_config.numeric_tolerance,
        temporal_mode=eval_config.temporal_mode,
        evaluation_date=eval_config.evaluation_date,
        relative_date_fields=eval_config.relative_date_fields,
        fiscal_year_end_month=eval_config.fiscal_year_end_month,
    )

    async def evaluator(test_case, response):
        """Evaluate the response against test case."""
        # Convert SystemResponse to system_output dict expected by pack
        system_output = {
            "raw_output": response.raw_output,
            "nl_response": response.nl_response,
        }

        return await pack.evaluate(
            test_case=test_case,
            system_output=system_output,
            context=EvalContext(worker_id=args.worker_id),
        )

    # Create response generator based on mode
    response_generator = None

    if eval_mode == EvalMode.SIMULATED:
        # Simulated mode: always return correct response
        async def simulated_response_generator(task, test_case):
            return await simulate_correct_response(test_case)

        response_generator = simulated_response_generator
        logger.info("Using simulated response generator")

    elif eval_mode == EvalMode.RESOLVER:
        # Resolver mode: use entity resolution
        from src.evalkit.common.storage.postgres.client import get_pool
        from src.nl2api.resolution.resolver import ExternalEntityResolver

        try:
            db_pool = await get_pool()
            resolver = ExternalEntityResolver(db_pool=db_pool, _internal=True)
            logger.info("Using EntityResolver with database")
        except RuntimeError:
            resolver = ExternalEntityResolver(_internal=True)
            logger.info("Using EntityResolver without database (static mappings)")

        deps.entity_resolver = resolver
        base_generator = create_entity_resolver_generator(resolver)

        async def resolver_response_generator(task, test_case):
            return await base_generator(test_case)

        response_generator = resolver_response_generator

    elif eval_mode == EvalMode.ORCHESTRATOR:
        # Full orchestrator mode
        from src.evalkit.batch.response_generators import create_nl2api_generator
        from src.nl2api.agents import AGENT_REGISTRY
        from src.nl2api.config import NL2APIConfig
        from src.nl2api.llm.factory import create_llm_provider
        from src.nl2api.orchestrator import NL2APIOrchestrator

        cfg = NL2APIConfig()
        llm = create_llm_provider(
            provider=cfg.llm_provider,
            api_key=cfg.get_llm_api_key(),
            model=cfg.llm_model,
        )
        agents = {name: cls(llm=llm) for name, cls in AGENT_REGISTRY.items()}
        orchestrator = NL2APIOrchestrator(llm=llm, agents=agents)
        deps.orchestrator = orchestrator

        base_generator = create_nl2api_generator(orchestrator)

        async def orchestrator_response_generator(task, test_case):
            return await base_generator(test_case)

        response_generator = orchestrator_response_generator
        logger.info(f"Using full orchestrator ({cfg.llm_provider}/{cfg.llm_model})")

    elif eval_mode == EvalMode.ROUTING:
        # Routing mode - LLM router only
        from src.evalkit.batch.response_generators import create_routing_generator
        from src.nl2api.config import NL2APIConfig
        from src.nl2api.llm.factory import create_llm_provider
        from src.nl2api.routing.llm_router import LLMToolRouter
        from src.nl2api.routing.protocols import RoutingToolDefinition, ToolProvider

        class StubToolProvider(ToolProvider):
            def __init__(self, name: str, description: str, capabilities: tuple[str, ...]):
                self._name = name
                self._description = description
                self._capabilities = capabilities

            @property
            def provider_name(self) -> str:
                return self._name

            @property
            def provider_description(self) -> str:
                return self._description

            @property
            def capabilities(self) -> tuple[str, ...]:
                return self._capabilities

            def get_routing_tools(self) -> list[RoutingToolDefinition]:
                return [
                    RoutingToolDefinition(
                        name=f"route_to_{self._name}",
                        description=self._description,
                        domain=self._name,
                    )
                ]

        tool_providers = [
            StubToolProvider("datastream", "Stock prices, market data", ("prices",)),
            StubToolProvider("estimates", "Analyst forecasts, EPS estimates", ("eps_forecast",)),
            StubToolProvider("fundamentals", "Financial statements", ("revenue",)),
            StubToolProvider("officers", "Executives, board members", ("ceo",)),
            StubToolProvider("screening", "Stock screening, rankings", ("top_n",)),
        ]

        cfg = NL2APIConfig()
        llm = create_llm_provider(
            provider=cfg.llm_provider,
            api_key=cfg.get_llm_api_key(),
            model=cfg.routing_model,
        )
        router = LLMToolRouter(llm=llm, tool_providers=tool_providers)

        base_generator = create_routing_generator(router)

        async def routing_response_generator(task, test_case):
            return await base_generator(test_case)

        response_generator = routing_response_generator
        logger.info(f"Using LLM router ({cfg.llm_provider}/{cfg.routing_model})")

    elif eval_mode == EvalMode.TOOL_ONLY:
        # Tool-only mode requires agent name from environment
        agent_name = os.environ.get("EVAL_AGENT_NAME", "datastream")

        from src.evalkit.batch.response_generators import create_tool_only_generator
        from src.evalkit.common.storage.postgres.client import get_pool
        from src.nl2api.agents import get_agent_by_name
        from src.nl2api.config import NL2APIConfig
        from src.nl2api.llm.factory import create_llm_provider
        from src.nl2api.resolution.resolver import ExternalEntityResolver

        cfg = NL2APIConfig()
        llm = create_llm_provider(
            provider=cfg.llm_provider,
            api_key=cfg.get_llm_api_key(),
            model=cfg.llm_model,
        )
        agent = get_agent_by_name(agent_name, llm=llm)

        try:
            db_pool = await get_pool()
            resolver = ExternalEntityResolver(db_pool=db_pool, _internal=True)
        except RuntimeError:
            resolver = ExternalEntityResolver(_internal=True)

        deps.entity_resolver = resolver

        base_generator = create_tool_only_generator(agent, entity_resolver=resolver)

        async def tool_only_response_generator(task, test_case):
            return await base_generator(test_case)

        response_generator = tool_only_response_generator
        logger.info(f"Using {agent_name} agent ({cfg.llm_provider}/{cfg.llm_model})")

    else:
        raise ValueError(f"Unsupported eval mode: {eval_mode}")

    return (
        response_generator,
        evaluator,
        scorecard_saver,
        test_case_fetcher,
        deps,
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

    queue = None
    deps = None

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
            deps,
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
        if deps:
            await deps.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
