"""
Batch command - Run batch evaluations on test cases from storage.

Usage:
    # Run all test cases
    python -m src.evaluation.cli.main batch run

    # Run with filters
    python -m src.evaluation.cli.main batch run --tag search --limit 20

    # Check batch status
    python -m src.evaluation.cli.main batch status <batch-id>

    # View results
    python -m src.evaluation.cli.main batch results <batch-id>
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from CONTRACTS import TaskStatus
from src.evaluation.cli.commands.api_batch import api_app

logger = logging.getLogger(__name__)

console = Console()

# Create the batch app (subcommand group)
batch_app = typer.Typer(
    name="batch",
    help="Batch evaluation commands",
    no_args_is_help=True,
)

# Add the Anthropic API subcommand group
batch_app.add_typer(api_app, name="api", help="Anthropic Message Batches API (50% cheaper)")


@batch_app.command("run")
def batch_run(
    pack: Annotated[
        str,
        typer.Option("--pack", "-p", help="Evaluation pack (nl2api, rag) - REQUIRED"),
    ],
    label: Annotated[
        str,
        typer.Option(
            "--label",
            "-l",
            help="Label for this run (e.g., 'baseline', 'new-embedder-v2') - REQUIRED",
        ),
    ],
    description: Annotated[
        str | None,
        typer.Option(
            "--description", help="Optional longer description of the change being tested"
        ),
    ] = None,
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Filter by tags (can be repeated)"),
    ] = None,
    complexity_min: Annotated[
        int | None,
        typer.Option("--min-complexity", help="Minimum complexity level (1-5)"),
    ] = None,
    complexity_max: Annotated[
        int | None,
        typer.Option("--max-complexity", help="Maximum complexity level (1-5)"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Maximum test cases to run"),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Concurrent evaluations"),
    ] = 10,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="Response mode: resolver, orchestrator, routing, tool_only, or simulated",
        ),
    ] = "resolver",
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent", "-a", help="Agent name for tool_only mode (datastream, estimates, etc.)"
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override LLM model (e.g., claude-3-5-haiku-20241022)"),
    ] = None,
    client: Annotated[
        str,
        typer.Option(
            "--client", help="Client type (internal, mcp_claude, mcp_chatgpt, mcp_custom)"
        ),
    ] = "internal",
    client_version: Annotated[
        str | None,
        typer.Option("--client-version", help="Client version identifier"),
    ] = None,
    semantics: Annotated[
        bool,
        typer.Option("--semantics", help="Enable LLM-as-Judge semantic evaluation (Stage 4)"),
    ] = False,
    semantics_model: Annotated[
        str | None,
        typer.Option("--semantics-model", help="Override model for semantic evaluation"),
    ] = None,
    semantics_threshold: Annotated[
        float,
        typer.Option(
            "--semantics-threshold", help="Minimum score to pass semantic evaluation (0.0-1.0)"
        ),
    ] = 0.7,
    eval_date: Annotated[
        str | None,
        typer.Option("--eval-date", help="Reference date for temporal normalization (YYYY-MM-DD)"),
    ] = None,
    temporal_mode: Annotated[
        str,
        typer.Option(
            "--temporal-mode", help="Temporal validation mode: behavioral, structural, or data"
        ),
    ] = "structural",
    distributed: Annotated[
        bool,
        typer.Option("--distributed", "-d", help="Use distributed workers via Redis queue"),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of worker subprocesses (distributed mode)"),
    ] = 4,
    redis_url: Annotated[
        str,
        typer.Option("--redis-url", help="Redis URL for distributed queue"),
    ] = "redis://localhost:6379",
) -> None:
    """
    Run batch evaluation on test cases from database.

    Fetches test cases matching the filters, runs evaluations concurrently,
    and saves scorecards with a batch ID for tracking.

    PACKS:
    - nl2api: Tool-calling LLM evaluation (syntax, logic, execution, semantics).
    - rag: RAG system evaluation (retrieval, faithfulness, relevance, citations).

    MODES:
    - resolver (default): Uses real EntityResolver for accuracy measurement.
      Results are persisted and tracked over time.
    - orchestrator: Uses full NL2API orchestrator (requires LLM API key).
      End-to-end accuracy measurement.
    - routing: Uses LLM router for routing evaluation (requires LLM API key).
      Tests query → domain routing accuracy.
    - tool_only: Tests a single agent in isolation (requires --agent and LLM API key).
      Useful for comparing agent performance with different LLMs.
    - simulated: Always returns correct answers (for pipeline testing only).
      Should NOT be used for accuracy tracking - use unit tests instead.

    SEMANTICS:
    - Use --semantics to enable LLM-as-Judge semantic evaluation (Stage 4).
    - Requires expected_nl_response to be populated in test cases.
    - Use --semantics-model to override the judge model (default: claude-3-5-haiku).
    - Use --semantics-threshold to set the minimum pass score (default: 0.7).

    TEMPORAL:
    - Use --eval-date to set the reference date for temporal normalization.
    - Use --temporal-mode to control date comparison:
      - structural (default): Normalizes dates before comparison (-1D == 2026-01-20)
      - behavioral: Only validates both have valid temporal expressions
      - data: Exact match required (for point-in-time validation)

    DISTRIBUTED:
    - Use --distributed to run with multiple worker subprocesses via Redis queue.
    - Use --workers to set number of worker processes (default: 4).
    - Use --redis-url to specify Redis connection (default: redis://localhost:6379).
    - Workers are auto-spawned and terminated on completion or Ctrl+C.
    - Requires Redis to be running (docker compose up -d).
    """
    from datetime import date as date_type

    # Validate pack selection
    valid_packs = ("nl2api", "rag")
    if pack not in valid_packs:
        console.print(f"[red]Error:[/red] Invalid pack '{pack}'.")
        console.print(f"Available packs: {', '.join(valid_packs)}")
        raise typer.Exit(1)

    if mode not in ("resolver", "orchestrator", "simulated", "routing", "tool_only"):
        console.print(f"[red]Error:[/red] Invalid mode '{mode}'.")
        console.print("Use 'resolver', 'orchestrator', 'routing', 'tool_only', or 'simulated'.")
        raise typer.Exit(1)

    if mode == "tool_only" and agent is None:
        console.print("[red]Error:[/red] --agent is required for tool_only mode.")
        console.print("Available agents: datastream, estimates, fundamentals, officers, screening")
        raise typer.Exit(1)

    if temporal_mode not in ("behavioral", "structural", "data"):
        console.print(f"[red]Error:[/red] Invalid temporal mode '{temporal_mode}'.")
        console.print("Use 'behavioral', 'structural', or 'data'.")
        raise typer.Exit(1)

    # Parse eval_date if provided
    parsed_eval_date: date_type | None = None
    if eval_date:
        try:
            parsed_eval_date = date_type.fromisoformat(eval_date)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid date format '{eval_date}'.")
            console.print("Use YYYY-MM-DD format (e.g., 2026-01-21).")
            raise typer.Exit(1)

    asyncio.run(
        _batch_run_async(
            pack=pack,
            run_label=label,
            run_description=description,
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            limit=limit,
            concurrency=concurrency,
            mode=mode,
            agent_name=agent,
            verbose=verbose,
            model=model,
            client=client,
            client_version=client_version,
            semantics_enabled=semantics,
            semantics_model=semantics_model,
            semantics_threshold=semantics_threshold,
            evaluation_date=parsed_eval_date,
            temporal_mode=temporal_mode,
            distributed=distributed,
            num_workers=workers,
            redis_url=redis_url,
        )
    )


async def _batch_run_async(
    pack: str,
    run_label: str,
    run_description: str | None,
    tags: list[str] | None,
    complexity_min: int | None,
    complexity_max: int | None,
    limit: int | None,
    concurrency: int,
    mode: str,
    agent_name: str | None = None,
    verbose: bool = False,
    model: str | None = None,
    client: str = "internal",
    client_version: str | None = None,
    semantics_enabled: bool = False,
    semantics_model: str | None = None,
    semantics_threshold: float = 0.7,
    evaluation_date: date | None = None,
    temporal_mode: str = "structural",
    distributed: bool = False,
    num_workers: int = 4,
    redis_url: str = "redis://localhost:6379",
) -> None:
    """Async implementation of batch run command."""
    from src.common.git_info import get_git_info
    from src.common.storage import (
        StorageConfig,
        close_repositories,
        create_repositories,
    )
    from src.evaluation.batch import BatchRunner, BatchRunnerConfig
    from src.evaluation.batch.response_generators import (
        create_entity_resolver_generator,
        create_nl2api_generator,
        create_routing_generator,
        create_tool_only_generator,
        simulate_correct_response,
    )

    # Capture git info from current working directory
    git_info = get_git_info()

    try:
        config = StorageConfig()
        test_case_repo, scorecard_repo, batch_repo = await create_repositories(config)

        # Select response generator based on mode
        response_generator = None
        if mode == "simulated":
            console.print("[yellow]WARNING: Simulated responses (pipeline test only).[/yellow]")
            console.print("[yellow]Results should NOT be used for tracking.[/yellow]\n")
            response_generator = simulate_correct_response
        elif mode == "resolver":
            # Use real EntityResolver for accuracy measurement
            from src.common.storage.postgres.client import get_pool
            from src.nl2api.resolution.resolver import ExternalEntityResolver

            # Get database pool for entity lookups (2.9M entities, 3.7M aliases)
            try:
                db_pool = await get_pool()
                resolver = ExternalEntityResolver(db_pool=db_pool)
                console.print(
                    "[green]Using EntityResolver with database (2.9M entities).[/green]\n"
                )
            except RuntimeError:
                # Pool not initialized (e.g., using memory backend)
                resolver = ExternalEntityResolver()
                console.print(
                    "[yellow]Using EntityResolver without database (static mappings only).[/yellow]\n"
                )

            response_generator = create_entity_resolver_generator(resolver)
        elif mode == "orchestrator":
            # Use full NL2API orchestrator
            from src.nl2api.agents import AGENT_REGISTRY
            from src.nl2api.config import NL2APIConfig
            from src.nl2api.llm.factory import create_llm_provider
            from src.nl2api.orchestrator import NL2APIOrchestrator

            cfg = NL2APIConfig()
            llm_model = model if model else cfg.llm_model
            llm = create_llm_provider(
                provider=cfg.llm_provider,
                api_key=cfg.get_llm_api_key(),
                model=llm_model,
            )

            # Create all domain agents
            agents = {name: cls(llm=llm) for name, cls in AGENT_REGISTRY.items()}

            orchestrator = NL2APIOrchestrator(llm=llm, agents=agents)
            response_generator = create_nl2api_generator(orchestrator)
            console.print(
                f"[green]Using full NL2API orchestrator ({cfg.llm_provider}/{llm_model}).[/green]\n"
            )

            # Override client_version with model for cost tracking
            if client_version is None:
                client_version = llm_model
        elif mode == "routing":
            # Use LLM router for routing evaluation
            from src.nl2api.config import NL2APIConfig
            from src.nl2api.llm.factory import create_llm_provider
            from src.nl2api.routing.llm_router import LLMToolRouter
            from src.nl2api.routing.protocols import RoutingToolDefinition, ToolProvider

            # Create stub tool providers for the 5 domains
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
                StubToolProvider(
                    "datastream",
                    "Stock prices, market data, trading volume, historical prices",
                    ("prices", "volume", "market_cap"),
                ),
                StubToolProvider(
                    "estimates",
                    "Analyst forecasts, EPS estimates, revenue projections, price targets",
                    ("eps_forecast", "recommendations", "price_targets"),
                ),
                StubToolProvider(
                    "fundamentals",
                    "Financial statements, balance sheet, income statement, reported data",
                    ("revenue", "net_income", "balance_sheet"),
                ),
                StubToolProvider(
                    "officers",
                    "Executives, board members, CEO, CFO, compensation",
                    ("ceo", "board", "compensation"),
                ),
                StubToolProvider(
                    "screening",
                    "Stock screening, rankings, top N queries, filtering",
                    ("top_n", "ranking", "screening"),
                ),
            ]

            cfg = NL2APIConfig()
            # Use --model override if provided, otherwise use routing_model from config
            llm_model = model if model else cfg.routing_model
            llm = create_llm_provider(
                provider=cfg.llm_provider,
                api_key=cfg.get_llm_api_key(),
                model=llm_model,
            )
            router = LLMToolRouter(llm=llm, tool_providers=tool_providers)
            response_generator = create_routing_generator(router)
            console.print(f"[green]Using LLM router ({cfg.llm_provider}/{llm_model}).[/green]\n")
        elif mode == "tool_only":
            # Use single agent for isolated tool generation testing
            # Includes entity resolution so agent gets proper RICs
            from src.common.storage.postgres.client import get_pool
            from src.nl2api.agents import get_agent_by_name
            from src.nl2api.config import NL2APIConfig
            from src.nl2api.llm.factory import create_llm_provider
            from src.nl2api.resolution.resolver import ExternalEntityResolver

            cfg = NL2APIConfig()
            llm_model = model if model else cfg.llm_model
            llm = create_llm_provider(
                provider=cfg.llm_provider,
                api_key=cfg.get_llm_api_key(),
                model=llm_model,
            )
            agent = get_agent_by_name(agent_name, llm=llm)

            # Create entity resolver for live resolution
            try:
                db_pool = await get_pool()
                resolver = ExternalEntityResolver(db_pool=db_pool)
                console.print("[dim]Entity resolver: database (2.9M entities)[/dim]")
            except RuntimeError:
                resolver = ExternalEntityResolver()
                console.print("[dim]Entity resolver: static mappings only[/dim]")

            response_generator = create_tool_only_generator(agent, entity_resolver=resolver)
            console.print(
                f"[green]Using {agent_name} agent ({cfg.llm_provider}/{llm_model}).[/green]\n"
            )

            # Override client_version with model for cost tracking
            if client_version is None:
                client_version = llm_model

        # Map mode to eval_mode
        eval_mode_map = {
            "resolver": "resolver",
            "orchestrator": "orchestrator",
            "routing": "routing",
            "tool_only": "tool_only",
            "simulated": "orchestrator",  # Simulated still uses orchestrator eval mode
        }

        # Handle distributed mode
        if distributed:
            await _run_distributed_batch(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                tags=tags,
                complexity_min=complexity_min,
                complexity_max=complexity_max,
                limit=limit,
                mode=mode,
                num_workers=num_workers,
                redis_url=redis_url,
                verbose=verbose,
            )
            return

        runner_config = BatchRunnerConfig(
            pack_name=pack,
            max_concurrency=concurrency,
            show_progress=True,
            verbose=verbose,
            client_type=client,
            client_version=client_version,
            eval_mode=eval_mode_map.get(mode, "orchestrator"),
            semantics_enabled=semantics_enabled,
            semantics_model=semantics_model,
            semantics_pass_threshold=semantics_threshold,
            evaluation_date=evaluation_date,
            temporal_mode=temporal_mode,
            run_label=run_label,
            run_description=run_description,
            git_commit=git_info.commit,
            git_branch=git_info.branch,
        )

        if semantics_enabled:
            console.print("[cyan]Semantics stage enabled[/cyan]")
            if semantics_model:
                console.print(f"  Model: {semantics_model}")
            else:
                console.print("  Model: claude-3-5-haiku-20241022 (default)")
            console.print(f"  Pass threshold: {semantics_threshold:.2f}\n")

        if evaluation_date or temporal_mode != "structural":
            console.print("[cyan]Temporal evaluation[/cyan]")
            if evaluation_date:
                console.print(f"  Reference date: {evaluation_date}")
            else:
                from datetime import date as date_class

                console.print(f"  Reference date: {date_class.today()} (today)")
            console.print(f"  Validation mode: {temporal_mode}\n")

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=runner_config,
        )

        batch_job = await runner.run(
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            limit=limit,
            response_simulator=response_generator,
        )

        # Handle no test cases found
        if batch_job is None:
            console.print("[yellow]No test cases found matching filters[/yellow]")
            raise typer.Exit(0)

        # Exit with appropriate code
        if batch_job.failed_count > 0:
            raise typer.Exit(1)
        else:
            raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(2)
    finally:
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


async def _run_distributed_batch(
    test_case_repo,
    scorecard_repo,
    batch_repo,
    tags: list[str] | None,
    complexity_min: int | None,
    complexity_max: int | None,
    limit: int | None,
    mode: str,
    num_workers: int,
    redis_url: str,
    verbose: bool,
) -> None:
    """
    Run batch evaluation using distributed workers.

    Spawns worker subprocesses that consume tasks from a Redis queue,
    monitor progress, and handle completion.
    """
    from datetime import UTC, datetime

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    from src.contracts.core import TaskStatus
    from src.contracts.worker import BatchJob
    from src.evaluation.distributed.config import CoordinatorConfig, QueueBackend, QueueConfig
    from src.evaluation.distributed.coordinator import BatchCoordinator
    from src.evaluation.distributed.manager import LocalWorkerManager
    from src.evaluation.distributed.queue import create_queue

    console.print("[cyan]Distributed mode enabled[/cyan]")
    console.print(f"  Workers: {num_workers}")
    console.print(f"  Redis: {redis_url}")
    console.print(f"  Mode: {mode}\n")

    # Fetch test cases
    test_cases = await test_case_repo.list(
        tags=tags,
        complexity_min=complexity_min,
        complexity_max=complexity_max,
        limit=limit or 10000,
    )

    if not test_cases:
        console.print("[yellow]No test cases found matching filters[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found [green]{len(test_cases)}[/green] test cases")

    # Create queue
    queue_config = QueueConfig(
        backend=QueueBackend.REDIS,
        redis_url=redis_url,
    )

    manager = None
    queue = None

    try:
        queue = await create_queue(queue_config)

        # Create batch job
        batch_job = BatchJob(
            total_tests=len(test_cases),
            status=TaskStatus.IN_PROGRESS,
            started_at=datetime.now(UTC),
            tags=tuple(tags) if tags else (),
        )
        await batch_repo.create(batch_job)
        batch_id = batch_job.batch_id

        console.print(f"Created batch: [cyan]{batch_id}[/cyan]\n")

        # Create coordinator
        coordinator_config = CoordinatorConfig(
            progress_poll_interval_seconds=2,
            batch_timeout_seconds=3600,  # 1 hour default
        )
        coordinator = BatchCoordinator(
            queue=queue,
            batch_repo=batch_repo,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            config=coordinator_config,
        )

        # Enqueue tasks
        console.print("Enqueuing tasks...")
        enqueued = await coordinator.start_batch(test_cases, batch_id, eval_mode=mode)
        console.print(f"Enqueued [green]{enqueued}[/green] tasks\n")

        # Start workers
        manager = LocalWorkerManager(
            worker_count=num_workers,
            redis_url=redis_url,
            eval_mode=mode,
            verbose=verbose,
        )
        manager.start(batch_id)
        console.print(f"Started [green]{num_workers}[/green] workers\n")

        # Wait for completion with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=len(test_cases))

            def on_progress(completed: int, total: int):
                progress.update(task, completed=completed)

            result = await coordinator.wait_for_completion(
                batch_id=batch_id,
                total_tasks=len(test_cases),
                on_progress=on_progress,
            )

        # Display results
        console.print()
        console.print(f"[bold]Batch Complete:[/bold] [cyan]{batch_id}[/cyan]")
        console.print(f"  Total: {result.total}")
        console.print(f"  Passed: [green]{result.passed}[/green]")
        console.print(f"  Failed: [red]{result.failed}[/red]")
        if result.in_dlq > 0:
            console.print(f"  In DLQ: [yellow]{result.in_dlq}[/yellow]")
        console.print(f"  Duration: {result.duration_seconds:.1f}s")

        # Update batch job
        completed_batch = BatchJob(
            batch_id=batch_id,
            total_tests=len(test_cases),
            completed_count=result.passed,
            failed_count=result.failed,
            status=TaskStatus.COMPLETED if result.completed else TaskStatus.FAILED,
            created_at=batch_job.created_at,
            started_at=batch_job.started_at,
            completed_at=datetime.now(UTC),
            tags=batch_job.tags,
        )
        await batch_repo.update(completed_batch)

        # Clean up queue resources
        await coordinator.cleanup(batch_id)

        # Exit code based on results
        if result.failed > 0 or not result.completed:
            raise typer.Exit(1)
        else:
            raise typer.Exit(0)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted - stopping workers...[/yellow]")
        raise typer.Exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(2)
    finally:
        # Always stop workers
        if manager:
            manager.stop(timeout=30)
        # Close queue
        if queue:
            await queue.close()


@batch_app.command("status")
def batch_status(
    batch_id: Annotated[str, typer.Argument(help="Batch ID to check")],
) -> None:
    """
    Show status of a batch run.

    Displays progress, pass/fail counts, and timing information.
    """
    asyncio.run(_batch_status_async(batch_id))


async def _batch_status_async(batch_id: str) -> None:
    """Async implementation of batch status command."""
    from src.common.storage import StorageConfig, close_repositories, create_repositories

    try:
        config = StorageConfig()
        _, scorecard_repo, batch_repo = await create_repositories(config)

        batch_job = await batch_repo.get(batch_id)
        if not batch_job:
            console.print(f"[red]Error:[/red] Batch not found: {batch_id}")
            raise typer.Exit(1)

        # Get summary from scorecards
        summary = await scorecard_repo.get_batch_summary(batch_id)

        # Display status
        console.print()
        table = Table(title=f"Batch Status: [cyan]{batch_id}[/cyan]")
        table.add_column("Field", style="bold")
        table.add_column("Value", justify="right")

        # Status with color
        status_display = batch_job.status.value
        if batch_job.status == TaskStatus.COMPLETED:
            status_display = f"[green]{status_display}[/green]"
        elif batch_job.status == TaskStatus.IN_PROGRESS:
            status_display = f"[yellow]{status_display}[/yellow]"
        elif batch_job.status == TaskStatus.FAILED:
            status_display = f"[red]{status_display}[/red]"

        table.add_row("Status", status_display)
        table.add_row("Run Label", f"[magenta]{batch_job.run_label}[/magenta]")
        if batch_job.run_description:
            table.add_row("Description", batch_job.run_description)
        if batch_job.git_commit:
            git_info = batch_job.git_commit
            if batch_job.git_branch:
                git_info += f" ({batch_job.git_branch})"
            table.add_row("Git", git_info)
        table.add_row("Total Tests", str(batch_job.total_tests))
        table.add_row("Completed", f"[green]{summary['passed']}[/green]")
        table.add_row("Failed", f"[red]{summary['failed']}[/red]" if summary["failed"] > 0 else "0")
        table.add_row("Progress", f"{batch_job.progress_pct:.1f}%")
        table.add_row("Avg Score", f"{summary['avg_score']:.2f}")

        if batch_job.created_at:
            table.add_row("Created", batch_job.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        if batch_job.started_at:
            table.add_row("Started", batch_job.started_at.strftime("%Y-%m-%d %H:%M:%S"))
        if batch_job.completed_at:
            table.add_row("Completed", batch_job.completed_at.strftime("%Y-%m-%d %H:%M:%S"))
            if batch_job.started_at:
                duration = (batch_job.completed_at - batch_job.started_at).total_seconds()
                table.add_row("Duration", f"{duration:.1f}s")

        console.print(table)
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


@batch_app.command("results")
def batch_results(
    batch_id: Annotated[str, typer.Argument(help="Batch ID to view results for")],
    failed_only: Annotated[
        bool,
        typer.Option("--failed", "-f", help="Show only failed tests"),
    ] = False,
    export: Annotated[
        Path | None,
        typer.Option("--export", "-e", help="Export results to JSON file"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum results to show"),
    ] = 50,
) -> None:
    """
    Show or export results from a batch run.

    Lists scorecards with their pass/fail status and scores.
    Use --export to save results to a JSON file.
    """
    asyncio.run(_batch_results_async(batch_id, failed_only, export, limit))


async def _batch_results_async(
    batch_id: str,
    failed_only: bool,
    export: Path | None,
    limit: int,
) -> None:
    """Async implementation of batch results command."""
    from src.common.storage import StorageConfig, close_repositories, create_repositories

    try:
        config = StorageConfig()
        test_case_repo, scorecard_repo, batch_repo = await create_repositories(config)

        # Get batch job
        batch_job = await batch_repo.get(batch_id)
        if not batch_job:
            console.print(f"[red]Error:[/red] Batch not found: {batch_id}")
            raise typer.Exit(1)

        # Get scorecards
        scorecards = await scorecard_repo.get_by_batch(batch_id)

        if failed_only:
            scorecards = [sc for sc in scorecards if not sc.overall_passed]

        # Export if requested
        if export:
            export_data = {
                "batch_id": batch_id,
                "total": len(scorecards),
                "results": [],
            }

            for sc in scorecards:
                # Get test case for query text
                test_case = await test_case_repo.get(sc.test_case_id)
                export_data["results"].append(
                    {
                        "test_case_id": sc.test_case_id,
                        "scorecard_id": sc.scorecard_id,
                        "nl_query": test_case.nl_query if test_case else "Unknown",
                        "passed": sc.overall_passed,
                        "score": sc.overall_score,
                        "syntax_passed": sc.syntax_result.passed,
                        "logic_passed": sc.logic_result.passed if sc.logic_result else None,
                        "logic_score": sc.logic_result.score if sc.logic_result else None,
                        "error_code": (
                            sc.logic_result.error_code.value
                            if sc.logic_result and sc.logic_result.error_code
                            else None
                        ),
                        "reason": sc.logic_result.reason if sc.logic_result else None,
                    }
                )

            export.write_text(json.dumps(export_data, indent=2))
            console.print(f"[green]Exported {len(scorecards)} results to {export}[/green]")
            raise typer.Exit(0)

        # Display results
        console.print()

        if not scorecards:
            console.print("[yellow]No results found[/yellow]")
            raise typer.Exit(0)

        # Summary
        summary = await scorecard_repo.get_batch_summary(batch_id)
        console.print(f"[bold]Batch Results:[/bold] [cyan]{batch_id}[/cyan]")
        passed = f"[green]{summary['passed']}[/green]"
        failed = f"[red]{summary['failed']}[/red]"
        console.print(f"  Total: {summary['total']} | Passed: {passed} | Failed: {failed}")
        console.print(f"  Avg Score: {summary['avg_score']:.2f}")
        console.print()

        # Results table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Test ID", width=12)
        table.add_column("Status", width=8)
        table.add_column("Score", width=8)
        table.add_column("Query", width=50)
        table.add_column("Error", width=20)

        displayed = 0
        for sc in scorecards:
            if displayed >= limit:
                break

            # Get test case for query
            test_case = await test_case_repo.get(sc.test_case_id)
            query = test_case.nl_query if test_case else "Unknown"
            if len(query) > 47:
                query = query[:47] + "..."

            status = "[green]PASS[/green]" if sc.overall_passed else "[red]FAIL[/red]"
            error = ""
            if sc.logic_result and sc.logic_result.error_code:
                error = sc.logic_result.error_code.value

            table.add_row(
                sc.test_case_id[:12] + "...",
                status,
                f"{sc.overall_score:.2f}",
                query,
                error,
            )
            displayed += 1

        console.print(table)

        if len(scorecards) > limit:
            console.print(f"\n[dim]Showing {limit}/{len(scorecards)}. Use --limit for more.[/dim]")

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


@batch_app.command("list")
def batch_list(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum batches to show"),
    ] = 10,
) -> None:
    """
    List recent batch runs.

    Shows batch IDs, status, and summary statistics.
    """
    asyncio.run(_batch_list_async(limit))


async def _batch_list_async(limit: int) -> None:
    """Async implementation of batch list command."""
    from src.common.storage import StorageConfig, close_repositories, create_repositories

    try:
        config = StorageConfig()
        _, scorecard_repo, batch_repo = await create_repositories(config)

        batches = await batch_repo.list_recent(limit)

        if not batches:
            console.print("[yellow]No batch runs found[/yellow]")
            raise typer.Exit(0)

        console.print()
        table = Table(title="Recent Batch Runs", show_header=True, header_style="bold")
        table.add_column("Batch ID", width=20)
        table.add_column("Run Label", width=20)
        table.add_column("Status", width=12)
        table.add_column("Tests", width=8, justify="right")
        table.add_column("Passed", width=8, justify="right")
        table.add_column("Failed", width=8, justify="right")
        table.add_column("Created", width=20)

        for batch in batches:
            status = batch.status.value
            if batch.status == TaskStatus.COMPLETED:
                status = f"[green]{status}[/green]"
            elif batch.status == TaskStatus.IN_PROGRESS:
                status = f"[yellow]{status}[/yellow]"
            elif batch.status == TaskStatus.FAILED:
                status = f"[red]{status}[/red]"

            passed = f"[green]{batch.completed_count}[/green]"
            failed = f"[red]{batch.failed_count}[/red]" if batch.failed_count > 0 else "0"

            # Truncate batch_id for display
            batch_id_short = batch.batch_id[:18] + "..."
            run_label = batch.run_label or "untracked"
            if len(run_label) > 18:
                run_label = run_label[:18] + "..."

            table.add_row(
                batch_id_short,
                f"[magenta]{run_label}[/magenta]",
                status,
                str(batch.total_tests),
                passed,
                failed,
                batch.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            )

        console.print(table)
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


@batch_app.command("compare")
def batch_compare(
    clients: Annotated[
        list[str],
        typer.Option("--client", "-c", help="Client types to compare (can be repeated)"),
    ],
    start: Annotated[
        str | None,
        typer.Option("--start", help="Start date (YYYY-MM-DD)"),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option("--end", help="End date (YYYY-MM-DD)"),
    ] = None,
) -> None:
    """
    Compare evaluation results across different client types.

    Shows pass rate, average score, token usage, and costs for each client.
    """
    asyncio.run(_batch_compare_async(clients, start, end))


async def _batch_compare_async(
    clients: list[str],
    start: str | None,
    end: str | None,
) -> None:
    """Async implementation of batch compare command."""
    from datetime import datetime

    from src.common.storage import StorageConfig, close_repositories, create_repositories

    try:
        config = StorageConfig()
        _, scorecard_repo, _ = await create_repositories(config)

        # Parse dates
        start_date = None
        end_date = None
        if start:
            start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        if end:
            end_date = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        # Get comparison data
        summaries = await scorecard_repo.get_comparison_summary(clients, start_date, end_date)

        if not summaries:
            console.print("[yellow]No data found for the specified clients and date range[/yellow]")
            raise typer.Exit(0)

        console.print()
        table = Table(title="Client Comparison", show_header=True, header_style="bold")
        table.add_column("Client", width=15)
        table.add_column("Version", width=25)
        table.add_column("Tests", width=8, justify="right")
        table.add_column("Pass Rate", width=10, justify="right")
        table.add_column("Avg Score", width=10, justify="right")
        table.add_column("Tokens", width=12, justify="right")
        table.add_column("Cost", width=10, justify="right")

        for summary in summaries:
            pass_rate_pct = summary["pass_rate"] * 100
            pass_rate_str = f"{pass_rate_pct:.1f}%"
            if pass_rate_pct >= 90:
                pass_rate_str = f"[green]{pass_rate_str}[/green]"
            elif pass_rate_pct < 70:
                pass_rate_str = f"[red]{pass_rate_str}[/red]"

            total_tokens = summary["total_input_tokens"] + summary["total_output_tokens"]

            table.add_row(
                summary["client_type"],
                summary["client_version"] or "-",
                str(summary["total_tests"]),
                pass_rate_str,
                f"{summary['avg_score']:.2f}",
                f"{total_tokens:,}",
                f"${summary['total_cost_usd']:.4f}",
            )

        console.print(table)
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


@batch_app.command("trend")
def batch_trend(
    client: Annotated[
        str,
        typer.Option("--client", "-c", help="Client type to show trend for"),
    ],
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help="Metric to track (pass_rate, avg_score, avg_latency_ms, total_cost_usd)",
        ),
    ] = "pass_rate",
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Number of days to include"),
    ] = 30,
) -> None:
    """
    Show daily trend for a specific client and metric.

    Displays a table of daily values for the selected metric over time.
    """
    asyncio.run(_batch_trend_async(client, metric, days))


async def _batch_trend_async(
    client: str,
    metric: str,
    days: int,
) -> None:
    """Async implementation of batch trend command."""
    from src.common.storage import StorageConfig, close_repositories, create_repositories

    try:
        config = StorageConfig()
        _, scorecard_repo, _ = await create_repositories(config)

        # Get trend data
        trend_data = await scorecard_repo.get_client_trend(client, metric, days)

        if not trend_data:
            console.print(
                f"[yellow]No data found for client '{client}' in the last {days} days[/yellow]"
            )
            raise typer.Exit(0)

        console.print()
        table = Table(
            title=f"Trend: {client} - {metric} (last {days} days)",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Date", width=15)
        table.add_column("Tests", width=10, justify="right")
        table.add_column(metric.replace("_", " ").title(), width=15, justify="right")

        for point in trend_data:
            date_str = point["date"][:10]  # Just the date part
            value = point["value"]

            # Format value based on metric type
            if metric == "pass_rate":
                value_str = f"{value * 100:.1f}%"
                if value >= 0.9:
                    value_str = f"[green]{value_str}[/green]"
                elif value < 0.7:
                    value_str = f"[red]{value_str}[/red]"
            elif metric == "avg_score":
                value_str = f"{value:.2f}"
            elif metric == "avg_latency_ms":
                value_str = f"{value:.0f}ms"
            elif metric == "total_cost_usd":
                value_str = f"${value:.4f}"
            else:
                value_str = f"{value:.2f}"

            table.add_row(
                date_str,
                str(point["total_tests"]),
                value_str,
            )

        console.print(table)
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
