"""
Eval Matrix CLI Commands

Multi-dimensional evaluation framework for comparing:
- Different components (orchestrator, router, resolver, individual agents)
- Different LLM models (Sonnet, Haiku, GPT-4o, etc.)
- Different configurations

Usage:
    # Evaluate DatastreamAgent with Haiku
    eval matrix run --component datastream --llm claude-3-5-haiku-20241022 --limit 50

    # Evaluate full orchestrator with Sonnet
    eval matrix run --component orchestrator --llm claude-3-5-sonnet-20241022 --limit 50

    # Compare runs
    eval matrix compare --runs abc123,def456

    # View component options
    eval matrix components
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# Create the matrix app (subcommand group)
matrix_app = typer.Typer(
    name="matrix",
    help="Multi-dimensional evaluation (components × LLMs × configs)",
    no_args_is_help=True,
)


@matrix_app.command("components")
def list_components() -> None:
    """
    List available components for evaluation.

    Shows all evaluable components: orchestrator, router, resolver, and individual agents.
    """
    from src.nl2api.agents import list_available_agents

    console.print()
    table = Table(title="Available Components", show_header=True, header_style="bold")
    table.add_column("Component", width=15)
    table.add_column("Type", width=12)
    table.add_column("Description", width=50)

    # System components
    table.add_row(
        "orchestrator",
        "system",
        "Full NL2API pipeline (routing + resolution + agent + response)",
    )
    table.add_row(
        "router",
        "system",
        "Query → domain routing decisions only",
    )
    table.add_row(
        "resolver",
        "system",
        "Entity resolution (company name → RIC)",
    )

    # Individual agents
    agents = list_available_agents()
    for agent_name in agents:
        table.add_row(
            agent_name,
            "agent",
            f"{agent_name.capitalize()}Agent - domain-specific tool generation",
        )

    console.print(table)
    console.print()
    console.print("[dim]Use 'eval matrix run --component <name>' to evaluate a component[/dim]")
    console.print()


@matrix_app.command("run")
def matrix_run(
    component: Annotated[
        str,
        typer.Option(
            "--component",
            "-c",
            help="Component to evaluate (orchestrator, router, resolver, or agent name)",
        ),
    ],
    llm: Annotated[
        str,
        typer.Option("--llm", "-l", help="LLM model to use"),
    ] = "claude-3-5-sonnet-20241022",
    client: Annotated[
        str,
        typer.Option("--client", help="Client identifier for tracking"),
    ] = "internal",
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Filter by tags (can be repeated)"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Maximum test cases to run"),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Concurrent evaluations"),
    ] = 10,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
) -> None:
    """
    Run evaluation matrix for a specific component.

    Evaluates a single component (orchestrator, router, resolver, or individual agent)
    with a specific LLM model. Results are tracked with client/version metadata for
    comparison across configurations.

    COMPONENTS:
    - orchestrator: Full end-to-end NL2API pipeline
    - router: Query → domain routing only
    - resolver: Entity resolution only
    - datastream/estimates/fundamentals/officers/screening: Individual agents

    EXAMPLES:
        # Compare Haiku vs Sonnet for datastream queries
        eval matrix run --component datastream --llm claude-3-5-haiku-20241022 --limit 50
        eval matrix run --component datastream --llm claude-3-5-sonnet-20241022 --limit 50
        eval matrix compare --client internal

        # Test full orchestrator with different models
        eval matrix run --component orchestrator --llm gpt-4o --limit 20
    """
    asyncio.run(
        _matrix_run_async(
            component=component,
            llm=llm,
            client=client,
            tags=tags,
            limit=limit,
            concurrency=concurrency,
            verbose=verbose,
        )
    )


async def _matrix_run_async(
    component: str,
    llm: str,
    client: str,
    tags: list[str] | None,
    limit: int | None,
    concurrency: int,
    verbose: bool,
) -> None:
    """Async implementation of matrix run command."""
    from src.evalkit.batch import BatchRunner, BatchRunnerConfig
    from src.evalkit.batch.response_generators import (
        create_entity_resolver_generator,
        create_nl2api_generator,
        create_routing_generator,
        create_tool_only_generator,
    )
    from src.evalkit.common.storage import StorageConfig, close_repositories, create_repositories
    from src.nl2api.agents import get_agent_by_name, list_available_agents
    from src.nl2api.config import NL2APIConfig
    from src.nl2api.llm.factory import create_llm_provider

    try:
        config = StorageConfig()
        test_case_repo, scorecard_repo, batch_repo = await create_repositories(config)

        # Create LLM provider
        cfg = NL2APIConfig()
        llm_provider = create_llm_provider(
            provider=cfg.llm_provider,
            api_key=cfg.get_llm_api_key(),
            model=llm,
        )

        # Determine eval mode and create response generator
        available_agents = list_available_agents()
        component_lower = component.lower()

        if component_lower == "orchestrator":
            eval_mode = "orchestrator"
            from src.nl2api.agents import (
                DatastreamAgent,
                EstimatesAgent,
                FundamentalsAgent,
                OfficersAgent,
                ScreeningAgent,
            )
            from src.nl2api.orchestrator import NL2APIOrchestrator

            # Create agents with specified LLM
            agents = {
                "datastream": DatastreamAgent(llm=llm_provider),
                "estimates": EstimatesAgent(llm=llm_provider),
                "fundamentals": FundamentalsAgent(llm=llm_provider),
                "officers": OfficersAgent(llm=llm_provider),
                "screening": ScreeningAgent(llm=llm_provider),
            }

            orchestrator = NL2APIOrchestrator(llm=llm_provider, agents=agents)
            response_generator = create_nl2api_generator(orchestrator)
            console.print(f"[green]Evaluating orchestrator with {llm}[/green]\n")

        elif component_lower == "router":
            eval_mode = "routing"
            from src.nl2api.routing.llm_router import LLMToolRouter
            from src.nl2api.routing.protocols import RoutingToolDefinition, ToolProvider

            # Create stub tool providers for routing evaluation
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
                    "datastream", "Stock prices, market data, trading volume", ("prices", "volume")
                ),
                StubToolProvider(
                    "estimates",
                    "Analyst forecasts, EPS estimates, price targets",
                    ("eps", "forecasts"),
                ),
                StubToolProvider(
                    "fundamentals",
                    "Financial statements, balance sheet, income statement",
                    ("revenue", "financials"),
                ),
                StubToolProvider(
                    "officers",
                    "Executives, board members, CEO, CFO, compensation",
                    ("ceo", "board"),
                ),
                StubToolProvider(
                    "screening",
                    "Stock screening, rankings, top N queries",
                    ("screening", "ranking"),
                ),
            ]

            router = LLMToolRouter(llm=llm_provider, tool_providers=tool_providers)
            response_generator = create_routing_generator(router)
            console.print(f"[green]Evaluating router with {llm}[/green]\n")

        elif component_lower == "resolver":
            eval_mode = "resolver"
            from src.evalkit.common.storage.postgres.client import get_pool
            from src.nl2api.resolution.resolver import ExternalEntityResolver

            try:
                db_pool = await get_pool()
                resolver = ExternalEntityResolver(db_pool=db_pool, _internal=True)
                console.print("[green]Evaluating resolver with database (2.9M entities)[/green]\n")
            except RuntimeError:
                resolver = ExternalEntityResolver(_internal=True)
                console.print("[yellow]Evaluating resolver (static mappings only)[/yellow]\n")

            response_generator = create_entity_resolver_generator(resolver)

        elif component_lower in available_agents:
            eval_mode = "tool_only"
            agent = get_agent_by_name(component_lower, llm=llm_provider)
            response_generator = create_tool_only_generator(agent)
            console.print(f"[green]Evaluating {component_lower} agent with {llm}[/green]\n")

        else:
            available = ", ".join(["orchestrator", "router", "resolver"] + available_agents)
            console.print(f"[red]Error:[/red] Unknown component '{component}'")
            console.print(f"Available components: {available}")
            raise typer.Exit(1)

        # Configure batch runner
        runner_config = BatchRunnerConfig(
            max_concurrency=concurrency,
            show_progress=True,
            verbose=verbose,
            client_type=client,
            client_version=llm,  # Use LLM model as version for cost tracking
            eval_mode=eval_mode,
        )

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=runner_config,
        )

        batch_job = await runner.run(
            tags=tags,
            limit=limit,
            response_simulator=response_generator,
        )

        if batch_job is None:
            console.print("[yellow]No test cases found matching filters[/yellow]")
            raise typer.Exit(0)

        # Print batch ID for comparison
        console.print(f"[bold]Run ID:[/bold] [cyan]{batch_job.batch_id}[/cyan]")
        console.print(
            f"[dim]Use 'eval matrix compare --runs {batch_job.batch_id},...' to compare runs[/dim]"
        )

        if batch_job.failed_count > 0:
            raise typer.Exit(1)
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


@matrix_app.command("compare")
def matrix_compare(
    runs: Annotated[
        str,
        typer.Option("--runs", "-r", help="Comma-separated batch IDs to compare"),
    ],
    metric: Annotated[
        str,
        typer.Option("--metric", "-m", help="Metrics to show (accuracy,cost,latency or all)"),
    ] = "all",
) -> None:
    """
    Compare multiple evaluation runs.

    Shows side-by-side comparison of pass rate, average score, token usage,
    estimated cost, and latency for each run.

    EXAMPLE:
        eval matrix run --component datastream --llm claude-3-5-haiku-20241022 --limit 50
        # Returns: Run ID: abc123

        eval matrix run --component datastream --llm claude-3-5-sonnet-20241022 --limit 50
        # Returns: Run ID: def456

        eval matrix compare --runs abc123,def456
    """
    asyncio.run(_matrix_compare_async(runs, metric))


async def _matrix_compare_async(runs: str, metric: str) -> None:
    """Async implementation of matrix compare command."""
    from src.evalkit.batch.pricing import format_cost
    from src.evalkit.common.storage import StorageConfig, close_repositories, create_repositories

    try:
        config = StorageConfig()
        _, scorecard_repo, batch_repo = await create_repositories(config)

        run_ids = [r.strip() for r in runs.split(",")]

        if len(run_ids) < 1:
            console.print("[red]Error:[/red] Provide at least one run ID")
            raise typer.Exit(1)

        console.print()
        table = Table(title="Run Comparison", show_header=True, header_style="bold")
        table.add_column("Run ID", width=38)
        table.add_column("Eval Mode", width=12)
        table.add_column("Model", width=28)
        table.add_column("Tests", width=8, justify="right")
        table.add_column("Pass Rate", width=10, justify="right")
        table.add_column("Avg Score", width=10, justify="right")

        if metric in ("all", "cost"):
            table.add_column("Input Tok", width=10, justify="right")
            table.add_column("Output Tok", width=10, justify="right")
            table.add_column("Est. Cost", width=10, justify="right")

        if metric in ("all", "latency"):
            table.add_column("Avg Latency", width=12, justify="right")

        for run_id in run_ids:
            batch = await batch_repo.get(run_id)
            if not batch:
                console.print(f"[yellow]Warning: Batch {run_id} not found, skipping[/yellow]")
                continue

            # Get scorecards for this batch
            scorecards = await scorecard_repo.get_by_batch(run_id)

            if not scorecards:
                console.print(f"[yellow]Warning: No scorecards for {run_id}, skipping[/yellow]")
                continue

            # Calculate metrics
            total = len(scorecards)
            passed = sum(1 for sc in scorecards if sc.overall_passed)
            pass_rate = passed / total * 100 if total > 0 else 0
            avg_score = sum(sc.overall_score for sc in scorecards) / total if total > 0 else 0

            # Get client metadata from first scorecard
            client_version = scorecards[0].client_version or "-"
            eval_mode = scorecards[0].eval_mode or "-"

            # Token and cost totals
            total_input = sum(sc.input_tokens or 0 for sc in scorecards)
            total_output = sum(sc.output_tokens or 0 for sc in scorecards)
            total_cost = sum(sc.estimated_cost_usd or 0 for sc in scorecards)

            # Average latency
            avg_latency = sum(sc.total_latency_ms for sc in scorecards) / total if total > 0 else 0

            # Format pass rate with color
            pass_rate_str = f"{pass_rate:.1f}%"
            if pass_rate >= 90:
                pass_rate_str = f"[green]{pass_rate_str}[/green]"
            elif pass_rate < 70:
                pass_rate_str = f"[red]{pass_rate_str}[/red]"

            row = [
                run_id,
                eval_mode,
                client_version[:28] if len(client_version) > 28 else client_version,
                str(total),
                pass_rate_str,
                f"{avg_score:.2f}",
            ]

            if metric in ("all", "cost"):
                row.extend(
                    [
                        f"{total_input:,}",
                        f"{total_output:,}",
                        format_cost(total_cost),
                    ]
                )

            if metric in ("all", "latency"):
                row.append(f"{avg_latency:.0f}ms")

            table.add_row(*row)

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
