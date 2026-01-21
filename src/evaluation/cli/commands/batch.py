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
            "--mode", "-m",
            help="Response mode: resolver (real), orchestrator (full), simulated"
        ),
    ] = "resolver",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
) -> None:
    """
    Run batch evaluation on test cases from database.

    Fetches test cases matching the filters, runs evaluations concurrently,
    and saves scorecards with a batch ID for tracking.

    MODES:
    - resolver (default): Uses real EntityResolver for accuracy measurement.
      Results are persisted and tracked over time.
    - orchestrator: Uses full NL2API orchestrator (requires LLM API key).
      End-to-end accuracy measurement.
    - simulated: Always returns correct answers (for pipeline testing only).
      Should NOT be used for accuracy tracking - use unit tests instead.
    """
    if mode not in ("resolver", "orchestrator", "simulated"):
        console.print(f"[red]Error:[/red] Invalid mode '{mode}'.")
        console.print("Use 'resolver', 'orchestrator', or 'simulated'.")
        raise typer.Exit(1)

    asyncio.run(_batch_run_async(
        tags=tags,
        complexity_min=complexity_min,
        complexity_max=complexity_max,
        limit=limit,
        concurrency=concurrency,
        mode=mode,
        verbose=verbose,
    ))


async def _batch_run_async(
    tags: list[str] | None,
    complexity_min: int | None,
    complexity_max: int | None,
    limit: int | None,
    concurrency: int,
    mode: str,
    verbose: bool,
) -> None:
    """Async implementation of batch run command."""
    from src.common.storage import (
        StorageConfig,
        close_repositories,
        create_repositories,
    )
    from src.evaluation.batch import BatchRunner, BatchRunnerConfig
    from src.evaluation.batch.response_generators import (
        create_entity_resolver_generator,
        create_nl2api_generator,
        simulate_correct_response,
    )

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
            from src.nl2api.resolution.resolver import ExternalEntityResolver
            resolver = ExternalEntityResolver()
            response_generator = create_entity_resolver_generator(resolver)
            console.print("[green]Using real EntityResolver for accuracy measurement.[/green]\n")
        elif mode == "orchestrator":
            # Use full NL2API orchestrator
            from src.nl2api.orchestrator import NL2APIOrchestrator
            orchestrator = NL2APIOrchestrator()
            response_generator = create_nl2api_generator(orchestrator)
            console.print("[green]Using full NL2API orchestrator (requires LLM API key).[/green]\n")

        runner_config = BatchRunnerConfig(
            max_concurrency=concurrency,
            show_progress=True,
            verbose=verbose,
        )

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
                export_data["results"].append({
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
                })

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
        table.add_column("Batch ID", width=38)
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

            table.add_row(
                batch.batch_id,
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
