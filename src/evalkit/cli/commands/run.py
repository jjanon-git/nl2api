"""
Run command - Execute evaluation on a test case file or from storage.

Usage:
    # From local file
    python -m src.evaluation.cli.main run tests/fixtures/search_products.json

    # From storage by ID
    python -m src.evaluation.cli.main run --test-id <uuid>

    # Save scorecard to storage
    python -m src.evaluation.cli.main run tests/fixtures/search_products.json --save
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from CONTRACTS import (
    EvalContext,
    EvaluationConfig,
    SystemResponse,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evaluation.packs import NL2APIPack

logger = logging.getLogger(__name__)

console = Console()


def run_command(
    test_file: Path | None = typer.Argument(
        None,
        help="Path to test case JSON file (optional if using --test-id)",
        exists=False,  # We validate ourselves to allow either file or test-id
    ),
    test_id: str | None = typer.Option(
        None,
        "--test-id",
        "-t",
        help="Test case ID to fetch from storage",
    ),
    response_file: Path | None = typer.Option(
        None,
        "--response",
        "-r",
        help="Path to system response JSON file (optional, uses 'response' from test file if not provided)",
    ),
    save: bool = typer.Option(
        False,
        "--save",
        "-s",
        help="Save scorecard to storage after evaluation",
    ),
    batch_id: str | None = typer.Option(
        None,
        "--batch-id",
        "-b",
        help="Batch ID to associate with the scorecard (requires --save)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output including artifacts",
    ),
) -> None:
    """
    Run evaluation on a test case.

    Reads a test case from JSON file or storage, evaluates the system response,
    and prints the scorecard to the terminal. Optionally saves to storage.
    """
    # Validate inputs
    if test_file is None and test_id is None:
        console.print("[red]Error:[/red] Provide either a test file path or --test-id")
        raise typer.Exit(1)

    if test_file is not None and not test_file.exists():
        console.print(f"[red]Error:[/red] File not found: {test_file}")
        raise typer.Exit(1)

    # Run async evaluation
    asyncio.run(_run_async(test_file, test_id, response_file, save, batch_id, verbose))


async def _run_async(
    test_file: Path | None,
    test_id: str | None,
    response_file: Path | None,
    save: bool,
    batch_id: str | None,
    verbose: bool,
) -> None:
    """Async implementation of run command."""
    from src.evalkit.common.storage import StorageConfig, close_repositories, create_repositories

    test_case: TestCase | None = None
    test_case_repo = None
    scorecard_repo = None

    try:
        # Fetch test case from storage or file
        if test_id:
            config = StorageConfig()
            test_case_repo, scorecard_repo, _ = await create_repositories(config)
            test_case = await test_case_repo.get(test_id)

            if not test_case:
                console.print(f"[red]Error:[/red] Test case not found: {test_id}")
                raise typer.Exit(1)

            console.print(f"[dim]Loaded test case from storage: {test_id}[/dim]")

        elif test_file:
            test_data = json.loads(test_file.read_text())
            test_case = _parse_test_case(test_data)

        # Load or extract system response
        if response_file:
            response_data = json.loads(response_file.read_text())
        elif test_file:
            test_data = json.loads(test_file.read_text())
            if "response" in test_data:
                response_data = test_data["response"]
            elif "system_response" in test_data:
                response_data = test_data["system_response"]
            else:
                console.print(
                    "[red]Error:[/red] No response file provided and test case has no 'response' field"
                )
                raise typer.Exit(1)
        else:
            console.print(
                "[red]Error:[/red] No response provided. Use --response or include 'response' in test file"
            )
            raise typer.Exit(1)

        system_response = _parse_system_response(response_data)

        # Run evaluation using NL2APIPack
        config = EvaluationConfig()
        pack = NL2APIPack(
            execution_enabled=config.execution_stage_enabled,
            semantics_enabled=config.semantics_stage_enabled,
            numeric_tolerance=config.numeric_tolerance,
            temporal_mode=config.temporal_mode,
            evaluation_date=config.evaluation_date,
            relative_date_fields=config.relative_date_fields,
            fiscal_year_end_month=config.fiscal_year_end_month,
        )

        # Convert SystemResponse to system_output dict expected by pack
        system_output = {
            "raw_output": system_response.raw_output,
            "nl_response": system_response.nl_response,
        }

        scorecard = await pack.evaluate(
            test_case=test_case,
            system_output=system_output,
            context=EvalContext(worker_id="cli-local"),
        )

        # Override batch_id if provided
        if batch_id:
            # Create a new scorecard with the batch_id
            scorecard = scorecard.model_copy(update={"batch_id": batch_id})

        # Save scorecard if requested
        if save:
            if scorecard_repo is None:
                config = StorageConfig()
                _, scorecard_repo, _ = await create_repositories(config)

            await scorecard_repo.save(scorecard)
            console.print(f"[dim]Saved scorecard to storage: {scorecard.scorecard_id}[/dim]")

        # Display results
        _display_results(test_case, scorecard, verbose)

        # Exit with appropriate code
        if scorecard.overall_passed:
            raise typer.Exit(0)
        else:
            raise typer.Exit(1)

    except typer.Exit:
        raise  # Re-raise typer.Exit to preserve exit code
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in input file: {e}")
        raise typer.Exit(2)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(2)
    finally:
        # Cleanup storage connections
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def _parse_test_case(data: dict[str, Any]) -> TestCase:
    """Parse test case from JSON data."""
    # Handle expected_tool_calls
    tool_calls = []
    for tc in data.get("expected_tool_calls", []):
        tool_calls.append(
            ToolCall(
                tool_name=tc.get("tool_name") or tc.get("name"),
                arguments=tc.get("arguments", {}),
            )
        )

    # Handle metadata
    metadata_data = data.get("metadata", {})
    metadata = TestCaseMetadata(
        api_version=metadata_data.get("api_version", "v1.0.0"),
        complexity_level=metadata_data.get("complexity_level", 1),
        tags=tuple(metadata_data.get("tags", [])),
    )

    return TestCase(
        id=data.get("id", "test-case"),
        nl_query=data["nl_query"],
        expected_tool_calls=tuple(tool_calls),
        expected_raw_data=data.get("expected_raw_data"),
        expected_nl_response=data.get("expected_nl_response", ""),
        metadata=metadata,
    )


def _parse_system_response(data: dict[str, Any]) -> SystemResponse:
    """Parse system response from JSON data."""
    # raw_output can be provided as string or we serialize tool_calls
    if "raw_output" in data:
        raw_output = data["raw_output"]
        if not isinstance(raw_output, str):
            raw_output = json.dumps(raw_output)
    elif "tool_calls" in data:
        raw_output = json.dumps(data["tool_calls"])
    else:
        raw_output = json.dumps(data)

    return SystemResponse(
        raw_output=raw_output,
        nl_response=data.get("nl_response"),
        latency_ms=data.get("latency_ms", 0),
    )


def _display_results(test_case: TestCase, scorecard: Any, verbose: bool) -> None:
    """Display evaluation results in a formatted way."""
    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]Test:[/bold] {test_case.id}\n"
            f"[bold]Query:[/bold] {test_case.nl_query[:80]}{'...' if len(test_case.nl_query) > 80 else ''}",
            title="Evaluation Results",
        )
    )

    # Results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Stage", width=20)
    table.add_column("Status", width=10)
    table.add_column("Score", width=10)
    table.add_column("Details", width=40)

    # Stage 1: Syntax
    syntax = scorecard.syntax_result
    status_icon = "[green]PASS[/green]" if syntax.passed else "[red]FAIL[/red]"
    table.add_row(
        "1. Syntax",
        status_icon,
        f"{syntax.score:.2f}",
        syntax.reason or "",
    )

    # Stage 2: Logic
    if scorecard.logic_result:
        logic = scorecard.logic_result
        status_icon = "[green]PASS[/green]" if logic.passed else "[red]FAIL[/red]"
        table.add_row(
            "2. Logic",
            status_icon,
            f"{logic.score:.2f}",
            logic.reason or "",
        )
    else:
        table.add_row("2. Logic", "[dim]SKIP[/dim]", "-", "Skipped (syntax failed)")

    # Stage 3: Execution (placeholder)
    if scorecard.execution_result:
        execution = scorecard.execution_result
        status_icon = "[green]PASS[/green]" if execution.passed else "[red]FAIL[/red]"
        table.add_row(
            "3. Execution",
            status_icon,
            f"{execution.score:.2f}",
            execution.reason or "",
        )
    else:
        table.add_row("3. Execution", "[dim]SKIP[/dim]", "-", "Disabled (Sprint 1)")

    # Stage 4: Semantics (placeholder)
    if scorecard.semantics_result:
        semantics = scorecard.semantics_result
        status_icon = "[green]PASS[/green]" if semantics.passed else "[red]FAIL[/red]"
        table.add_row(
            "4. Semantics",
            status_icon,
            f"{semantics.score:.2f}",
            semantics.reason or "",
        )
    else:
        table.add_row("4. Semantics", "[dim]SKIP[/dim]", "-", "Disabled (Sprint 1)")

    console.print(table)

    # Overall result
    console.print()
    if scorecard.overall_passed:
        console.print(
            "[bold green]Overall: PASS[/bold green]", f"(score: {scorecard.overall_score:.2f})"
        )
    else:
        console.print(
            "[bold red]Overall: FAIL[/bold red]", f"(score: {scorecard.overall_score:.2f})"
        )

    # Verbose output
    if verbose:
        console.print()
        console.print("[bold]Artifacts:[/bold]")

        if scorecard.syntax_result.artifacts:
            console.print("  Syntax:", scorecard.syntax_result.artifacts)

        if scorecard.logic_result and scorecard.logic_result.artifacts:
            console.print("  Logic:", scorecard.logic_result.artifacts)

        if scorecard.generated_tool_calls:
            console.print()
            console.print("[bold]Generated Tool Calls:[/bold]")
            for tc in scorecard.generated_tool_calls:
                console.print(f"  - {tc.tool_name}({tc.arguments})")

    console.print()
