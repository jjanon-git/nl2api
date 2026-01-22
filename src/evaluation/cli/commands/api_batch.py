"""
Anthropic Message Batches API integration for evaluation.

Uses Anthropic's asynchronous batch processing for:
- 50% cost reduction
- No rate limits
- Better for large evaluation runs

Usage:
    # Submit a batch to Anthropic
    python -m src.evaluation.cli.main batch api submit --limit 100

    # Check status
    python -m src.evaluation.cli.main batch api status <batch_id>

    # Poll until complete
    python -m src.evaluation.cli.main batch api poll <batch_id>

    # Download and evaluate results
    python -m src.evaluation.cli.main batch api results <batch_id>
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

# Create the api subcommand group
api_app = typer.Typer(
    name="api",
    help="Anthropic Message Batches API commands (50% cheaper, no rate limits)",
    no_args_is_help=True,
)


def _load_env() -> None:
    """Load environment variables from .env file."""
    from pathlib import Path

    # Find project root (where .env should be)
    current = Path(__file__).resolve()
    for parent in current.parents:
        env_file = parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break


def _get_anthropic_client():
    """Get Anthropic client with API key from environment."""
    _load_env()

    api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY or NL2API_ANTHROPIC_API_KEY not set")
        console.print("  Set in .env file or environment variable")
        raise typer.Exit(1)

    os.environ["ANTHROPIC_API_KEY"] = api_key

    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        console.print("[red]Error:[/red] anthropic package not installed")
        console.print("  Run: pip install anthropic")
        raise typer.Exit(1)


def _get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "CONTRACTS.py").exists():
            return parent
    return Path.cwd()


def _get_agent_config(agent_type: str = "datastream") -> tuple[str, list[dict]]:
    """
    Get system prompt and tools from the actual domain agent.

    This ensures batch evaluation uses the exact same prompts as the agents.
    """
    import sys
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    # Import agents dynamically to get their actual prompts
    if agent_type == "datastream":
        from src.nl2api.agents.datastream import DatastreamAgent
        # Create a minimal agent instance just to get prompts
        # We pass None for llm since we only need the prompt methods
        agent = DatastreamAgent(llm=None)  # type: ignore
    elif agent_type == "estimates":
        from src.nl2api.agents.estimates import EstimatesAgent
        agent = EstimatesAgent(llm=None)  # type: ignore
    elif agent_type == "fundamentals":
        from src.nl2api.agents.fundamentals import FundamentalsAgent
        agent = FundamentalsAgent(llm=None)  # type: ignore
    elif agent_type == "screening":
        from src.nl2api.agents.screening import ScreeningAgent
        agent = ScreeningAgent(llm=None)  # type: ignore
    elif agent_type == "officers":
        from src.nl2api.agents.officers import OfficersAgent
        agent = OfficersAgent(llm=None)  # type: ignore
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Get the system prompt from the agent
    system_prompt = agent.get_system_prompt()

    # Get tools and convert to API format
    tools = []
    for tool_def in agent.get_tools():
        tools.append({
            "name": tool_def.name,
            "description": tool_def.description,
            "input_schema": tool_def.parameters,
        })

    return system_prompt, tools


@api_app.command("submit")
def api_submit(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Number of test cases to evaluate"),
    ] = 100,
    category: Annotated[
        str | None,
        typer.Option("--category", "-c", help="Specific category to test"),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help="Agent to evaluate (datastream, estimates, fundamentals, screening, officers)"),
    ] = "datastream",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use (haiku or sonnet)"),
    ] = "haiku",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Create batch file but don't submit"),
    ] = False,
) -> None:
    """
    Submit evaluation batch to Anthropic's Message Batches API.

    Creates a batch from test fixtures and submits it for async processing.
    Uses the actual agent's system prompt and tools for accurate evaluation.
    Use 'batch api status' or 'batch api poll' to track progress.
    """
    import sys
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from tests.unit.nl2api.fixture_loader import FixtureLoader

    # Model mapping
    model_map = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-sonnet-4-20250514",
    }
    model_id = model_map.get(model, model)

    console.print(f"\n[bold]Anthropic Batch API Evaluation[/bold]")
    console.print(f"  Agent: {agent}")
    console.print(f"  Model: {model_id}")
    console.print(f"  Limit: {limit}")
    if category:
        console.print(f"  Category: {category}")
    console.print()

    # Get agent's actual prompt and tools
    with console.status(f"Loading {agent} agent configuration..."):
        try:
            system_prompt, tools = _get_agent_config(agent)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    console.print(f"  Loaded agent prompt ({len(system_prompt)} chars) and {len(tools)} tools")

    # Load test fixtures
    with console.status("Loading test fixtures..."):
        loader = FixtureLoader()
        if category:
            test_cases = loader.load_category(category)
        else:
            test_cases = list(loader.iterate_all())

        if limit:
            test_cases = test_cases[:limit]

    console.print(f"  Loaded [cyan]{len(test_cases)}[/cyan] test cases")

    # Build batch requests
    batch_file = project_root / "batch_requests.jsonl"

    with open(batch_file, "w") as f:
        for tc in test_cases:
            request = {
                "custom_id": tc.id,
                "params": {
                    "model": model_id,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": tc.nl_query}],
                    "tools": tools,
                    "tool_choice": {"type": "auto"}
                }
            }
            f.write(json.dumps(request) + "\n")

    file_size = batch_file.stat().st_size / 1024
    console.print(f"  Created batch file: {batch_file} ({file_size:.1f} KB)")

    if dry_run:
        console.print("\n[yellow]Dry run - batch not submitted[/yellow]")
        raise typer.Exit(0)

    # Submit to Anthropic
    client = _get_anthropic_client()

    with console.status("Submitting batch to Anthropic..."):
        with open(batch_file, "rb") as f:
            batch = client.beta.messages.batches.create(
                requests=[json.loads(line) for line in f]
            )

    console.print(f"\n[green]Batch submitted successfully![/green]")
    console.print(f"  Batch ID: [cyan]{batch.id}[/cyan]")
    console.print(f"  Status: {batch.processing_status}")

    # Save batch info
    batch_info = {
        "batch_id": batch.id,
        "created_at": datetime.now().isoformat(),
        "limit": limit,
        "category": category,
        "agent": agent,
        "model": model_id,
    }
    batch_info_file = project_root / "batch_info.json"
    with open(batch_info_file, "w") as f:
        json.dump(batch_info, f, indent=2)

    console.print(f"\n[dim]Batch info saved to: {batch_info_file}[/dim]")
    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"  Check status:  python -m src.evaluation.cli.main batch api status {batch.id}")
    console.print(f"  Poll progress: python -m src.evaluation.cli.main batch api poll {batch.id}")
    console.print(f"  Get results:   python -m src.evaluation.cli.main batch api results {batch.id}")


@api_app.command("status")
def api_status(
    batch_id: Annotated[str, typer.Argument(help="Batch ID to check")],
) -> None:
    """
    Check status of an Anthropic batch.

    Shows processing progress and request counts.
    """
    client = _get_anthropic_client()

    with console.status(f"Checking batch {batch_id}..."):
        batch = client.beta.messages.batches.retrieve(batch_id)

    console.print()
    table = Table(title=f"Batch Status: [cyan]{batch_id}[/cyan]")
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")

    # Status with color
    status = batch.processing_status
    if status == "ended":
        status_display = f"[green]{status}[/green]"
    elif status == "in_progress":
        status_display = f"[yellow]{status}[/yellow]"
    else:
        status_display = status

    table.add_row("Status", status_display)
    table.add_row("Processing", str(batch.request_counts.processing))
    table.add_row("Succeeded", f"[green]{batch.request_counts.succeeded}[/green]")
    table.add_row("Errored", f"[red]{batch.request_counts.errored}[/red]" if batch.request_counts.errored > 0 else "0")
    table.add_row("Canceled", str(batch.request_counts.canceled))
    table.add_row("Expired", str(batch.request_counts.expired))

    if batch.created_at:
        table.add_row("Created", str(batch.created_at))
    if batch.ended_at:
        table.add_row("Ended", str(batch.ended_at))

    console.print(table)
    console.print()


@api_app.command("poll")
def api_poll(
    batch_id: Annotated[str, typer.Argument(help="Batch ID to poll")],
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Poll interval in seconds"),
    ] = 30,
) -> None:
    """
    Poll batch until complete.

    Continuously checks status until the batch finishes processing.
    """
    client = _get_anthropic_client()

    console.print(f"\n[bold]Polling batch {batch_id}[/bold]")
    console.print(f"  Interval: {interval}s")
    console.print()

    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)

        succeeded = batch.request_counts.succeeded
        errored = batch.request_counts.errored
        processing = batch.request_counts.processing
        total = succeeded + errored + processing

        timestamp = datetime.now().strftime("%H:%M:%S")

        if batch.processing_status == "ended":
            console.print(f"  [{timestamp}] [green]Complete![/green] Succeeded: {succeeded}/{total}, Errors: {errored}")
            console.print(f"\n[green]Batch complete![/green]")
            console.print(f"\nGet results with:")
            console.print(f"  python -m src.evaluation.cli.main batch api results {batch_id}")
            break
        else:
            console.print(f"  [{timestamp}] Processing... Succeeded: {succeeded}/{total}, Errors: {errored}")
            time.sleep(interval)


@api_app.command("results")
def api_results(
    batch_id: Annotated[str, typer.Argument(help="Batch ID to get results for")],
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Limit for fixture matching"),
    ] = 100,
    category: Annotated[
        str | None,
        typer.Option("--category", "-c", help="Category filter for fixture matching"),
    ] = None,
    export: Annotated[
        Path | None,
        typer.Option("--export", "-e", help="Export detailed results to JSON"),
    ] = None,
) -> None:
    """
    Download and evaluate batch results.

    Downloads results from Anthropic, evaluates against test fixtures,
    and displays summary statistics.
    """
    import sys
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from tests.unit.nl2api.fixture_loader import FixtureLoader
    from CONTRACTS import ToolRegistry

    client = _get_anthropic_client()

    # Download results
    console.print(f"\n[bold]Downloading results for batch {batch_id}[/bold]")

    results_file = project_root / f"batch_results_{batch_id}.jsonl"

    with console.status("Downloading from Anthropic..."):
        with open(results_file, "w") as f:
            for result in client.beta.messages.batches.results(batch_id):
                f.write(json.dumps({
                    "custom_id": result.custom_id,
                    "result": result.result.model_dump() if hasattr(result.result, 'model_dump') else result.result
                }) + "\n")

    console.print(f"  Saved to: {results_file}")

    # Load test cases for comparison
    with console.status("Loading test fixtures..."):
        loader = FixtureLoader()
        if category:
            test_cases = loader.load_category(category)
        else:
            test_cases = list(loader.iterate_all())

        if limit:
            test_cases = test_cases[:limit]

        expected_by_id = {tc.id: tc for tc in test_cases}

    # Evaluate results
    console.print("  Evaluating results...")

    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    passed = 0
    failed = 0
    errors = 0
    failures = []

    for result in results:
        custom_id = result["custom_id"]
        expected = expected_by_id.get(custom_id)

        if not expected:
            errors += 1
            continue

        result_data = result["result"]

        # Check for API errors
        if result_data.get("type") == "error":
            errors += 1
            failures.append({
                "id": custom_id,
                "query": expected.nl_query[:50],
                "error": result_data.get("error", {}).get("message", "Unknown error")
            })
            continue

        # Extract tool calls
        message = result_data.get("message", {})
        content = message.get("content", [])
        tool_calls = [c for c in content if c.get("type") == "tool_use"]

        if not tool_calls:
            failed += 1
            failures.append({
                "id": custom_id,
                "query": expected.nl_query[:50],
                "error": "No tool calls returned"
            })
            continue

        # Compare tool calls
        actual_func = tool_calls[0].get("name", "")
        expected_func = expected.expected_tool_calls[0].get("function", "") if expected.expected_tool_calls else ""

        actual_normalized = ToolRegistry.normalize(actual_func)
        expected_normalized = ToolRegistry.normalize(expected_func)

        if actual_normalized == expected_normalized:
            # Check fields
            actual_args = tool_calls[0].get("input", {})
            expected_args = expected.expected_tool_calls[0].get("arguments", {})

            actual_fields = set(actual_args.get("fields", []))
            expected_fields = set(expected_args.get("fields", []))

            if not expected_fields or actual_fields == expected_fields:
                passed += 1
            else:
                failed += 1
                failures.append({
                    "id": custom_id,
                    "query": expected.nl_query[:50],
                    "error": f"Fields mismatch: expected {expected_fields}, got {actual_fields}"
                })
        else:
            failed += 1
            failures.append({
                "id": custom_id,
                "query": expected.nl_query[:50],
                "error": f"Function mismatch: expected {expected_func}, got {actual_func}"
            })

    total = passed + failed + errors
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Display summary
    console.print()
    console.print("=" * 60)
    console.print("[bold]BATCH EVALUATION COMPLETE[/bold]")
    console.print("=" * 60)

    table = Table(show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total", str(total))
    table.add_row("Passed", f"[green]{passed}[/green] ({pass_rate:.1f}%)")
    table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    table.add_row("Errors", f"[yellow]{errors}[/yellow]" if errors > 0 else "0")

    console.print(table)

    if failures:
        console.print(f"\n[bold]Sample Failures (first 10):[/bold]")
        for f in failures[:10]:
            console.print(f"  - {f['query']}...")
            console.print(f"    [dim]{f['error']}[/dim]")

    console.print()

    # Save summary
    summary = {
        "batch_id": batch_id,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": pass_rate,
        "failures": failures[:20],
    }

    summary_file = project_root / f"batch_summary_{batch_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[dim]Summary saved to: {summary_file}[/dim]")

    if export:
        with open(export, "w") as f:
            json.dump({"summary": summary, "all_failures": failures}, f, indent=2)
        console.print(f"[dim]Detailed results exported to: {export}[/dim]")

    # Exit with appropriate code
    if failed > 0 or errors > 0:
        raise typer.Exit(1)
