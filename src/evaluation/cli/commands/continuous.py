"""
Continuous evaluation CLI commands.

Usage:
    # Start scheduler
    python -m src.evaluation.cli.main continuous start --config eval-schedules.yaml

    # Check status
    python -m src.evaluation.cli.main continuous status

    # Trigger a schedule manually
    python -m src.evaluation.cli.main continuous trigger --client mcp_claude

    # View alerts
    python -m src.evaluation.cli.main continuous alerts --days 7

    # Acknowledge an alert
    python -m src.evaluation.cli.main continuous acknowledge <alert_id>
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()

# Create the continuous app (subcommand group)
continuous_app = typer.Typer(
    name="continuous",
    help="Continuous evaluation commands",
    no_args_is_help=True,
)


@continuous_app.command("start")
def continuous_start(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to YAML config file"),
    ],
    foreground: Annotated[
        bool,
        typer.Option("--foreground", "-f", help="Run in foreground (blocking)"),
    ] = True,
) -> None:
    """
    Start the continuous evaluation scheduler.

    Reads schedule configuration from a YAML file and starts
    periodic evaluation runs with regression detection.
    """
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    asyncio.run(_continuous_start_async(config_path, foreground))


async def _continuous_start_async(config_path: Path, foreground: bool) -> None:
    """Async implementation of continuous start command."""
    from src.common.storage import StorageConfig, close_repositories, create_repositories
    from src.common.storage.postgres.client import get_pool
    from src.evaluation.continuous import EvalScheduler
    from src.evaluation.continuous.config import ContinuousConfig

    try:
        # Load configuration
        config = ContinuousConfig.from_yaml_file(str(config_path))

        console.print("\n[bold]Continuous Evaluation Scheduler[/bold]")
        console.print(f"  Loaded {len(config.schedules)} schedules from {config_path}")
        console.print(f"  Check interval: {config.check_interval_seconds}s")
        console.print()

        # Show schedules
        for schedule in config.schedules:
            status = "[green]enabled[/green]" if schedule.enabled else "[yellow]disabled[/yellow]"
            console.print(f"  • {schedule.name}: {schedule.cron_expression} ({status})")

        console.print()

        # Create repositories
        storage_config = StorageConfig()
        test_case_repo, scorecard_repo, batch_repo = await create_repositories(storage_config)

        # Get database pool
        try:
            pool = await get_pool()
        except RuntimeError:
            pool = None

        # Create and start scheduler
        scheduler = EvalScheduler(
            config=config,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            pool=pool,
        )

        console.print("[green]Starting scheduler...[/green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        try:
            await scheduler.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping scheduler...[/yellow]")
            scheduler.stop()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        try:
            await close_repositories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


@continuous_app.command("status")
def continuous_status() -> None:
    """
    Show status of the continuous evaluation scheduler.

    Displays active schedules and their current state.
    """
    console.print(
        "[yellow]Note:[/yellow] Scheduler status is only available when scheduler is running."
    )
    console.print("Start the scheduler with: [cyan]continuous start --config <file>[/cyan]")


@continuous_app.command("trigger")
def continuous_trigger(
    schedule_name: Annotated[
        str,
        typer.Argument(help="Name of the schedule to trigger"),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to YAML config file"),
    ],
) -> None:
    """
    Manually trigger a scheduled evaluation run.

    Runs the specified schedule immediately without waiting for the cron time.
    """
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    asyncio.run(_continuous_trigger_async(schedule_name, config_path))


async def _continuous_trigger_async(schedule_name: str, config_path: Path) -> None:
    """Async implementation of continuous trigger command."""
    from src.common.storage import StorageConfig, close_repositories, create_repositories
    from src.common.storage.postgres.client import get_pool
    from src.evaluation.continuous import EvalScheduler
    from src.evaluation.continuous.config import ContinuousConfig

    try:
        # Load configuration
        config = ContinuousConfig.from_yaml_file(str(config_path))

        # Find the schedule
        schedule = None
        for s in config.schedules:
            if s.name == schedule_name:
                schedule = s
                break

        if not schedule:
            console.print(f"[red]Error:[/red] Schedule not found: {schedule_name}")
            console.print("\nAvailable schedules:")
            for s in config.schedules:
                console.print(f"  • {s.name}")
            raise typer.Exit(1)

        console.print(f"\n[bold]Triggering schedule: {schedule_name}[/bold]")
        console.print(f"  Client: {schedule.client_type}")
        console.print(f"  Eval mode: {schedule.eval_mode}")
        if schedule.test_suite_tags:
            console.print(f"  Tags: {', '.join(schedule.test_suite_tags)}")
        console.print()

        # Create repositories
        storage_config = StorageConfig()
        test_case_repo, scorecard_repo, batch_repo = await create_repositories(storage_config)

        # Get database pool
        try:
            pool = await get_pool()
        except RuntimeError:
            pool = None

        # Create scheduler
        scheduler = EvalScheduler(
            config=config,
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            pool=pool,
        )

        # Trigger the schedule
        batch_id = await scheduler.trigger(schedule_name)

        if batch_id:
            console.print("[green]Evaluation complete![/green]")
            console.print(f"  Batch ID: [cyan]{batch_id}[/cyan]")
        else:
            console.print("[yellow]No test cases found or evaluation failed[/yellow]")

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


@continuous_app.command("alerts")
def continuous_alerts(
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Number of days to look back"),
    ] = 7,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum alerts to show"),
    ] = 50,
) -> None:
    """
    View recent regression alerts.

    Shows unacknowledged alerts from the specified time period.
    """
    asyncio.run(_continuous_alerts_async(days, limit))


async def _continuous_alerts_async(days: int, limit: int) -> None:
    """Async implementation of continuous alerts command."""
    import os

    from src.common.storage.postgres import close_pool, create_pool
    from src.evaluation.continuous import AlertHandler

    try:
        # Create database pool
        database_url = os.environ.get(
            "DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api"
        )
        try:
            pool = await create_pool(database_url)
        except Exception:
            console.print("[yellow]Database not available.[/yellow]")
            console.print("Start PostgreSQL with: [cyan]docker compose up -d[/cyan]")
            raise typer.Exit(1)

        # Create alert handler
        handler = AlertHandler(pool=pool)

        # Get unacknowledged alerts
        alerts = await handler.get_unacknowledged(days=days, limit=limit)

        if not alerts:
            console.print(f"[green]No unacknowledged alerts in the last {days} days[/green]")
            raise typer.Exit(0)

        console.print()
        table = Table(
            title=f"Regression Alerts (last {days} days)", show_header=True, header_style="bold"
        )
        table.add_column("ID", width=15)
        table.add_column("Severity", width=10)
        table.add_column("Metric", width=15)
        table.add_column("Previous", width=10, justify="right")
        table.add_column("Current", width=10, justify="right")
        table.add_column("Change", width=10, justify="right")
        table.add_column("Created", width=20)

        for alert in alerts:
            severity_str = alert.severity.value.upper()
            if alert.severity.value == "critical":
                severity_str = f"[red]{severity_str}[/red]"
            else:
                severity_str = f"[yellow]{severity_str}[/yellow]"

            prev_str = f"{alert.previous_value:.4f}" if alert.previous_value else "-"
            change_str = f"{alert.delta_pct:+.2f}%" if alert.delta_pct else "-"

            table.add_row(
                alert.id[:15] + "...",
                severity_str,
                alert.metric_name,
                prev_str,
                f"{alert.current_value:.4f}",
                change_str,
                alert.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print()
        console.print(f"[dim]Showing {len(alerts)} unacknowledged alerts[/dim]")
        console.print("[dim]Use 'continuous acknowledge <id>' to acknowledge[/dim]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        await close_pool()


@continuous_app.command("acknowledge")
def continuous_acknowledge(
    alert_id: Annotated[
        str,
        typer.Argument(help="Alert ID to acknowledge"),
    ],
    notes: Annotated[
        str | None,
        typer.Option("--notes", "-n", help="Notes about the acknowledgment"),
    ] = None,
    user: Annotated[
        str,
        typer.Option("--user", "-u", help="User acknowledging the alert"),
    ] = "cli",
) -> None:
    """
    Acknowledge a regression alert.

    Marks the alert as acknowledged so it won't appear in future alert listings.
    """
    asyncio.run(_continuous_acknowledge_async(alert_id, notes, user))


async def _continuous_acknowledge_async(
    alert_id: str,
    notes: str | None,
    user: str,
) -> None:
    """Async implementation of continuous acknowledge command."""
    import os

    from src.common.storage.postgres import close_pool, create_pool
    from src.evaluation.continuous import AlertHandler

    try:
        # Create database pool
        database_url = os.environ.get(
            "DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api"
        )
        try:
            pool = await create_pool(database_url)
        except Exception:
            console.print("[yellow]Database not available.[/yellow]")
            raise typer.Exit(1)

        # Create alert handler
        handler = AlertHandler(pool=pool)

        # Acknowledge the alert
        success = await handler.acknowledge(alert_id, user, notes)

        if success:
            console.print(f"[green]Alert acknowledged:[/green] {alert_id}")
            if notes:
                console.print(f"  Notes: {notes}")
        else:
            console.print(f"[red]Failed to acknowledge alert:[/red] {alert_id}")
            console.print("Alert may not exist or already be acknowledged.")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)
    finally:
        await close_pool()
