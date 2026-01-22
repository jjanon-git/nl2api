"""
EvalPlatform CLI

Entry point for the evaluation command-line interface.

Usage:
    python -m src.evaluation.cli.main run tests/fixtures/search_products.json
    python -m src.evaluation.cli.main batch run --limit 10
    python -m src.evaluation.cli.main --help
"""

import atexit

import typer

from src.common.telemetry import init_telemetry, shutdown_telemetry
from src.evaluation.cli.commands.batch import batch_app
from src.evaluation.cli.commands.continuous import continuous_app
from src.evaluation.cli.commands.matrix import matrix_app
from src.evaluation.cli.commands.run import run_command

# Initialize OpenTelemetry at CLI startup
init_telemetry(service_name="nl2api-evaluation")
atexit.register(shutdown_telemetry)

app = typer.Typer(
    name="eval",
    help="EvalPlatform - Distributed evaluation framework for LLM tool-calling",
    no_args_is_help=True,
)

# Register commands
app.command(name="run", help="Run evaluation on a test case file")(run_command)

# Register subcommand groups
app.add_typer(batch_app, name="batch", help="Batch evaluation commands")
app.add_typer(continuous_app, name="continuous", help="Continuous evaluation commands")
app.add_typer(matrix_app, name="matrix", help="Multi-dimensional evaluation (components × LLMs)")


@app.command()
def version() -> None:
    """Show version information."""
    from src import __version__
    typer.echo(f"eval-platform version {__version__}")


if __name__ == "__main__":
    app()
