"""
EvalPlatform CLI

Entry point for the evaluation command-line interface.

Usage:
    python -m src.cli.main run tests/fixtures/search_products.json
    python -m src.cli.main --help
"""

import typer

from src.cli.commands.run import run_command

app = typer.Typer(
    name="eval",
    help="EvalPlatform - Distributed evaluation framework for LLM tool-calling",
    no_args_is_help=True,
)

# Register commands
app.command(name="run", help="Run evaluation on a test case file")(run_command)


@app.command()
def version() -> None:
    """Show version information."""
    from src import __version__
    typer.echo(f"eval-platform version {__version__}")


if __name__ == "__main__":
    app()
