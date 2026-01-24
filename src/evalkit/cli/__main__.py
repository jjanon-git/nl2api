"""Allow running the CLI as a module: python -m src.evalkit.cli"""

from src.evalkit.cli.main import app

if __name__ == "__main__":
    app()
