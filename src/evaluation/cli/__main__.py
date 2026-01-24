"""
Compatibility shim for src.evaluation.cli -> evalkit.cli migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

from src.evalkit.cli.main import app

if __name__ == "__main__":
    app()
