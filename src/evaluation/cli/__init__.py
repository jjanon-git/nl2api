"""
Compatibility shim for src.evaluation.cli -> src.evalkit.cli migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export everything from src.evalkit.cli for backwards compatibility
from src.evalkit.cli import *  # noqa: F401, F403
