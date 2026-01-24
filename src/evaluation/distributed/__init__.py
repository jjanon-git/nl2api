"""
Compatibility shim for src.evaluation.distributed -> src.evalkit.distributed migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export everything from src.evalkit.distributed for backwards compatibility
from src.evalkit.distributed import *  # noqa: F401, F403
