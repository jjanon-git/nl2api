"""
Compatibility shim for src.evaluation.core -> src.evalkit.core migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export everything from src.evalkit.core for backwards compatibility
from src.evalkit.core import *  # noqa: F401, F403
