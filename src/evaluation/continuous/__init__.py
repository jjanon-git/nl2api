"""
Compatibility shim for src.evaluation.continuous -> src.evalkit.continuous migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export everything from src.evalkit.continuous for backwards compatibility
from src.evalkit.continuous import *  # noqa: F401, F403
