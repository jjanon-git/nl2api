"""
Compatibility shim for src.evaluation.batch -> src.evalkit.batch migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export everything from src.evalkit.batch for backwards compatibility
from src.evalkit.batch import *  # noqa: F401, F403
from src.evalkit.batch import BatchRunner, BatchRunnerConfig  # noqa: F401
