"""
Compatibility shim for src.contracts -> evalkit.contracts migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export everything from src.evalkit.contracts for backwards compatibility
from src.evalkit.contracts import *  # noqa: F401, F403
from src.evalkit.contracts import __all__  # noqa: F401
