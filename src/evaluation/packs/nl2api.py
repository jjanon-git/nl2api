"""
Compatibility shim for src.evaluation.packs.nl2api -> src.nl2api.evaluation.pack.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export from new location
from src.nl2api.evaluation.pack import *  # noqa: F401, F403
from src.nl2api.evaluation.pack import NL2APIPack  # noqa: F401
