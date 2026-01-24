"""
Compatibility shim for src.common -> evalkit.common migration.

TODO: Remove after validation period (Stage 2 of codebase separation).
"""

# Re-export everything from src.evalkit.common for backwards compatibility
from src.evalkit.common import *  # noqa: F401, F403

# Import submodules to make them available as attributes
from src.evalkit.common import (
    cache,  # noqa: F401
    git_info,  # noqa: F401
    logging,  # noqa: F401
    resilience,  # noqa: F401
    storage,  # noqa: F401
    telemetry,  # noqa: F401
)
