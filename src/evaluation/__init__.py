"""
Compatibility shim for src.evaluation -> evalkit migration.

TODO: Remove after validation period (Stage 2 of codebase separation).

The evaluation module has been reorganized into evalkit:
- src.evaluation.core -> evalkit.core
- src.evaluation.batch -> evalkit.batch
- src.evaluation.cli -> evalkit.cli
- src.evaluation.distributed -> evalkit.distributed
- src.evaluation.continuous -> evalkit.continuous

Packs remain in src.evaluation.packs for now (will move in Phase 3).
"""

import importlib
from typing import Any

# Lazy loading to avoid circular imports with evalkit.batch
_SUBMODULES = ["batch", "cli", "continuous", "core", "distributed", "packs"]


def __getattr__(name: str) -> Any:
    """Lazy load submodules to avoid circular imports."""
    if name in _SUBMODULES:
        return importlib.import_module(f"src.evaluation.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
