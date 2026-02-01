"""
Entity Resolution Models - Re-export from canonical location.

The canonical definitions live in src/evalkit/common/entity_resolution/models.py.
This module re-exports them for backwards compatibility.
"""

from src.evalkit.common.entity_resolution import EntityType, ResolvedEntity

__all__ = ["EntityType", "ResolvedEntity"]
