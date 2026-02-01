"""
Entity Resolution - Shared Core Module

Provides canonical models and shared logic for entity resolution.
Used by both the standalone entity resolution service and the
embedded resolver in NL2API.

Usage:
    from src.evalkit.common.entity_resolution import (
        ResolvedEntity,
        resolve_via_database,
        resolve_via_openfigi,
        normalize_entity,
    )
"""

from .database import normalize_entity, resolve_via_database
from .models import EntityType, ResolvedEntity
from .openfigi import resolve_via_openfigi

__all__ = [
    # Models
    "ResolvedEntity",
    "EntityType",
    # Database operations
    "resolve_via_database",
    "normalize_entity",
    # External API
    "resolve_via_openfigi",
]
