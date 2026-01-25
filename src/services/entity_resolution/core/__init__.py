"""
Core entity resolution logic.

This module contains the domain logic for entity resolution,
independent of any transport or framework.
"""

from .models import EntityType, ResolvedEntity
from .protocols import EntityResolver

__all__ = ["EntityResolver", "EntityType", "ResolvedEntity"]
