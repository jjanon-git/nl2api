"""
Entity Resolution

Handles resolution of company names, ticker symbols, and other
entities to standardized identifiers (e.g., RICs).
"""

from src.nl2api.resolution.protocols import EntityResolver, ResolvedEntity
from src.nl2api.resolution.resolver import ExternalEntityResolver
from src.nl2api.resolution.mock_resolver import MockEntityResolver

__all__ = [
    "EntityResolver",
    "ResolvedEntity",
    "ExternalEntityResolver",
    "MockEntityResolver",
]
