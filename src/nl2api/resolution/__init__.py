"""
Entity Resolution

Handles resolution of company names, ticker symbols, and other
entities to standardized identifiers (e.g., RICs).
"""

from src.nl2api.resolution.protocols import EntityResolver
from src.nl2api.resolution.resolver import ExternalEntityResolver

__all__ = [
    "EntityResolver",
    "ExternalEntityResolver",
]
