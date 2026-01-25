"""
Entity Resolution

Handles resolution of company names, ticker symbols, and other
entities to standardized identifiers (e.g., RICs).

Two resolver implementations available:
- ExternalEntityResolver: Uses local database + OpenFIGI fallback
- HttpEntityResolver: Calls the standalone entity resolution HTTP service
"""

from src.nl2api.resolution.factory import create_entity_resolver
from src.nl2api.resolution.http_client import HttpEntityResolver
from src.nl2api.resolution.mock_resolver import MockEntityResolver
from src.nl2api.resolution.protocols import EntityResolver, ResolvedEntity
from src.nl2api.resolution.resolver import ExternalEntityResolver

__all__ = [
    "EntityResolver",
    "ResolvedEntity",
    "ExternalEntityResolver",
    "HttpEntityResolver",
    "MockEntityResolver",
    "create_entity_resolver",
]
