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

# Note: ExternalEntityResolver is intentionally not exported.
# Use create_entity_resolver(config) instead, which returns the appropriate
# resolver based on configuration.

__all__ = [
    # Public API
    "EntityResolver",  # Protocol
    "ResolvedEntity",  # Data class
    "create_entity_resolver",  # Factory (preferred way to get a resolver)
    # Implementations (use factory instead of instantiating directly)
    "HttpEntityResolver",
    "MockEntityResolver",
]
