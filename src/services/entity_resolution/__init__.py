"""
Entity Resolution Service

Standalone service for resolving entity names to financial identifiers (RICs, tickers).
Supports HTTP/REST, MCP protocol, and stdio transports.

Usage:
    # As a service
    python -m src.services.entity_resolution --port 8085

    # Programmatic
    from src.services.entity_resolution import EntityResolver, EntityExtractor
"""

__version__ = "0.1.0"

from .config import EntityServiceConfig
from .core.extractor import EntityExtractor
from .core.models import EntityType, ResolvedEntity
from .core.resolver import EntityResolver

__all__ = [
    "EntityServiceConfig",
    "EntityExtractor",
    "EntityResolver",
    "EntityType",
    "ResolvedEntity",
]
