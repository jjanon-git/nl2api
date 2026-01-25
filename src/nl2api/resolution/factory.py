"""
Entity Resolver Factory

Creates the appropriate entity resolver based on configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.nl2api.resolution.http_client import HttpEntityResolver
from src.nl2api.resolution.protocols import EntityResolver
from src.nl2api.resolution.resolver import ExternalEntityResolver

if TYPE_CHECKING:
    import asyncpg

    from src.evalkit.common.cache import RedisCache
    from src.nl2api.config import NL2APIConfig

logger = logging.getLogger(__name__)


def create_entity_resolver(
    config: NL2APIConfig,
    db_pool: asyncpg.Pool | None = None,
    redis_cache: RedisCache | None = None,
) -> EntityResolver:
    """
    Create an entity resolver based on configuration.

    If `entity_resolution_api_endpoint` is set, creates an HttpEntityResolver
    that calls the standalone service. Otherwise, creates an ExternalEntityResolver
    that uses local database + OpenFIGI fallback.

    Args:
        config: NL2API configuration
        db_pool: Optional database pool (for local resolver)
        redis_cache: Optional Redis cache (for local resolver)

    Returns:
        EntityResolver instance
    """
    if not config.entity_resolution_enabled:
        logger.info("Entity resolution disabled")
        # Return a no-op resolver that won't resolve anything
        from src.nl2api.resolution.mock_resolver import MockEntityResolver

        return MockEntityResolver(additional_mappings={})

    if config.entity_resolution_api_endpoint:
        # Use HTTP client to call the standalone service
        logger.info(
            f"Using HTTP entity resolver: {config.entity_resolution_api_endpoint}"
        )
        return HttpEntityResolver(
            base_url=config.entity_resolution_api_endpoint,
            api_key=config.entity_resolution_api_key,
            timeout_seconds=config.entity_resolution_timeout_seconds,
            circuit_failure_threshold=config.entity_resolution_circuit_failure_threshold,
            circuit_recovery_seconds=config.entity_resolution_circuit_recovery_seconds,
            retry_max_attempts=config.entity_resolution_retry_max_attempts,
        )
    else:
        # Use local resolver with database + OpenFIGI
        logger.info("Using local entity resolver with database + OpenFIGI")
        return ExternalEntityResolver(
            db_pool=db_pool,
            redis_cache=redis_cache,
            use_cache=config.entity_resolution_cache_enabled,
            timeout_seconds=config.entity_resolution_timeout_seconds,
            circuit_failure_threshold=config.entity_resolution_circuit_failure_threshold,
            circuit_recovery_seconds=config.entity_resolution_circuit_recovery_seconds,
            retry_max_attempts=config.entity_resolution_retry_max_attempts,
        )
