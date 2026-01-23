"""
MCP Resource Definitions and Handlers for Entity Resolution

Defines the resources exposed by the Entity Resolution MCP Server:
- entity://stats: Database statistics and cache metrics
- entity://health: Health check for load balancers
- entity://exchanges: Supported exchanges with RIC suffixes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.common.telemetry import get_tracer

if TYPE_CHECKING:
    import asyncpg

    from src.common.cache import RedisCache
    from src.nl2api.resolution.resolver import ExternalEntityResolver

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


# =============================================================================
# Resource Definitions (MCP Schema)
# =============================================================================

RESOURCE_DEFINITIONS = [
    {
        "uri": "entity://stats",
        "name": "Entity Resolution Statistics",
        "description": (
            "Database statistics including entity counts, cache hit rates, "
            "and circuit breaker state. Useful for monitoring and debugging."
        ),
        "mimeType": "application/json",
    },
    {
        "uri": "entity://health",
        "name": "Health Check",
        "description": (
            "Health status of the entity resolution service. "
            "Returns OK if database and cache are accessible."
        ),
        "mimeType": "application/json",
    },
    {
        "uri": "entity://exchanges",
        "name": "Supported Exchanges",
        "description": (
            "List of supported exchanges with their RIC suffixes. "
            "Useful for understanding RIC format and coverage."
        ),
        "mimeType": "application/json",
    },
]


# =============================================================================
# Exchange Reference Data
# =============================================================================

# Common RIC suffixes by exchange
EXCHANGE_RIC_SUFFIXES = {
    "NASDAQ": {
        "suffix": ".O",
        "full_name": "NASDAQ Stock Market",
        "country": "US",
        "examples": ["AAPL.O", "MSFT.O", "GOOGL.O"],
    },
    "NYSE": {
        "suffix": ".N",
        "full_name": "New York Stock Exchange",
        "country": "US",
        "examples": ["IBM.N", "JPM.N", "BA.N"],
    },
    "LSE": {
        "suffix": ".L",
        "full_name": "London Stock Exchange",
        "country": "GB",
        "examples": ["HSBA.L", "BP.L", "SHEL.L"],
    },
    "TSE": {
        "suffix": ".T",
        "full_name": "Tokyo Stock Exchange",
        "country": "JP",
        "examples": ["7203.T", "6758.T", "9984.T"],
    },
    "HKEX": {
        "suffix": ".HK",
        "full_name": "Hong Kong Stock Exchange",
        "country": "HK",
        "examples": ["0700.HK", "9988.HK", "0005.HK"],
    },
    "Euronext Paris": {
        "suffix": ".PA",
        "full_name": "Euronext Paris",
        "country": "FR",
        "examples": ["LVMH.PA", "AIR.PA", "BNP.PA"],
    },
    "Xetra": {
        "suffix": ".DE",
        "full_name": "Deutsche Boerse Xetra",
        "country": "DE",
        "examples": ["SAP.DE", "SIE.DE", "VOW3.DE"],
    },
    "Toronto": {
        "suffix": ".TO",
        "full_name": "Toronto Stock Exchange",
        "country": "CA",
        "examples": ["RY.TO", "TD.TO", "ENB.TO"],
    },
    "ASX": {
        "suffix": ".AX",
        "full_name": "Australian Securities Exchange",
        "country": "AU",
        "examples": ["BHP.AX", "CBA.AX", "CSL.AX"],
    },
    "SIX Swiss": {
        "suffix": ".S",
        "full_name": "SIX Swiss Exchange",
        "country": "CH",
        "examples": ["NESN.S", "NOVN.S", "ROG.S"],
    },
}


# =============================================================================
# Response Models
# =============================================================================


@dataclass
class StatsResponse:
    """Response for entity://stats resource."""

    # Database stats
    total_entities: int = 0
    total_aliases: int = 0
    entities_with_ric: int = 0

    # Cache stats
    l1_cache_size: int = 0
    l2_cache_connected: bool = False
    l2_cache_hit_rate: float = 0.0

    # Circuit breaker
    circuit_state: str = "unknown"
    circuit_failure_count: int = 0
    circuit_total_calls: int = 0

    # Timestamps
    stats_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "database": {
                "total_entities": self.total_entities,
                "total_aliases": self.total_aliases,
                "entities_with_ric": self.entities_with_ric,
            },
            "cache": {
                "l1_cache_size": self.l1_cache_size,
                "l2_cache_connected": self.l2_cache_connected,
                "l2_cache_hit_rate": self.l2_cache_hit_rate,
            },
            "circuit_breaker": {
                "state": self.circuit_state,
                "failure_count": self.circuit_failure_count,
                "total_calls": self.circuit_total_calls,
            },
            "timestamp": self.stats_timestamp,
        }


@dataclass
class HealthResponse:
    """Response for entity://health resource."""

    status: str = "unknown"
    database_connected: bool = False
    cache_connected: bool = False
    circuit_breaker_open: bool = False
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "checks": {
                "database": "ok" if self.database_connected else "fail",
                "cache": "ok" if self.cache_connected else "degraded",
                "circuit_breaker": "open" if self.circuit_breaker_open else "closed",
            },
            "message": self.message,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Resource Handlers
# =============================================================================


class ResourceHandlers:
    """
    Handlers for MCP resource requests.

    Provides access to stats, health, and reference data.
    """

    def __init__(
        self,
        resolver: ExternalEntityResolver,
        db_pool: asyncpg.Pool | None = None,
        redis_cache: RedisCache | None = None,
    ):
        """
        Initialize resource handlers.

        Args:
            resolver: The entity resolver instance
            db_pool: Database connection pool
            redis_cache: Redis cache instance
        """
        self._resolver = resolver
        self._db_pool = db_pool
        self._redis_cache = redis_cache

    async def handle_resource_read(
        self,
        uri: str,
    ) -> dict[str, Any]:
        """
        Handle a resource read request.

        Args:
            uri: Resource URI to read

        Returns:
            Resource content as dictionary

        Raises:
            ValueError: If URI is unknown
        """
        with tracer.start_as_current_span(f"mcp.resource.{uri}") as span:
            span.set_attribute("mcp.resource.uri", uri)

            if uri == "entity://stats":
                result = await self._get_stats()
            elif uri == "entity://health":
                result = await self._get_health()
            elif uri == "entity://exchanges":
                result = self._get_exchanges()
            else:
                raise ValueError(f"Unknown resource URI: {uri}")

            span.set_attribute("mcp.resource.success", True)
            return result

    async def _get_stats(self) -> dict[str, Any]:
        """
        Get entity resolution statistics.

        Returns:
            StatsResponse as dictionary
        """
        with tracer.start_as_current_span("mcp.resource.stats") as span:
            stats = StatsResponse()

            # Database stats
            if self._db_pool:
                try:
                    async with self._db_pool.acquire() as conn:
                        # Get entity counts
                        row = await conn.fetchrow(
                            """
                            SELECT
                                (SELECT COUNT(*) FROM entities) as total_entities,
                                (SELECT COUNT(*) FROM entity_aliases) as total_aliases,
                                (SELECT COUNT(*) FROM entities WHERE ric IS NOT NULL) as entities_with_ric
                            """
                        )
                        if row:
                            stats.total_entities = row["total_entities"]
                            stats.total_aliases = row["total_aliases"]
                            stats.entities_with_ric = row["entities_with_ric"]
                except Exception as e:
                    logger.warning(f"Failed to get database stats: {e}")

            # L1 cache stats (from resolver's internal cache)
            if hasattr(self._resolver, "_cache"):
                stats.l1_cache_size = len(self._resolver._cache)

            # L2 cache stats (Redis)
            if self._redis_cache:
                stats.l2_cache_connected = self._redis_cache.is_connected
                if hasattr(self._redis_cache, "stats"):
                    cache_stats = self._redis_cache.stats
                    stats.l2_cache_hit_rate = cache_stats.hit_rate

            # Circuit breaker stats
            cb_stats = self._resolver.circuit_breaker_stats
            stats.circuit_state = cb_stats.get("state", "unknown")
            stats.circuit_failure_count = cb_stats.get("failure_count", 0)
            stats.circuit_total_calls = cb_stats.get("total_calls", 0)

            span.set_attribute("stats.total_entities", stats.total_entities)
            span.set_attribute("stats.circuit_state", stats.circuit_state)

            return stats.to_dict()

    async def _get_health(self) -> dict[str, Any]:
        """
        Get health status.

        Returns:
            HealthResponse as dictionary
        """
        with tracer.start_as_current_span("mcp.resource.health") as span:
            health = HealthResponse()
            issues = []

            # Check database
            if self._db_pool:
                try:
                    async with self._db_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    health.database_connected = True
                except Exception as e:
                    logger.warning(f"Database health check failed: {e}")
                    issues.append(f"Database: {e}")
            else:
                issues.append("Database: not configured")

            # Check Redis
            if self._redis_cache:
                health.cache_connected = self._redis_cache.is_connected
                if not health.cache_connected:
                    issues.append("Cache: not connected")
            else:
                # Cache is optional, so not connected is okay
                health.cache_connected = False

            # Check circuit breaker
            cb_stats = self._resolver.circuit_breaker_stats
            health.circuit_breaker_open = cb_stats.get("state") == "open"
            if health.circuit_breaker_open:
                issues.append("Circuit breaker: open")

            # Determine overall status
            if health.database_connected and not health.circuit_breaker_open:
                health.status = "healthy"
                health.message = "All systems operational"
            elif health.database_connected:
                health.status = "degraded"
                health.message = "; ".join(issues) if issues else "Degraded"
            else:
                health.status = "unhealthy"
                health.message = "; ".join(issues) if issues else "Unhealthy"

            span.set_attribute("health.status", health.status)
            span.set_attribute("health.database", health.database_connected)

            return health.to_dict()

    def _get_exchanges(self) -> dict[str, Any]:
        """
        Get supported exchanges reference data.

        Returns:
            Exchange data as dictionary
        """
        return {
            "exchanges": EXCHANGE_RIC_SUFFIXES,
            "total_exchanges": len(EXCHANGE_RIC_SUFFIXES),
            "note": ("RIC suffixes indicate the exchange. For example, .O is NASDAQ, .N is NYSE."),
        }
