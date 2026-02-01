"""
Entity Resolution Database Operations

Shared database lookup logic for entity resolution.
Used by both standalone service and embedded resolver.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .models import ResolvedEntity

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


def normalize_entity(entity: str) -> str:
    """
    Normalize entity name for lookup and caching.

    Lowercases, strips whitespace, and removes common company suffixes.

    Args:
        entity: Raw entity name

    Returns:
        Normalized entity name
    """
    normalized = entity.lower().strip()
    # Strip common company suffixes: Inc, Corp, Ltd, LLC, PLC, & Co, & Company
    normalized = re.sub(
        r"\s*(&\s*(co\.?|company))?\s*(inc\.?|corp\.?|ltd\.?|llc|plc)?\.?$",
        "",
        normalized,
        flags=re.I,
    )
    return normalized.strip()


async def resolve_via_database(
    db_pool: asyncpg.Pool,
    entity: str,
    normalized: str | None = None,
) -> ResolvedEntity | None:
    """
    Resolve entity using database entity_aliases table.

    Tries multiple lookup strategies in order:
    1. Exact alias match (case-insensitive)
    2. Original entity as alias (preserves case for tickers)
    3. Direct primary_name lookup
    4. Direct ticker lookup
    5. Fuzzy match using pg_trgm (for queries >= 4 chars)

    Args:
        db_pool: asyncpg connection pool
        entity: Original entity string
        normalized: Normalized entity (computed if not provided)

    Returns:
        ResolvedEntity if found, None otherwise
    """
    if normalized is None:
        normalized = normalize_entity(entity)

    try:
        async with db_pool.acquire() as conn:
            row = await _try_alias_lookup(conn, normalized)

            if not row:
                row = await _try_alias_lookup(conn, entity)

            if not row:
                row = await _try_primary_name_lookup(conn, entity)

            if not row:
                row = await _try_ticker_lookup(conn, entity)

            if not row and len(entity) >= 4:
                row = await _try_fuzzy_lookup(conn, entity)

            if row:
                return _row_to_resolved_entity(entity, row)

    except Exception as e:
        logger.warning(f"Database lookup failed for '{entity}': {e}")

    return None


async def _try_alias_lookup(conn: Any, search_term: str) -> Any | None:
    """Try to find entity via alias table."""
    return await conn.fetchrow(
        """
        SELECT e.primary_name, e.ticker, e.ric, e.entity_type, a.alias_type
        FROM entity_aliases a
        JOIN entities e ON a.entity_id = e.id
        WHERE a.alias ILIKE $1 AND e.ric IS NOT NULL
        LIMIT 1
        """,
        search_term,
    )


async def _try_primary_name_lookup(conn: Any, entity: str) -> Any | None:
    """Try direct primary_name lookup."""
    return await conn.fetchrow(
        """
        SELECT primary_name, ticker, ric, entity_type,
               'primary_name' as alias_type
        FROM entities
        WHERE primary_name ILIKE $1 AND ric IS NOT NULL
        LIMIT 1
        """,
        entity,
    )


async def _try_ticker_lookup(conn: Any, entity: str) -> Any | None:
    """Try direct ticker lookup."""
    return await conn.fetchrow(
        """
        SELECT primary_name, ticker, ric, entity_type,
               'ticker_direct' as alias_type
        FROM entities
        WHERE ticker ILIKE $1 AND ric IS NOT NULL
        LIMIT 1
        """,
        entity,
    )


async def _try_fuzzy_lookup(conn: Any, entity: str) -> Any | None:
    """Try fuzzy match using pg_trgm trigram similarity."""
    row = await conn.fetchrow(
        """
        SELECT primary_name, ticker, ric, entity_type,
               'fuzzy' as alias_type,
               similarity(primary_name, $1) as sim_score
        FROM entities
        WHERE ric IS NOT NULL AND similarity(primary_name, $1) > 0.3
        ORDER BY similarity(primary_name, $1) DESC
        LIMIT 1
        """,
        entity,
    )
    if row:
        logger.debug(
            f"Fuzzy match: '{entity}' -> '{row['primary_name']}' (score={row['sim_score']:.2f})"
        )
    return row


def _row_to_resolved_entity(entity: str, row: Any) -> ResolvedEntity:
    """Convert database row to ResolvedEntity."""
    alias_type = row["alias_type"]

    if alias_type in ("ticker", "ticker_direct"):
        entity_type = "ticker"
        confidence = 0.99
    elif alias_type == "legal_name":
        entity_type = "company"
        confidence = 0.98
    elif alias_type == "fuzzy":
        entity_type = row["entity_type"] or "company"
        confidence = row.get("sim_score", 0.7)
    else:
        entity_type = row["entity_type"] or "company"
        confidence = 0.95

    return ResolvedEntity(
        original=entity,
        identifier=row["ric"],
        entity_type=entity_type,
        confidence=confidence,
        metadata={
            "ticker": row["ticker"] or "",
            "company_name": row["primary_name"] or "",
        },
    )
