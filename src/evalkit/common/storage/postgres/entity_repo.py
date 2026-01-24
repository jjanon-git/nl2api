"""
PostgreSQL Entity Repository

Implements data access for entity resolution storage.
Provides CRUD operations and resolution queries against the entities table.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from src.common.telemetry import get_tracer

tracer = get_tracer(__name__)


@dataclass
class Entity:
    """
    Entity data model for storage.

    Represents a company, fund, or other legal entity with identifiers.
    """

    id: str
    primary_name: str
    data_source: str

    # Optional identifiers
    lei: str | None = None
    cik: str | None = None
    permid: str | None = None
    figi: str | None = None

    # Display and stock identifiers
    display_name: str | None = None
    ticker: str | None = None
    ric: str | None = None
    exchange: str | None = None

    # Classification
    entity_type: str = "company"
    entity_status: str = "active"
    is_public: bool = False

    # Geographic
    country_code: str | None = None
    region: str | None = None
    city: str | None = None

    # Industry
    sic_code: str | None = None
    naics_code: str | None = None
    gics_sector: str | None = None

    # Hierarchy
    parent_entity_id: str | None = None
    ultimate_parent_id: str | None = None

    # Data quality
    confidence_score: float = 1.0
    ric_validated: bool = False
    last_verified_at: datetime | None = None

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class EntityAlias:
    """Entity alias for fuzzy matching."""

    id: str
    entity_id: str
    alias: str
    alias_type: str
    is_primary: bool = False
    created_at: datetime | None = None


@dataclass
class EntityMatch:
    """Result from entity resolution query."""

    entity_id: str
    primary_name: str
    display_name: str | None
    ric: str | None
    ticker: str | None
    exchange: str | None
    match_type: str  # exact, ticker, ric, fuzzy
    similarity: float


@dataclass
class EntityStats:
    """Entity statistics for monitoring."""

    total_entities: int
    public_entities: int
    private_entities: int
    entities_with_ric: int
    entities_with_validated_ric: int
    countries: int
    exchanges: int
    data_sources: int


class PostgresEntityRepository:
    """
    PostgreSQL implementation of entity storage.

    Provides data access for entity resolution including:
    - CRUD operations for entities and aliases
    - Resolution queries using PostgreSQL functions
    - Bulk import via COPY protocol
    - Statistics and monitoring views
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository with connection pool.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    # =========================================================================
    # Entity CRUD Operations
    # =========================================================================

    async def get(self, entity_id: str) -> Entity | None:
        """Fetch entity by ID."""
        try:
            eid = uuid.UUID(entity_id)
        except ValueError:
            return None

        row = await self.pool.fetchrow(
            "SELECT * FROM entities WHERE id = $1",
            eid,
        )
        return self._row_to_entity(row) if row else None

    async def get_by_lei(self, lei: str) -> Entity | None:
        """Fetch entity by LEI (Legal Entity Identifier)."""
        row = await self.pool.fetchrow(
            "SELECT * FROM entities WHERE lei = $1",
            lei.upper(),
        )
        return self._row_to_entity(row) if row else None

    async def get_by_cik(self, cik: str) -> Entity | None:
        """Fetch entity by CIK (SEC Central Index Key)."""
        # Normalize CIK to 10 digits with leading zeros
        normalized_cik = cik.zfill(10)
        row = await self.pool.fetchrow(
            "SELECT * FROM entities WHERE cik = $1",
            normalized_cik,
        )
        return self._row_to_entity(row) if row else None

    async def get_by_ticker(self, ticker: str) -> Entity | None:
        """Fetch entity by ticker symbol."""
        row = await self.pool.fetchrow(
            "SELECT * FROM entities WHERE upper(ticker) = $1 AND entity_status = 'active'",
            ticker.upper(),
        )
        return self._row_to_entity(row) if row else None

    async def get_by_ric(self, ric: str) -> Entity | None:
        """Fetch entity by RIC (Reuters Instrument Code)."""
        row = await self.pool.fetchrow(
            "SELECT * FROM entities WHERE ric = $1 AND entity_status = 'active'",
            ric.upper(),
        )
        return self._row_to_entity(row) if row else None

    async def save(self, entity: Entity) -> str:
        """
        Save entity (insert or update).

        Returns:
            Entity ID
        """
        eid = uuid.UUID(entity.id) if entity.id else uuid.uuid4()

        parent_id = uuid.UUID(entity.parent_entity_id) if entity.parent_entity_id else None
        ultimate_id = uuid.UUID(entity.ultimate_parent_id) if entity.ultimate_parent_id else None

        await self.pool.execute(
            """
            INSERT INTO entities (
                id, lei, cik, permid, figi,
                primary_name, display_name,
                ticker, ric, exchange,
                entity_type, entity_status, is_public,
                country_code, region, city,
                sic_code, naics_code, gics_sector,
                parent_entity_id, ultimate_parent_id,
                data_source, confidence_score, ric_validated, last_verified_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7,
                $8, $9, $10,
                $11, $12, $13,
                $14, $15, $16,
                $17, $18, $19,
                $20, $21,
                $22, $23, $24, $25
            )
            ON CONFLICT (id) DO UPDATE SET
                lei = COALESCE(EXCLUDED.lei, entities.lei),
                cik = COALESCE(EXCLUDED.cik, entities.cik),
                permid = COALESCE(EXCLUDED.permid, entities.permid),
                figi = COALESCE(EXCLUDED.figi, entities.figi),
                primary_name = EXCLUDED.primary_name,
                display_name = COALESCE(EXCLUDED.display_name, entities.display_name),
                ticker = COALESCE(EXCLUDED.ticker, entities.ticker),
                ric = COALESCE(EXCLUDED.ric, entities.ric),
                exchange = COALESCE(EXCLUDED.exchange, entities.exchange),
                entity_type = EXCLUDED.entity_type,
                entity_status = EXCLUDED.entity_status,
                is_public = EXCLUDED.is_public,
                country_code = COALESCE(EXCLUDED.country_code, entities.country_code),
                region = COALESCE(EXCLUDED.region, entities.region),
                city = COALESCE(EXCLUDED.city, entities.city),
                sic_code = COALESCE(EXCLUDED.sic_code, entities.sic_code),
                naics_code = COALESCE(EXCLUDED.naics_code, entities.naics_code),
                gics_sector = COALESCE(EXCLUDED.gics_sector, entities.gics_sector),
                parent_entity_id = COALESCE(EXCLUDED.parent_entity_id, entities.parent_entity_id),
                ultimate_parent_id = COALESCE(EXCLUDED.ultimate_parent_id, entities.ultimate_parent_id),
                confidence_score = EXCLUDED.confidence_score,
                ric_validated = EXCLUDED.ric_validated,
                last_verified_at = EXCLUDED.last_verified_at,
                updated_at = NOW()
            """,
            eid,
            entity.lei,
            entity.cik,
            entity.permid,
            entity.figi,
            entity.primary_name,
            entity.display_name,
            entity.ticker,
            entity.ric,
            entity.exchange,
            entity.entity_type,
            entity.entity_status,
            entity.is_public,
            entity.country_code,
            entity.region,
            entity.city,
            entity.sic_code,
            entity.naics_code,
            entity.gics_sector,
            parent_id,
            ultimate_id,
            entity.data_source,
            entity.confidence_score,
            entity.ric_validated,
            entity.last_verified_at,
        )

        return str(eid)

    async def save_batch(self, entities: list[Entity]) -> int:
        """
        Save multiple entities in a single transaction.

        Uses executemany for efficiency.

        Args:
            entities: List of entities to save

        Returns:
            Number of entities saved
        """
        if not entities:
            return 0

        records = []
        for entity in entities:
            eid = uuid.UUID(entity.id) if entity.id else uuid.uuid4()
            parent_id = uuid.UUID(entity.parent_entity_id) if entity.parent_entity_id else None
            ultimate_id = (
                uuid.UUID(entity.ultimate_parent_id) if entity.ultimate_parent_id else None
            )

            records.append(
                (
                    eid,
                    entity.lei,
                    entity.cik,
                    entity.permid,
                    entity.figi,
                    entity.primary_name,
                    entity.display_name,
                    entity.ticker,
                    entity.ric,
                    entity.exchange,
                    entity.entity_type,
                    entity.entity_status,
                    entity.is_public,
                    entity.country_code,
                    entity.region,
                    entity.city,
                    entity.sic_code,
                    entity.naics_code,
                    entity.gics_sector,
                    parent_id,
                    ultimate_id,
                    entity.data_source,
                    entity.confidence_score,
                    entity.ric_validated,
                    entity.last_verified_at,
                )
            )

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO entities (
                        id, lei, cik, permid, figi,
                        primary_name, display_name,
                        ticker, ric, exchange,
                        entity_type, entity_status, is_public,
                        country_code, region, city,
                        sic_code, naics_code, gics_sector,
                        parent_entity_id, ultimate_parent_id,
                        data_source, confidence_score, ric_validated, last_verified_at
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7,
                        $8, $9, $10,
                        $11, $12, $13,
                        $14, $15, $16,
                        $17, $18, $19,
                        $20, $21,
                        $22, $23, $24, $25
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        lei = COALESCE(EXCLUDED.lei, entities.lei),
                        cik = COALESCE(EXCLUDED.cik, entities.cik),
                        permid = COALESCE(EXCLUDED.permid, entities.permid),
                        figi = COALESCE(EXCLUDED.figi, entities.figi),
                        primary_name = EXCLUDED.primary_name,
                        display_name = COALESCE(EXCLUDED.display_name, entities.display_name),
                        ticker = COALESCE(EXCLUDED.ticker, entities.ticker),
                        ric = COALESCE(EXCLUDED.ric, entities.ric),
                        exchange = COALESCE(EXCLUDED.exchange, entities.exchange),
                        entity_type = EXCLUDED.entity_type,
                        entity_status = EXCLUDED.entity_status,
                        is_public = EXCLUDED.is_public,
                        country_code = COALESCE(EXCLUDED.country_code, entities.country_code),
                        sic_code = COALESCE(EXCLUDED.sic_code, entities.sic_code),
                        data_source = EXCLUDED.data_source,
                        updated_at = NOW()
                    """,
                    records,
                )

        return len(records)

    async def delete(self, entity_id: str) -> bool:
        """Delete entity. Returns True if deleted."""
        try:
            eid = uuid.UUID(entity_id)
        except ValueError:
            return False

        result = await self.pool.execute(
            "DELETE FROM entities WHERE id = $1",
            eid,
        )
        return result == "DELETE 1"

    # =========================================================================
    # Resolution Queries
    # =========================================================================

    async def resolve(
        self,
        query: str,
        fuzzy_threshold: float = 0.3,
        limit: int = 5,
    ) -> list[EntityMatch]:
        """
        Resolve entity query using database function.

        Resolution order:
        1. Exact alias match
        2. Ticker match
        3. RIC match
        4. Fuzzy match (trigram similarity)

        Args:
            query: Entity name or identifier to resolve
            fuzzy_threshold: Minimum similarity for fuzzy matches (0-1)
            limit: Maximum results to return

        Returns:
            List of EntityMatch ordered by match quality
        """
        with tracer.start_as_current_span("db.entity.resolve") as span:
            span.set_attribute("query", query[:100])  # Truncate for safety
            span.set_attribute("fuzzy_threshold", fuzzy_threshold)
            span.set_attribute("limit", limit)

            rows = await self.pool.fetch(
                "SELECT * FROM resolve_entity($1, $2, $3)",
                query,
                fuzzy_threshold,
                limit,
            )

            span.set_attribute("result_count", len(rows))
            if rows:
                span.set_attribute("best_match_type", rows[0]["match_type"])
                span.set_attribute("best_similarity", rows[0]["similarity"])

            return [
                EntityMatch(
                    entity_id=str(row["entity_id"]),
                    primary_name=row["primary_name"],
                    display_name=row["display_name"],
                    ric=row["ric"],
                    ticker=row["ticker"],
                    exchange=row["exchange"],
                    match_type=row["match_type"],
                    similarity=row["similarity"],
                )
                for row in rows
            ]

    async def resolve_batch(
        self,
        queries: list[str],
        fuzzy_threshold: float = 0.3,
    ) -> dict[str, EntityMatch | None]:
        """
        Resolve multiple entities in batch.

        Args:
            queries: List of entity names/identifiers
            fuzzy_threshold: Minimum similarity for fuzzy matches

        Returns:
            Dict mapping query -> EntityMatch (or None if not found)
        """
        with tracer.start_as_current_span("db.entity.resolve_batch") as span:
            span.set_attribute("query_count", len(queries))
            span.set_attribute("fuzzy_threshold", fuzzy_threshold)

            if not queries:
                span.set_attribute("resolved_count", 0)
                return {}

            rows = await self.pool.fetch(
                "SELECT * FROM resolve_entities_batch($1, $2)",
                queries,
                fuzzy_threshold,
            )

            results: dict[str, EntityMatch | None] = {q: None for q in queries}
            for row in rows:
                results[row["query"]] = EntityMatch(
                    entity_id=str(row["entity_id"]),
                    primary_name=row["primary_name"],
                    display_name=None,  # Not returned by batch function
                    ric=row["ric"],
                    ticker=row["ticker"],
                    exchange=None,
                    match_type=row["match_type"],
                    similarity=row["similarity"],
                )

            resolved_count = sum(1 for v in results.values() if v is not None)
            span.set_attribute("resolved_count", resolved_count)
            span.set_attribute("resolution_rate", resolved_count / len(queries) if queries else 0)

            return results

    # =========================================================================
    # Alias Operations
    # =========================================================================

    async def add_alias(
        self,
        entity_id: str,
        alias: str,
        alias_type: str = "generated",
        is_primary: bool = False,
    ) -> str | None:
        """
        Add alias for entity.

        Args:
            entity_id: Entity UUID
            alias: Alias text (will be normalized to lowercase)
            alias_type: Type of alias (ticker, legal_name, trade_name, abbreviation, generated)
            is_primary: Whether this is the primary alias

        Returns:
            Alias ID if created, None if duplicate
        """
        try:
            eid = uuid.UUID(entity_id)
        except ValueError:
            return None

        aid = uuid.uuid4()
        normalized_alias = alias.lower().strip()

        try:
            await self.pool.execute(
                """
                INSERT INTO entity_aliases (id, entity_id, alias, alias_type, is_primary)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (entity_id, alias) DO NOTHING
                """,
                aid,
                eid,
                normalized_alias,
                alias_type,
                is_primary,
            )
            return str(aid)
        except asyncpg.ForeignKeyViolationError:
            return None

    async def add_aliases_batch(
        self,
        aliases: list[tuple[str, str, str]],  # (entity_id, alias, alias_type)
    ) -> int:
        """
        Add multiple aliases in batch.

        Args:
            aliases: List of (entity_id, alias, alias_type) tuples

        Returns:
            Number of aliases added
        """
        if not aliases:
            return 0

        records = []
        for entity_id, alias, alias_type in aliases:
            try:
                eid = uuid.UUID(entity_id)
                aid = uuid.uuid4()
                normalized_alias = alias.lower().strip()
                records.append((aid, eid, normalized_alias, alias_type, False))
            except ValueError:
                continue

        if not records:
            return 0

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO entity_aliases (id, entity_id, alias, alias_type, is_primary)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (entity_id, alias) DO NOTHING
                    """,
                    records,
                )

        return len(records)

    async def get_aliases(self, entity_id: str) -> list[EntityAlias]:
        """Get all aliases for an entity."""
        try:
            eid = uuid.UUID(entity_id)
        except ValueError:
            return []

        rows = await self.pool.fetch(
            "SELECT * FROM entity_aliases WHERE entity_id = $1 ORDER BY is_primary DESC, alias",
            eid,
        )

        return [
            EntityAlias(
                id=str(row["id"]),
                entity_id=str(row["entity_id"]),
                alias=row["alias"],
                alias_type=row["alias_type"],
                is_primary=row["is_primary"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def delete_aliases(self, entity_id: str) -> int:
        """Delete all aliases for an entity. Returns count deleted."""
        try:
            eid = uuid.UUID(entity_id)
        except ValueError:
            return 0

        result = await self.pool.execute(
            "DELETE FROM entity_aliases WHERE entity_id = $1",
            eid,
        )
        # Result is like "DELETE 5"
        return int(result.split()[-1])

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    async def get_stats(self) -> EntityStats:
        """Get entity statistics."""
        row = await self.pool.fetchrow("SELECT * FROM entity_stats")

        return EntityStats(
            total_entities=row["total_entities"] or 0,
            public_entities=row["public_entities"] or 0,
            private_entities=row["private_entities"] or 0,
            entities_with_ric=row["entities_with_ric"] or 0,
            entities_with_validated_ric=row["entities_with_validated_ric"] or 0,
            countries=row["countries"] or 0,
            exchanges=row["exchanges"] or 0,
            data_sources=row["data_sources"] or 0,
        )

    async def get_coverage_by_source(self) -> list[dict[str, Any]]:
        """Get entity coverage breakdown by data source."""
        rows = await self.pool.fetch("SELECT * FROM entity_coverage_by_source")

        return [
            {
                "data_source": row["data_source"],
                "entity_count": row["entity_count"],
                "with_ric": row["with_ric"],
                "ric_validated": row["ric_validated"],
                "ric_coverage_pct": float(row["ric_coverage_pct"] or 0),
            }
            for row in rows
        ]

    async def count(self, data_source: str | None = None) -> int:
        """Count entities, optionally filtered by data source."""
        if data_source:
            result = await self.pool.fetchval(
                "SELECT COUNT(*) FROM entities WHERE data_source = $1 AND entity_status = 'active'",
                data_source,
            )
        else:
            result = await self.pool.fetchval(
                "SELECT COUNT(*) FROM entities WHERE entity_status = 'active'"
            )
        return result or 0

    async def count_aliases(self) -> int:
        """Count total aliases."""
        result = await self.pool.fetchval("SELECT COUNT(*) FROM entity_aliases")
        return result or 0

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        country_code: str | None = None,
        exchange: str | None = None,
        is_public: bool | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """
        Full-text search for entities.

        Args:
            query: Search query
            country_code: Filter by country
            exchange: Filter by exchange
            is_public: Filter by public/private status
            limit: Maximum results

        Returns:
            List of matching entities
        """
        with tracer.start_as_current_span("db.entity.search") as span:
            span.set_attribute("query", query[:100])
            span.set_attribute("limit", limit)
            if country_code:
                span.set_attribute("country_code", country_code)
            if exchange:
                span.set_attribute("exchange", exchange)

            sql, params = self._build_search_query(
                query=query,
                country_code=country_code,
                exchange=exchange,
                is_public=is_public,
                limit=limit,
            )
            rows = await self.pool.fetch(sql, *params)

            span.set_attribute("result_count", len(rows))
            return [self._row_to_entity(row) for row in rows]

    def _build_search_query(
        self,
        query: str,
        country_code: str | None = None,
        exchange: str | None = None,
        is_public: bool | None = None,
        limit: int = 20,
    ) -> tuple[str, list[Any]]:
        """
        Build a parameterized search query with filters.

        Uses a whitelist approach for conditions to prevent SQL injection.
        All dynamic values are passed as parameters, never interpolated.

        Returns:
            Tuple of (query_string, params_list)
        """
        # Whitelist of allowed condition templates
        CONDITION_TEMPLATES = {
            "full_text": """(to_tsvector('english', primary_name) @@ plainto_tsquery('english', ${idx})
             OR to_tsvector('english', COALESCE(display_name, '')) @@ plainto_tsquery('english', ${idx}))""",
            "country_code": "country_code = ${idx}",
            "exchange": "exchange = ${idx}",
            "is_public": "is_public = ${idx}",
        }

        conditions = ["entity_status = 'active'"]
        params: list[Any] = []
        param_idx = 1

        # Full-text search condition (always included)
        template = CONDITION_TEMPLATES["full_text"]
        conditions.append(template.replace("${idx}", f"${param_idx}"))
        params.append(query)
        param_idx += 1

        if country_code is not None:
            template = CONDITION_TEMPLATES["country_code"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(country_code.upper())
            param_idx += 1

        if exchange is not None:
            template = CONDITION_TEMPLATES["exchange"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(exchange.upper())
            param_idx += 1

        if is_public is not None:
            template = CONDITION_TEMPLATES["is_public"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(is_public)
            param_idx += 1

        params.append(limit)

        # Build final query with static structure
        sql = f"""
            SELECT * FROM entities
            WHERE {" AND ".join(conditions)}
            ORDER BY
                ts_rank(to_tsvector('english', primary_name), plainto_tsquery('english', $1)) DESC,
                primary_name
            LIMIT ${param_idx}
        """

        return sql, params

    # =========================================================================
    # Bulk Import (COPY protocol)
    # =========================================================================

    async def bulk_import(
        self,
        records: list[tuple[Any, ...]],
        columns: list[str],
    ) -> int:
        """
        Bulk import entities using PostgreSQL COPY protocol.

        This is the fastest way to import large datasets.

        Args:
            records: List of tuples matching column order
            columns: Column names in order

        Returns:
            Number of records imported
        """
        with tracer.start_as_current_span("db.entity.bulk_import") as span:
            span.set_attribute("record_count", len(records))
            span.set_attribute("column_count", len(columns))

            if not records:
                span.set_attribute("imported_count", 0)
                return 0

            async with self.pool.acquire() as conn:
                result = await conn.copy_records_to_table(
                    "entities",
                    records=records,
                    columns=columns,
                )

            # Result is like "COPY 1000"
            imported = int(result.split()[-1])
            span.set_attribute("imported_count", imported)
            return imported

    async def bulk_import_aliases(
        self,
        records: list[tuple[Any, ...]],
    ) -> int:
        """
        Bulk import aliases using COPY protocol.

        Args:
            records: List of (id, entity_id, alias, alias_type, is_primary) tuples

        Returns:
            Number of records imported
        """
        with tracer.start_as_current_span("db.entity.bulk_import_aliases") as span:
            span.set_attribute("record_count", len(records))

            if not records:
                span.set_attribute("imported_count", 0)
                return 0

            async with self.pool.acquire() as conn:
                result = await conn.copy_records_to_table(
                    "entity_aliases",
                    records=records,
                    columns=["id", "entity_id", "alias", "alias_type", "is_primary"],
                )

            imported = int(result.split()[-1])
            span.set_attribute("imported_count", imported)
            return imported

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_entity(self, row: asyncpg.Record) -> Entity:
        """Convert database row to Entity model."""
        return Entity(
            id=str(row["id"]),
            primary_name=row["primary_name"],
            data_source=row["data_source"],
            lei=row["lei"],
            cik=row["cik"],
            permid=row["permid"],
            figi=row["figi"],
            display_name=row["display_name"],
            ticker=row["ticker"],
            ric=row["ric"],
            exchange=row["exchange"],
            entity_type=row["entity_type"],
            entity_status=row["entity_status"],
            is_public=row["is_public"],
            country_code=row["country_code"],
            region=row["region"],
            city=row["city"],
            sic_code=row["sic_code"],
            naics_code=row["naics_code"],
            gics_sector=row["gics_sector"],
            parent_entity_id=str(row["parent_entity_id"]) if row["parent_entity_id"] else None,
            ultimate_parent_id=str(row["ultimate_parent_id"])
            if row["ultimate_parent_id"]
            else None,
            confidence_score=row["confidence_score"],
            ric_validated=row["ric_validated"],
            last_verified_at=row["last_verified_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
