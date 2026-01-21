-- Entity Resolution Schema
-- Canonical database for company/entity resolution
--
-- Supports 2M+ entities with:
-- - Full-text search (GIN index)
-- - Fuzzy matching (pg_trgm extension)
-- - Vector embeddings for semantic search (pgvector)

-- Enable trigram extension for fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =============================================================================
-- Entities Table (Canonical Store)
-- =============================================================================

CREATE TABLE IF NOT EXISTS entities (
    -- Primary key (UUID for flexibility)
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core identifiers (external systems)
    lei VARCHAR(20) UNIQUE,              -- Legal Entity Identifier (GLEIF)
    cik VARCHAR(10),                     -- SEC Central Index Key
    permid VARCHAR(20),                  -- Refinitiv PermID
    figi VARCHAR(12),                    -- OpenFIGI identifier

    -- Primary identifiers for resolution
    primary_name TEXT NOT NULL,          -- Official legal name
    display_name TEXT,                   -- Common display name (e.g., "Apple" vs "Apple Inc.")

    -- Stock identifiers (for public companies)
    ticker VARCHAR(20),                  -- Primary ticker symbol
    ric VARCHAR(50),                     -- Reuters Instrument Code
    exchange VARCHAR(20),                -- Primary exchange (NYSE, NASDAQ, LSE, etc.)

    -- Classification
    entity_type VARCHAR(50) NOT NULL DEFAULT 'company',  -- company, fund, government, branch
    entity_status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, inactive, merged, dissolved
    is_public BOOLEAN NOT NULL DEFAULT false,

    -- Geographic
    country_code CHAR(2),                -- ISO 3166-1 alpha-2
    region VARCHAR(100),                 -- State/Province
    city VARCHAR(100),

    -- Industry classification
    sic_code VARCHAR(4),                 -- Standard Industrial Classification
    naics_code VARCHAR(6),               -- North American Industry Classification
    gics_sector VARCHAR(100),            -- Global Industry Classification Standard

    -- Corporate hierarchy
    parent_entity_id UUID REFERENCES entities(id),
    ultimate_parent_id UUID REFERENCES entities(id),

    -- Data quality tracking
    data_source VARCHAR(50) NOT NULL,    -- gleif, sec_edgar, permid, openfigi, manual
    confidence_score FLOAT DEFAULT 1.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    ric_validated BOOLEAN DEFAULT false, -- True if RIC confirmed via OpenFIGI
    last_verified_at TIMESTAMPTZ,

    -- Vector embedding for semantic search (optional, for future use)
    name_embedding vector(1536),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique constraint on CIK (SEC companies)
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_cik_unique ON entities(cik) WHERE cik IS NOT NULL;

-- =============================================================================
-- Entity Aliases Table (for fuzzy matching)
-- =============================================================================

CREATE TABLE IF NOT EXISTS entity_aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,

    alias TEXT NOT NULL,                 -- Normalized alias (lowercase, stripped)
    alias_type VARCHAR(50) NOT NULL,     -- ticker, legal_name, trade_name, abbreviation, generated
    is_primary BOOLEAN DEFAULT false,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Dedup: one alias per entity
    UNIQUE(entity_id, alias)
);

-- =============================================================================
-- Indexes for Entity Resolution
-- =============================================================================

-- Primary identifier lookups (exact match)
CREATE INDEX IF NOT EXISTS idx_entities_lei ON entities(lei) WHERE lei IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_ticker ON entities(upper(ticker)) WHERE ticker IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_ric ON entities(ric) WHERE ric IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_permid ON entities(permid) WHERE permid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_figi ON entities(figi) WHERE figi IS NOT NULL;

-- Full-text search on names
CREATE INDEX IF NOT EXISTS idx_entities_primary_name_fts ON entities
    USING GIN(to_tsvector('english', primary_name));
CREATE INDEX IF NOT EXISTS idx_entities_display_name_fts ON entities
    USING GIN(to_tsvector('english', COALESCE(display_name, '')));

-- Trigram indexes for fuzzy matching (requires pg_trgm)
CREATE INDEX IF NOT EXISTS idx_entities_primary_name_trgm ON entities
    USING GIN(lower(primary_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_display_name_trgm ON entities
    USING GIN(lower(COALESCE(display_name, '')) gin_trgm_ops);

-- Alias lookups (critical for resolution performance)
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_exact ON entity_aliases(lower(alias));
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_trgm ON entity_aliases
    USING GIN(lower(alias) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);

-- Classification filters
CREATE INDEX IF NOT EXISTS idx_entities_country ON entities(country_code) WHERE country_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_exchange ON entities(exchange) WHERE exchange IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_is_public ON entities(is_public);
CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(entity_status);
CREATE INDEX IF NOT EXISTS idx_entities_data_source ON entities(data_source);

-- Vector similarity (for semantic entity matching) - only create if we have enough rows
-- Note: IVFFlat requires sufficient data; will be created by ingestion script after bulk load
-- CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
--     USING ivfflat (name_embedding vector_cosine_ops) WITH (lists = 1000);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Fast entity lookup by query string (with fuzzy matching fallback)
-- Returns best matches ordered by match quality
CREATE OR REPLACE FUNCTION resolve_entity(
    p_query TEXT,
    p_fuzzy_threshold FLOAT DEFAULT 0.3,
    p_limit INT DEFAULT 5
)
RETURNS TABLE (
    entity_id UUID,
    primary_name TEXT,
    display_name TEXT,
    ric VARCHAR(50),
    ticker VARCHAR(20),
    exchange VARCHAR(20),
    match_type TEXT,
    similarity FLOAT
) AS $$
DECLARE
    v_normalized TEXT := lower(trim(p_query));
BEGIN
    -- 1. Exact alias match (fastest path)
    RETURN QUERY
    SELECT e.id, e.primary_name, e.display_name, e.ric, e.ticker, e.exchange,
           'exact'::TEXT, 1.0::FLOAT
    FROM entities e
    JOIN entity_aliases a ON e.id = a.entity_id
    WHERE lower(a.alias) = v_normalized
      AND e.entity_status = 'active'
    LIMIT 1;

    IF FOUND THEN RETURN; END IF;

    -- 2. Ticker match (case-insensitive)
    RETURN QUERY
    SELECT e.id, e.primary_name, e.display_name, e.ric, e.ticker, e.exchange,
           'ticker'::TEXT, 1.0::FLOAT
    FROM entities e
    WHERE upper(e.ticker) = upper(p_query)
      AND e.entity_status = 'active'
    LIMIT 1;

    IF FOUND THEN RETURN; END IF;

    -- 3. RIC match (exact)
    RETURN QUERY
    SELECT e.id, e.primary_name, e.display_name, e.ric, e.ticker, e.exchange,
           'ric'::TEXT, 1.0::FLOAT
    FROM entities e
    WHERE e.ric = upper(p_query)
      AND e.entity_status = 'active'
    LIMIT 1;

    IF FOUND THEN RETURN; END IF;

    -- 4. Fuzzy match on aliases (trigram similarity)
    RETURN QUERY
    SELECT e.id, e.primary_name, e.display_name, e.ric, e.ticker, e.exchange,
           'fuzzy'::TEXT,
           similarity(lower(a.alias), v_normalized)::FLOAT AS sim
    FROM entities e
    JOIN entity_aliases a ON e.id = a.entity_id
    WHERE similarity(lower(a.alias), v_normalized) > p_fuzzy_threshold
      AND e.entity_status = 'active'
    ORDER BY sim DESC
    LIMIT p_limit;

END;
$$ LANGUAGE plpgsql STABLE;

-- Batch resolve multiple entities in one call
CREATE OR REPLACE FUNCTION resolve_entities_batch(
    p_queries TEXT[],
    p_fuzzy_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    query TEXT,
    entity_id UUID,
    primary_name TEXT,
    ric VARCHAR(50),
    ticker VARCHAR(20),
    match_type TEXT,
    similarity FLOAT
) AS $$
DECLARE
    v_query TEXT;
BEGIN
    FOREACH v_query IN ARRAY p_queries
    LOOP
        RETURN QUERY
        SELECT v_query, r.entity_id, r.primary_name, r.ric, r.ticker, r.match_type, r.similarity
        FROM resolve_entity(v_query, p_fuzzy_threshold, 1) r;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Triggers
-- =============================================================================

-- Update updated_at timestamp on entity changes
CREATE TRIGGER update_entities_updated_at
    BEFORE UPDATE ON entities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Statistics Views (for monitoring)
-- =============================================================================

CREATE OR REPLACE VIEW entity_stats AS
SELECT
    COUNT(*) AS total_entities,
    COUNT(*) FILTER (WHERE is_public) AS public_entities,
    COUNT(*) FILTER (WHERE NOT is_public) AS private_entities,
    COUNT(*) FILTER (WHERE ric IS NOT NULL) AS entities_with_ric,
    COUNT(*) FILTER (WHERE ric_validated) AS entities_with_validated_ric,
    COUNT(DISTINCT country_code) AS countries,
    COUNT(DISTINCT exchange) AS exchanges,
    COUNT(DISTINCT data_source) AS data_sources
FROM entities
WHERE entity_status = 'active';

CREATE OR REPLACE VIEW entity_coverage_by_source AS
SELECT
    data_source,
    COUNT(*) AS entity_count,
    COUNT(*) FILTER (WHERE ric IS NOT NULL) AS with_ric,
    COUNT(*) FILTER (WHERE ric_validated) AS ric_validated,
    ROUND(100.0 * COUNT(*) FILTER (WHERE ric IS NOT NULL) / NULLIF(COUNT(*), 0), 2) AS ric_coverage_pct
FROM entities
WHERE entity_status = 'active'
GROUP BY data_source
ORDER BY entity_count DESC;
