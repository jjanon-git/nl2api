-- Routing Cache Table
-- Stores cached routing decisions for semantic similarity matching
--
-- Migration: 006_routing_cache.sql
--
-- This table enables:
-- 1. Exact match cache lookup via unique query constraint
-- 2. Semantic similarity matching via pgvector embedding index
--
-- Used by LLMToolRouter and EscalatingLLMRouter for caching routing decisions
-- to reduce LLM API calls and latency.

-- =============================================================================
-- Create routing_cache table
-- =============================================================================

CREATE TABLE IF NOT EXISTS routing_cache (
    -- Unique identifier
    id SERIAL PRIMARY KEY,

    -- Original query text (exact match key)
    query TEXT NOT NULL UNIQUE,

    -- Embedding vector for semantic similarity (1536 dims for text-embedding-3-small)
    embedding vector(1536),

    -- Routing decision
    domain TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    reasoning TEXT,

    -- Metadata
    model_used TEXT,
    latency_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Create indexes
-- =============================================================================

-- IVFFlat index for semantic similarity search
-- Using lists = 100 for expected ~10K cached routing decisions
-- (sqrt(10000) = 100, appropriate for this size)
CREATE INDEX IF NOT EXISTS routing_cache_embedding_idx
ON routing_cache USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for domain statistics
CREATE INDEX IF NOT EXISTS routing_cache_domain_idx
ON routing_cache (domain);

-- Index for timestamp-based cleanup
CREATE INDEX IF NOT EXISTS routing_cache_updated_at_idx
ON routing_cache (updated_at);

-- =============================================================================
-- Create update trigger for updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_routing_cache_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS routing_cache_updated_at_trigger ON routing_cache;
CREATE TRIGGER routing_cache_updated_at_trigger
    BEFORE UPDATE ON routing_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_routing_cache_updated_at();

-- =============================================================================
-- Helper views
-- =============================================================================

-- View for cache statistics
CREATE OR REPLACE VIEW routing_cache_stats AS
SELECT
    domain,
    COUNT(*) as cached_queries,
    AVG(confidence) as avg_confidence,
    MIN(created_at) as oldest_entry,
    MAX(updated_at) as newest_entry
FROM routing_cache
GROUP BY domain
ORDER BY cached_queries DESC;

-- =============================================================================
-- Cleanup function for old entries
-- =============================================================================

-- Function to clean up old cache entries (default: older than 24 hours)
CREATE OR REPLACE FUNCTION cleanup_routing_cache(max_age_hours INTEGER DEFAULT 24)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM routing_cache
    WHERE updated_at < NOW() - INTERVAL '1 hour' * max_age_hours;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Usage notes
-- =============================================================================
--
-- Semantic similarity query with probes:
--   SET LOCAL ivfflat.probes = 10;
--   SELECT domain, confidence, reasoning,
--          1 - (embedding <=> $1::vector) as similarity
--   FROM routing_cache
--   WHERE 1 - (embedding <=> $1::vector) > 0.92
--   ORDER BY similarity DESC
--   LIMIT 1;
--
-- Cleanup old entries:
--   SELECT cleanup_routing_cache(24);  -- Remove entries older than 24 hours
--
-- View cache statistics:
--   SELECT * FROM routing_cache_stats;
