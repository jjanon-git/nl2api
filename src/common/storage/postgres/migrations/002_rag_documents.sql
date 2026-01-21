-- RAG Documents Table for NL2API
-- Stores field codes, query examples, and economic indicators for retrieval
--
-- Migration: 002_rag_documents.sql

-- =============================================================================
-- RAG Documents Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS rag_documents (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Document content
    content TEXT NOT NULL,
    document_type VARCHAR(50) NOT NULL,  -- 'field_code', 'query_example', 'economic_indicator'

    -- Domain classification
    domain VARCHAR(50),  -- 'estimates', 'fundamentals', 'datastream', 'officers', 'screening'

    -- Field code specific fields
    field_code VARCHAR(100),  -- e.g., 'TR.EPSMean'

    -- Query example specific fields
    example_query TEXT,
    example_api_call TEXT,

    -- Additional metadata as JSONB
    metadata JSONB DEFAULT '{}',

    -- Vector embedding for similarity search (1536 dimensions for OpenAI ada-002/3-small)
    embedding vector(1536),

    -- Full-text search vector
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(field_code, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(example_query, '')), 'B')
    ) STORED,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for document type filtering
CREATE INDEX IF NOT EXISTS idx_rag_documents_type ON rag_documents(document_type);

-- Index for domain filtering
CREATE INDEX IF NOT EXISTS idx_rag_documents_domain ON rag_documents(domain);

-- Index for field code lookup
CREATE INDEX IF NOT EXISTS idx_rag_documents_field_code ON rag_documents(field_code);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_rag_documents_search ON rag_documents USING GIN(search_vector);

-- IVFFlat index for vector similarity search (for large datasets)
-- Note: Run this after inserting initial data for better index quality
-- CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding ON rag_documents
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- For smaller datasets, use HNSW (better quality but slower build)
CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding ON rag_documents
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Unique constraint for field codes
-- =============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_documents_unique_field
    ON rag_documents(domain, field_code)
    WHERE document_type = 'field_code' AND field_code IS NOT NULL;

-- =============================================================================
-- Trigger to update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION update_rag_documents_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_rag_documents_timestamp ON rag_documents;
CREATE TRIGGER trigger_rag_documents_timestamp
    BEFORE UPDATE ON rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_rag_documents_timestamp();
