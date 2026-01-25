-- Parent-Child Chunks for Small-to-Big Retrieval
-- Adds parent_id and chunk_level to support hierarchical chunking strategy
--
-- Migration: 015_parent_child_chunks.sql
--
-- Small-to-big retrieval strategy:
-- 1. Create small child chunks (512 chars) for precise semantic matching
-- 2. Link children to larger parent chunks (4000 chars) for context
-- 3. Search children, return parents for better context

-- =============================================================================
-- Add parent-child relationship columns
-- =============================================================================

-- Parent reference (self-referential foreign key)
ALTER TABLE rag_documents
ADD COLUMN IF NOT EXISTS parent_id UUID REFERENCES rag_documents(id) ON DELETE SET NULL;

-- Chunk level: 0 = parent (large context), 1 = child (small, precise)
ALTER TABLE rag_documents
ADD COLUMN IF NOT EXISTS chunk_level INTEGER DEFAULT 0;

-- Add comment explaining the columns
COMMENT ON COLUMN rag_documents.parent_id IS 'Reference to parent chunk for small-to-big retrieval. NULL for parent chunks or standalone documents.';
COMMENT ON COLUMN rag_documents.chunk_level IS 'Chunk hierarchy level: 0=parent (4000 chars), 1=child (512 chars)';

-- =============================================================================
-- Indexes for efficient parent-child queries
-- =============================================================================

-- Index for looking up children by parent
CREATE INDEX IF NOT EXISTS idx_rag_documents_parent_id ON rag_documents(parent_id)
WHERE parent_id IS NOT NULL;

-- Index for filtering by chunk level
CREATE INDEX IF NOT EXISTS idx_rag_documents_chunk_level ON rag_documents(chunk_level);

-- Composite index for efficient child retrieval
CREATE INDEX IF NOT EXISTS idx_rag_documents_chunk_search ON rag_documents(chunk_level, domain)
WHERE chunk_level = 1;

-- =============================================================================
-- Function to get parent context for a list of child IDs
-- =============================================================================

CREATE OR REPLACE FUNCTION get_parent_chunks(child_ids UUID[])
RETURNS TABLE (
    parent_id UUID,
    content TEXT,
    domain VARCHAR(50),
    metadata JSONB,
    child_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT
        p.id as parent_id,
        p.content,
        p.domain,
        p.metadata,
        COUNT(c.id) as child_count
    FROM rag_documents p
    INNER JOIN rag_documents c ON c.parent_id = p.id
    WHERE c.id = ANY(child_ids)
    GROUP BY p.id, p.content, p.domain, p.metadata
    ORDER BY child_count DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_parent_chunks(UUID[]) IS 'Returns parent chunks for given child IDs, ordered by matching child count';
