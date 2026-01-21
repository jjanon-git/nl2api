-- IVFFlat Index for Large Datasets
-- Replaces HNSW with IVFFlat for better performance at 1M+ vectors
--
-- Migration: 005_ivfflat_index.sql
--
-- Note: Run this migration AFTER you have inserted a significant amount
-- of initial data for better index quality. IVFFlat performs better when
-- built on existing data.
--
-- lists = 1000 is tuned for ~1M vectors (sqrt(n) to 4*sqrt(n) rule of thumb)
-- For 1M vectors: sqrt(1M) = 1000, so lists = 1000 is appropriate

-- =============================================================================
-- Drop existing HNSW index and create IVFFlat
-- =============================================================================

-- Drop the existing HNSW index
DROP INDEX IF EXISTS idx_rag_documents_embedding;

-- Create IVFFlat index with 1000 lists (optimized for 1M+ vectors)
-- vector_cosine_ops for cosine similarity (common for text embeddings)
CREATE INDEX idx_rag_documents_embedding ON rag_documents
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);

-- =============================================================================
-- Usage notes for probes setting
-- =============================================================================
--
-- At query time, set ivfflat.probes to balance speed vs accuracy:
--   SET ivfflat.probes = 10;   -- Fast, ~70% recall
--   SET ivfflat.probes = 50;   -- Balanced, ~90% recall
--   SET ivfflat.probes = 100;  -- High quality, ~95% recall
--
-- Example query with probes:
--   SET LOCAL ivfflat.probes = 50;
--   SELECT * FROM rag_documents
--   ORDER BY embedding <=> $1::vector
--   LIMIT 10;
