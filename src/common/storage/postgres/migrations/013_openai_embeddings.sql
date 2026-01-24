-- Migration: Change embedding dimension to support OpenAI embeddings
--
-- OpenAI text-embedding-3-small uses 1536 dimensions.
-- This migration reverts from 384 (local models) to 1536 for OpenAI.
--
-- IMPORTANT: This clears all existing embeddings. You must re-index after running.

-- Check current embedding count (for safety)
DO $$
DECLARE
    embedding_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO embedding_count FROM rag_documents WHERE embedding IS NOT NULL;
    IF embedding_count > 0 THEN
        RAISE WARNING 'Found % existing embeddings. They will be cleared and need regeneration.', embedding_count;
    END IF;
END $$;

-- Clear any existing embeddings (incompatible with new dimension)
UPDATE rag_documents SET embedding = NULL WHERE embedding IS NOT NULL;

-- Change rag_documents embedding column to 1536 dimensions for OpenAI
ALTER TABLE rag_documents
    ALTER COLUMN embedding TYPE vector(1536);

-- Recreate the HNSW index for the new dimension
-- Using higher ef_construction for better recall (was 64, now 100)
DROP INDEX IF EXISTS idx_rag_embedding;
CREATE INDEX idx_rag_embedding ON rag_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 100);

-- Add comment documenting the dimension
COMMENT ON COLUMN rag_documents.embedding IS 'Vector embedding (1536 dims for OpenAI text-embedding-3-small)';
