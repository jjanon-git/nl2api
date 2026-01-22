-- Migration: Change embedding dimension to support local models
--
-- Local embedding models (e.g., all-MiniLM-L6-v2) use 384 dimensions.
-- OpenAI models use 1536 dimensions.
--
-- This migration changes the dimension to 384 for local models.
-- To switch back to OpenAI, regenerate embeddings and run:
--   ALTER TABLE rag_documents ALTER COLUMN embedding TYPE vector(1536);
--
-- IMPORTANT: Only run this if no embeddings exist or you're willing to regenerate them.

-- Check current embedding count (for safety)
DO $$
DECLARE
    embedding_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO embedding_count FROM rag_documents WHERE embedding IS NOT NULL;
    IF embedding_count > 0 THEN
        RAISE WARNING 'Found % existing embeddings. They will need to be regenerated.', embedding_count;
    END IF;
END $$;

-- Clear any existing embeddings (they're incompatible with new dimension)
UPDATE rag_documents SET embedding = NULL WHERE embedding IS NOT NULL;

-- Change rag_documents embedding column to 384 dimensions
ALTER TABLE rag_documents
    ALTER COLUMN embedding TYPE vector(384);

-- Recreate the HNSW index for the new dimension
DROP INDEX IF EXISTS idx_rag_embedding;
CREATE INDEX idx_rag_embedding ON rag_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Add comment documenting the dimension
COMMENT ON COLUMN rag_documents.embedding IS 'Vector embedding (384 dims for local models, regenerate for OpenAI 1536)';
