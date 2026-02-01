-- Migration 018: Add composite indexes for batch query performance
--
-- Problem: Queries like "WHERE batch_id = $1 ORDER BY created_at DESC" use
-- idx_scorecards_batch for filtering but then sort in memory.
--
-- Solution: Composite indexes that include both filter and sort columns,
-- enabling index-only scans without additional sorting.
--
-- Affected queries:
--   - get_by_batch(): SELECT * FROM scorecards WHERE batch_id = $1 ORDER BY created_at DESC
--   - list_by_test_case(): SELECT * FROM scorecards WHERE test_case_id = $1 ORDER BY created_at DESC

-- Composite index for batch queries with created_at ordering
-- Replaces separate lookups on idx_scorecards_batch + memory sort
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scorecards_batch_created
    ON scorecards(batch_id, created_at DESC);

-- Composite index for test_case queries with created_at ordering
-- Improves list_by_test_case() and history queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scorecards_test_case_created
    ON scorecards(test_case_id, created_at DESC);

-- Composite index for RAG documents (document_type, domain) filter
-- Improves hybrid retrieval queries that filter by both columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_documents_type_domain
    ON rag_documents(document_type, domain)
    WHERE document_type IS NOT NULL;
