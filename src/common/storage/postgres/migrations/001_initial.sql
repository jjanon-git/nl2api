-- NL2API Initial Schema
-- PostgreSQL + pgvector migration
--
-- This schema mirrors the Azure Table Storage / AI Search design for local development.
-- Partition/Row key patterns are preserved for easy migration.

-- Enable pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Test Cases Table (Gold Store)
-- =============================================================================
-- Equivalent to Azure AI Search index
-- Supports full-text search and vector similarity

CREATE TABLE IF NOT EXISTS test_cases (
    -- Primary key (maps to Azure AI Search document ID)
    id UUID PRIMARY KEY,

    -- The 4-tuple core data
    nl_query TEXT NOT NULL,
    expected_tool_calls JSONB NOT NULL,
    expected_raw_data JSONB,
    expected_nl_response TEXT NOT NULL,

    -- Metadata fields
    api_version VARCHAR(50) NOT NULL,
    complexity_level INTEGER NOT NULL DEFAULT 1 CHECK (complexity_level BETWEEN 1 AND 5),
    tags TEXT[] DEFAULT '{}',
    author VARCHAR(255),
    source VARCHAR(100),

    -- Lifecycle status
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    stale_reason TEXT,

    -- Content hash for deduplication
    content_hash VARCHAR(32),

    -- Vector embedding for similarity search (ada-002 = 1536 dimensions)
    embedding vector(1536),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_test_cases_tags ON test_cases USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_test_cases_complexity ON test_cases(complexity_level);
CREATE INDEX IF NOT EXISTS idx_test_cases_status ON test_cases(status);
CREATE INDEX IF NOT EXISTS idx_test_cases_api_version ON test_cases(api_version);
CREATE INDEX IF NOT EXISTS idx_test_cases_content_hash ON test_cases(content_hash);
CREATE INDEX IF NOT EXISTS idx_test_cases_created ON test_cases(created_at DESC);

-- Full-text search index on nl_query
CREATE INDEX IF NOT EXISTS idx_test_cases_nl_query_fts ON test_cases
    USING GIN(to_tsvector('english', nl_query));

-- Vector similarity index (IVFFlat for efficient approximate nearest neighbor)
-- Note: Requires at least 100 rows for IVFFlat to be effective
-- For small datasets, exact search (no index) is used automatically
CREATE INDEX IF NOT EXISTS idx_test_cases_embedding ON test_cases
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- Scorecards Table (Results Store)
-- =============================================================================
-- Equivalent to Azure Table Storage with PartitionKey/RowKey pattern

CREATE TABLE IF NOT EXISTS scorecards (
    -- Primary key
    id UUID PRIMARY KEY,

    -- Foreign key to test case
    test_case_id UUID NOT NULL,

    -- Batch/run context (maps to Azure Table Storage PartitionKey pattern)
    batch_id VARCHAR(100),
    run_id VARCHAR(100),

    -- Stage results stored as JSONB for flexibility
    syntax_result JSONB NOT NULL,
    logic_result JSONB,
    execution_result JSONB,
    semantics_result JSONB,

    -- Captured outputs
    generated_tool_calls JSONB,
    generated_nl_response TEXT,

    -- Computed metrics (denormalized for query efficiency)
    overall_passed BOOLEAN NOT NULL,
    overall_score FLOAT NOT NULL,

    -- Execution context
    worker_id VARCHAR(100) NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 1,
    message_id VARCHAR(255),
    total_latency_ms INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Indexes for scorecard queries
CREATE INDEX IF NOT EXISTS idx_scorecards_test_case ON scorecards(test_case_id);
CREATE INDEX IF NOT EXISTS idx_scorecards_batch ON scorecards(batch_id);
CREATE INDEX IF NOT EXISTS idx_scorecards_run ON scorecards(run_id);
CREATE INDEX IF NOT EXISTS idx_scorecards_created ON scorecards(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scorecards_passed ON scorecards(overall_passed);

-- Composite index for Azure Table Storage partition-like queries
CREATE INDEX IF NOT EXISTS idx_scorecards_batch_test ON scorecards(batch_id, test_case_id);
CREATE INDEX IF NOT EXISTS idx_scorecards_test_created ON scorecards(test_case_id, created_at DESC);

-- =============================================================================
-- Batch Jobs Table (Optional - for tracking batch submissions)
-- =============================================================================

CREATE TABLE IF NOT EXISTS batch_jobs (
    id UUID PRIMARY KEY,

    -- Tracking
    total_tests INTEGER NOT NULL CHECK (total_tests >= 1),
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Metadata
    submitted_by VARCHAR(255),
    priority VARCHAR(20) NOT NULL DEFAULT 'normal',
    tags TEXT[] DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_created ON batch_jobs(created_at DESC);

-- =============================================================================
-- Idempotency Records Table (for exactly-once processing)
-- =============================================================================

CREATE TABLE IF NOT EXISTS idempotency_records (
    idempotency_key VARCHAR(255) PRIMARY KEY,
    scorecard_id UUID NOT NULL,
    worker_id VARCHAR(100) NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_idempotency_processed ON idempotency_records(processed_at);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for test_cases updated_at
DROP TRIGGER IF EXISTS update_test_cases_updated_at ON test_cases;
CREATE TRIGGER update_test_cases_updated_at
    BEFORE UPDATE ON test_cases
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
