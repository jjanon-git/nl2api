-- Indexing Checkpoint Table
-- Stores checkpoint state for resumable large indexing jobs
--
-- Migration: 004_indexing_checkpoint.sql

-- =============================================================================
-- Indexing Checkpoints Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS indexing_checkpoints (
    -- Job identifier (user-provided or auto-generated)
    job_id VARCHAR(100) PRIMARY KEY,

    -- Progress tracking
    total_items INTEGER NOT NULL DEFAULT 0,
    processed_items INTEGER NOT NULL DEFAULT 0,
    last_offset INTEGER NOT NULL DEFAULT 0,

    -- Job status
    status VARCHAR(20) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'paused')),

    -- Error tracking
    error_message TEXT,
    error_count INTEGER NOT NULL DEFAULT 0,

    -- Job metadata
    domain VARCHAR(50),  -- Domain being indexed
    batch_size INTEGER NOT NULL DEFAULT 100,
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for status filtering (find running/failed jobs)
CREATE INDEX IF NOT EXISTS idx_indexing_checkpoints_status
    ON indexing_checkpoints(status);

-- Index for domain filtering
CREATE INDEX IF NOT EXISTS idx_indexing_checkpoints_domain
    ON indexing_checkpoints(domain);

-- =============================================================================
-- Trigger to update last_updated timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION update_indexing_checkpoint_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_indexing_checkpoint_timestamp ON indexing_checkpoints;
CREATE TRIGGER trigger_indexing_checkpoint_timestamp
    BEFORE UPDATE ON indexing_checkpoints
    FOR EACH ROW
    EXECUTE FUNCTION update_indexing_checkpoint_timestamp();
