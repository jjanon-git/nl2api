-- Migration: 016_batch_checkpoint.sql
-- Purpose: Add checkpoint fields to batch_jobs for resume capability
-- Date: 2026-01-25

-- Add checkpoint tracking columns
ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS last_checkpoint_at TIMESTAMPTZ;
ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS error_count INTEGER DEFAULT 0;
ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS last_error TEXT;

-- Index for finding resumable batches (in_progress or failed status)
CREATE INDEX IF NOT EXISTS idx_batch_jobs_resumable
    ON batch_jobs(status)
    WHERE status IN ('in_progress', 'failed');

-- Comment for documentation
COMMENT ON COLUMN batch_jobs.last_checkpoint_at IS 'Last time progress was checkpointed to database';
COMMENT ON COLUMN batch_jobs.error_count IS 'Number of errors encountered during batch evaluation';
COMMENT ON COLUMN batch_jobs.last_error IS 'Most recent error message (if any)';
