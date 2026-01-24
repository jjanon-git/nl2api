-- Run Label tracking for batch evaluations
-- Adds ability to label/describe evaluation runs for experiment tracking

-- Add new columns to batch_jobs table
ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS run_label TEXT;
ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS run_description TEXT;
ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS git_commit TEXT;
ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS git_branch TEXT;

-- Backfill existing jobs with 'untracked' label
UPDATE batch_jobs SET run_label = 'untracked' WHERE run_label IS NULL;

-- Make run_label required going forward
ALTER TABLE batch_jobs ALTER COLUMN run_label SET NOT NULL;

-- Add index for filtering by run_label
CREATE INDEX IF NOT EXISTS idx_batch_jobs_run_label ON batch_jobs(run_label);
