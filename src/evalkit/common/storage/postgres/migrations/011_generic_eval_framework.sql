-- Migration: Generic Evaluation Framework Support
-- Adds generic fields to support evaluation packs beyond NL2API
--
-- This migration adds:
-- 1. Generic input/expected columns to test_cases
-- 2. Generic stage_results/pack_name columns to scorecards
--
-- Existing NL2API-specific columns are preserved for backwards compatibility.

-- =============================================================================
-- Test Cases: Add Generic Fields
-- =============================================================================

-- Generic input data (pack-specific schema)
ALTER TABLE test_cases
ADD COLUMN IF NOT EXISTS input_json JSONB DEFAULT '{}';

-- Generic expected output (pack-specific schema)
ALTER TABLE test_cases
ADD COLUMN IF NOT EXISTS expected_json JSONB DEFAULT '{}';

-- =============================================================================
-- Scorecards: Add Generic Fields
-- =============================================================================

-- Name of the evaluation pack used
ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS pack_name VARCHAR(50) DEFAULT 'nl2api';

-- Generic stage results (keyed by stage name)
ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS stage_results JSONB DEFAULT '{}';

-- Scoring weights used for overall calculation
ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS stage_weights JSONB DEFAULT '{}';

-- Generic captured output from target system
ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS generated_output JSONB DEFAULT '{}';

-- =============================================================================
-- Make NL2API-specific columns nullable for generic packs
-- =============================================================================

-- syntax_result was NOT NULL but is only required for NL2API pack
ALTER TABLE scorecards ALTER COLUMN syntax_result DROP NOT NULL;

-- =============================================================================
-- Indexes for Generic Queries
-- =============================================================================

-- Index for pack-based queries
CREATE INDEX IF NOT EXISTS idx_scorecards_pack ON scorecards(pack_name);

-- GIN index for stage_results queries (e.g., find all with retrieval stage)
CREATE INDEX IF NOT EXISTS idx_scorecards_stage_results ON scorecards USING GIN(stage_results);

-- =============================================================================
-- Migrate existing data (populate generic fields from NL2API-specific fields)
-- =============================================================================

-- Populate input_json from nl_query for existing test cases
UPDATE test_cases
SET input_json = jsonb_build_object('nl_query', nl_query)
WHERE input_json = '{}' AND nl_query IS NOT NULL;

-- Populate expected_json from expected_tool_calls and expected_nl_response
UPDATE test_cases
SET expected_json = jsonb_build_object(
    'tool_calls', expected_tool_calls,
    'nl_response', expected_nl_response
)
WHERE expected_json = '{}' AND expected_tool_calls IS NOT NULL;

-- Note: stage_results on scorecards is NOT backfilled automatically
-- because converting from the individual result columns would require
-- complex JSON manipulation. The repository code handles dual-read
-- (reading from both old columns and new stage_results column).
