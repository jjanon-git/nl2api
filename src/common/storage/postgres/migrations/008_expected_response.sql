-- Migration 008: Evaluation Data Contract Improvements
--
-- This migration updates the test_cases table to support the new evaluation data contract:
-- 1. Rename expected_raw_data -> expected_response (clearer semantics)
-- 2. Make expected_nl_response nullable (not all capabilities require NL response)
--
-- See docs/plans/evaluation-data-contract-plan.md for full context.

-- =============================================================================
-- Step 1: Rename expected_raw_data to expected_response
-- =============================================================================
-- This provides clearer semantics:
-- - expected_response: structured data from API execution
-- - expected_nl_response: human-readable summary sentence

ALTER TABLE test_cases
RENAME COLUMN expected_raw_data TO expected_response;

-- =============================================================================
-- Step 2: Make expected_nl_response nullable
-- =============================================================================
-- Different evaluation capabilities have different requirements:
-- - nl2api: requires expected_nl_response
-- - entity_extraction: no NL response needed
-- - tool_generation: no NL response needed
--
-- The TestCaseSetConfig._meta block in fixture files declares per-set requirements.

ALTER TABLE test_cases
ALTER COLUMN expected_nl_response DROP NOT NULL;

-- =============================================================================
-- Verification
-- =============================================================================
-- Verify the changes (uncomment to run manually):
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'test_cases'
--   AND column_name IN ('expected_response', 'expected_nl_response');
