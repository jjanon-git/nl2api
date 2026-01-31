-- Multi-Source Gold Evaluation Data Migration
-- Adds source_type classification and structured metadata for test cases
--
-- Source types:
--   - customer: Real production questions
--   - sme: Subject matter expert curated
--   - synthetic: Generated (LLM or programmatic)
--   - hybrid: Mixed origin (e.g., customer Q + SME answer)

-- 1. Create enums
DO $$ BEGIN
    CREATE TYPE data_source_type AS ENUM ('customer', 'sme', 'synthetic', 'hybrid');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE review_status_type AS ENUM ('pending', 'approved', 'rejected', 'needs_revision');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- 2. Add columns (without default initially for data migration)
ALTER TABLE test_cases
ADD COLUMN IF NOT EXISTS source_type data_source_type,
ADD COLUMN IF NOT EXISTS source_metadata JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS review_status review_status_type DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS quality_score REAL;

-- 3. Migrate existing data based on source field
-- Maps: "generated:*" -> synthetic, "fixture:*" -> synthetic
UPDATE test_cases
SET source_type = CASE
    WHEN source LIKE 'generated:%' THEN 'synthetic'::data_source_type
    WHEN source LIKE 'fixture:%' THEN 'synthetic'::data_source_type
    ELSE 'synthetic'::data_source_type  -- Default for unknown
END
WHERE source_type IS NULL;

-- 4. Preserve original source info in source_metadata for provenance
UPDATE test_cases
SET source_metadata = jsonb_build_object(
    'source_type', source_type::text,
    'generator_name',
    CASE
        WHEN source LIKE 'generated:%' THEN substring(source from 11)  -- Strip "generated:"
        WHEN source LIKE 'fixture:%' THEN substring(source from 9)     -- Strip "fixture:"
        ELSE source
    END,
    'migrated_from', source,
    'migration_date', NOW()::text,
    'review_status', 'pending'
)
WHERE source IS NOT NULL
  AND (source_metadata IS NULL OR source_metadata = '{}'::jsonb);

-- 5. Set default for new rows
ALTER TABLE test_cases
ALTER COLUMN source_type SET DEFAULT 'synthetic';

-- 6. Create indexes for filtering
CREATE INDEX IF NOT EXISTS idx_test_cases_source_type ON test_cases(source_type);
CREATE INDEX IF NOT EXISTS idx_test_cases_review_status ON test_cases(review_status);
CREATE INDEX IF NOT EXISTS idx_test_cases_source_metadata ON test_cases USING GIN (source_metadata);

-- 7. Add check constraint for quality_score range
ALTER TABLE test_cases
ADD CONSTRAINT chk_quality_score_range
CHECK (quality_score IS NULL OR (quality_score >= 0.0 AND quality_score <= 1.0));
