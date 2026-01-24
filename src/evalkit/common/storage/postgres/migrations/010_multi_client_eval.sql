-- Multi-Client Evaluation Platform Schema
-- Adds client tracking, eval mode, and cost tracking to scorecards
-- Adds continuous evaluation scheduling and regression alerts

-- =============================================================================
-- Scorecard Extensions for Multi-Client Tracking
-- =============================================================================

-- Client type tracking (internal, mcp_claude, mcp_chatgpt, mcp_custom)
ALTER TABLE scorecards ADD COLUMN IF NOT EXISTS client_type VARCHAR(50) DEFAULT 'internal';

-- Client version (e.g., "claude-opus-4.5-20251101")
ALTER TABLE scorecards ADD COLUMN IF NOT EXISTS client_version VARCHAR(100);

-- Evaluation mode (orchestrator, tool_only, routing, resolver, mcp_passthrough)
ALTER TABLE scorecards ADD COLUMN IF NOT EXISTS eval_mode VARCHAR(50) DEFAULT 'orchestrator';

-- Token usage tracking
ALTER TABLE scorecards ADD COLUMN IF NOT EXISTS input_tokens INTEGER;
ALTER TABLE scorecards ADD COLUMN IF NOT EXISTS output_tokens INTEGER;

-- Cost tracking in USD
ALTER TABLE scorecards ADD COLUMN IF NOT EXISTS estimated_cost_usd DECIMAL(10, 6);

-- Indexes for client-based queries
CREATE INDEX IF NOT EXISTS idx_scorecards_client ON scorecards(client_type, client_version);
CREATE INDEX IF NOT EXISTS idx_scorecards_eval_mode ON scorecards(eval_mode);
CREATE INDEX IF NOT EXISTS idx_scorecards_client_created ON scorecards(client_type, created_at DESC);

-- =============================================================================
-- Evaluation Schedules Table (Continuous Evaluation)
-- =============================================================================

CREATE TABLE IF NOT EXISTS eval_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Schedule identity
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,

    -- Target configuration
    client_type VARCHAR(50) NOT NULL DEFAULT 'internal',
    client_version VARCHAR(100),
    eval_mode VARCHAR(50) NOT NULL DEFAULT 'orchestrator',

    -- Schedule configuration
    cron_expression VARCHAR(100) NOT NULL,  -- e.g., "0 2 * * *" = daily at 2 AM
    test_suite_tags TEXT[] DEFAULT '{}',
    test_limit INTEGER,  -- Max tests per run, NULL for all

    -- State
    enabled BOOLEAN NOT NULL DEFAULT true,
    last_run_at TIMESTAMPTZ,
    last_run_batch_id VARCHAR(100),
    next_run_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_schedules_enabled ON eval_schedules(enabled);
CREATE INDEX IF NOT EXISTS idx_eval_schedules_next_run ON eval_schedules(next_run_at) WHERE enabled = true;

-- Trigger for eval_schedules updated_at
DROP TRIGGER IF EXISTS update_eval_schedules_updated_at ON eval_schedules;
CREATE TRIGGER update_eval_schedules_updated_at
    BEFORE UPDATE ON eval_schedules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Regression Alerts Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS regression_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Alert context
    schedule_id UUID REFERENCES eval_schedules(id) ON DELETE SET NULL,
    batch_id VARCHAR(100) NOT NULL,
    previous_batch_id VARCHAR(100),

    -- Alert details
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('warning', 'critical')),
    metric_name VARCHAR(100) NOT NULL,  -- e.g., "pass_rate", "p95_latency_ms"
    previous_value DECIMAL(10, 4),
    current_value DECIMAL(10, 4) NOT NULL,
    threshold_value DECIMAL(10, 4) NOT NULL,
    delta_pct DECIMAL(10, 4),  -- Percentage change

    -- Status
    acknowledged BOOLEAN NOT NULL DEFAULT false,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMPTZ,
    notes TEXT,

    -- Notification tracking
    webhook_sent BOOLEAN NOT NULL DEFAULT false,
    email_sent BOOLEAN NOT NULL DEFAULT false,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regression_alerts_batch ON regression_alerts(batch_id);
CREATE INDEX IF NOT EXISTS idx_regression_alerts_severity ON regression_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_regression_alerts_unacknowledged ON regression_alerts(acknowledged, created_at DESC) WHERE acknowledged = false;
CREATE INDEX IF NOT EXISTS idx_regression_alerts_created ON regression_alerts(created_at DESC);

-- =============================================================================
-- Client Comparison View (for quick summaries)
-- =============================================================================

CREATE OR REPLACE VIEW client_comparison_summary AS
SELECT
    client_type,
    client_version,
    eval_mode,
    DATE_TRUNC('day', created_at) as eval_date,
    COUNT(*) as total_tests,
    COUNT(*) FILTER (WHERE overall_passed = true) as passed_count,
    COUNT(*) FILTER (WHERE overall_passed = false) as failed_count,
    ROUND(AVG(overall_score)::numeric, 4) as avg_score,
    ROUND((COUNT(*) FILTER (WHERE overall_passed = true)::numeric / NULLIF(COUNT(*), 0))::numeric, 4) as pass_rate,
    SUM(COALESCE(input_tokens, 0)) as total_input_tokens,
    SUM(COALESCE(output_tokens, 0)) as total_output_tokens,
    SUM(COALESCE(estimated_cost_usd, 0)) as total_cost_usd
FROM scorecards
GROUP BY client_type, client_version, eval_mode, DATE_TRUNC('day', created_at)
ORDER BY eval_date DESC, client_type, client_version;
