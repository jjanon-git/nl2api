-- Migration 019: Add trace_id and span_id to scorecards for Jaeger correlation
-- This enables direct lookup from scorecard failures to distributed traces

-- Add trace_id column (32 hex characters for 128-bit OTEL trace ID)
ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS trace_id VARCHAR(32);

-- Add span_id column (16 hex characters for 64-bit OTEL span ID)
ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS span_id VARCHAR(16);

-- Index for trace lookups (when searching scorecards by trace)
CREATE INDEX IF NOT EXISTS idx_scorecards_trace_id
ON scorecards(trace_id)
WHERE trace_id IS NOT NULL;

-- Comment for documentation
COMMENT ON COLUMN scorecards.trace_id IS 'OpenTelemetry trace ID (32 hex chars) for Jaeger lookup';
COMMENT ON COLUMN scorecards.span_id IS 'OpenTelemetry span ID (16 hex chars) for root evaluation span';
