-- Migration: 003_conversations
-- Description: Add conversations table for multi-turn support

-- Conversations table stores each turn in a conversation session
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    turn_number INTEGER NOT NULL,

    -- Query information
    user_query TEXT NOT NULL,
    expanded_query TEXT,  -- After context expansion (e.g., "their EPS" -> "Apple's EPS")

    -- Response information
    tool_calls JSONB,
    resolved_entities JSONB,  -- {"Apple": "AAPL.O", ...}
    domain VARCHAR(50),
    confidence FLOAT,

    -- Clarification (if needed)
    needs_clarification BOOLEAN DEFAULT FALSE,
    clarification_questions JSONB,
    clarification_response TEXT,  -- User's answer to clarification

    -- Metadata
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE(session_id, turn_number)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_conversations_session
    ON conversations(session_id, turn_number DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_created
    ON conversations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_domain
    ON conversations(domain) WHERE domain IS NOT NULL;

-- Sessions table for tracking active sessions
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Session metadata
    user_id VARCHAR(255),  -- Optional user identifier
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),

    -- Session state
    is_active BOOLEAN DEFAULT TRUE,
    total_turns INTEGER DEFAULT 0,

    -- Context summary (for long conversations)
    context_summary TEXT,

    -- Configuration overrides for this session
    config_overrides JSONB
);

CREATE INDEX IF NOT EXISTS idx_sessions_active
    ON conversation_sessions(is_active, last_activity_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON conversation_sessions(user_id) WHERE user_id IS NOT NULL;

-- Function to update session activity
CREATE OR REPLACE FUNCTION update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversation_sessions
    SET last_activity_at = NOW(),
        total_turns = total_turns + 1
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update session on new conversation turn
DROP TRIGGER IF EXISTS trg_update_session_activity ON conversations;
CREATE TRIGGER trg_update_session_activity
    AFTER INSERT ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_session_activity();
