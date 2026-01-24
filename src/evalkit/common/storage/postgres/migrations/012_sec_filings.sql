-- SEC Filings Ingestion Tables
-- Tracks SEC EDGAR filing metadata and ingestion status
--
-- Migration: 012_sec_filings.sql

-- =============================================================================
-- SEC Filings Table
-- =============================================================================
-- Tracks metadata and ingestion status for SEC 10-K and 10-Q filings

CREATE TABLE IF NOT EXISTS sec_filings (
    -- Primary key: SEC accession number (unique identifier for each filing)
    accession_number VARCHAR(30) PRIMARY KEY,

    -- Company identifiers
    cik VARCHAR(10) NOT NULL,           -- 10-digit CIK (zero-padded)
    ticker VARCHAR(20),                  -- Stock ticker (if available)
    company_name TEXT NOT NULL,

    -- Filing information
    filing_type VARCHAR(10) NOT NULL,    -- '10-K', '10-Q', '10-K/A', '10-Q/A'
    filing_date DATE NOT NULL,           -- Date filed with SEC
    period_of_report DATE NOT NULL,      -- Fiscal period end date

    -- Ingestion status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- Status values: pending, downloading, parsing, indexing, complete, failed

    -- Processing metadata
    download_path TEXT,                  -- Path to downloaded HTML file
    sections_extracted INTEGER DEFAULT 0, -- Number of sections extracted
    chunks_count INTEGER DEFAULT 0,       -- Number of RAG chunks created

    -- Error tracking
    error_message TEXT,                  -- Error message if failed
    retry_count INTEGER DEFAULT 0,       -- Number of retry attempts

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    downloaded_at TIMESTAMPTZ,
    parsed_at TIMESTAMPTZ,
    indexed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Indexes for SEC Filings
-- =============================================================================

-- Index for company lookups
CREATE INDEX IF NOT EXISTS idx_sec_filings_cik ON sec_filings(cik);

-- Index for ticker lookups
CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings(ticker);

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS idx_sec_filings_filing_date ON sec_filings(filing_date DESC);

-- Index for status filtering (e.g., find pending filings)
CREATE INDEX IF NOT EXISTS idx_sec_filings_status ON sec_filings(status);

-- Composite index for common query pattern: company + filing type + date
CREATE INDEX IF NOT EXISTS idx_sec_filings_company_type_date
    ON sec_filings(cik, filing_type, filing_date DESC);

-- =============================================================================
-- SEC Filing Ingestion Jobs Table
-- =============================================================================
-- Tracks batch ingestion jobs for SEC filings

CREATE TABLE IF NOT EXISTS sec_filing_ingestion_jobs (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Job identification
    job_name VARCHAR(100),               -- Human-readable job name

    -- Scope
    filing_types VARCHAR(50)[] NOT NULL DEFAULT ARRAY['10-K', '10-Q'],
    years_back INTEGER NOT NULL DEFAULT 2,
    target_companies INTEGER,            -- Number of companies to process
    target_filings INTEGER,              -- Estimated total filings

    -- Progress tracking
    companies_processed INTEGER DEFAULT 0,
    filings_processed INTEGER DEFAULT 0,
    chunks_indexed INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- Status values: pending, running, paused, completed, failed

    -- Error tracking
    error_count INTEGER DEFAULT 0,
    last_error TEXT,

    -- Checkpoint data (JSON for flexibility)
    checkpoint_data JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Add document type support to rag_documents
-- =============================================================================
-- The rag_documents table already supports document_type as a VARCHAR field.
-- We'll use 'sec_filing' as the document_type for SEC filing chunks.
--
-- Create a partial index for efficient SEC filing lookups
CREATE INDEX IF NOT EXISTS idx_rag_documents_sec_filing
    ON rag_documents(domain, (metadata->>'accession_number'))
    WHERE document_type = 'sec_filing';

-- Index for company lookups in RAG documents
CREATE INDEX IF NOT EXISTS idx_rag_documents_sec_company
    ON rag_documents((metadata->>'cik'))
    WHERE document_type = 'sec_filing';

-- Index for ticker lookups in RAG documents
CREATE INDEX IF NOT EXISTS idx_rag_documents_sec_ticker
    ON rag_documents((metadata->>'ticker'))
    WHERE document_type = 'sec_filing';

-- =============================================================================
-- Trigger to update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION update_sec_filings_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_sec_filings_timestamp ON sec_filings;
CREATE TRIGGER trigger_sec_filings_timestamp
    BEFORE UPDATE ON sec_filings
    FOR EACH ROW
    EXECUTE FUNCTION update_sec_filings_timestamp();

DROP TRIGGER IF EXISTS trigger_sec_filing_jobs_timestamp ON sec_filing_ingestion_jobs;
CREATE TRIGGER trigger_sec_filing_jobs_timestamp
    BEFORE UPDATE ON sec_filing_ingestion_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_sec_filings_timestamp();

-- =============================================================================
-- Helper Views
-- =============================================================================

-- View for filing ingestion summary by company
CREATE OR REPLACE VIEW sec_filing_ingestion_summary AS
SELECT
    cik,
    ticker,
    company_name,
    COUNT(*) as total_filings,
    COUNT(*) FILTER (WHERE status = 'complete') as completed,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    SUM(chunks_count) as total_chunks,
    MAX(filing_date) as latest_filing_date,
    MAX(indexed_at) as last_indexed_at
FROM sec_filings
GROUP BY cik, ticker, company_name;

-- View for daily ingestion stats
CREATE OR REPLACE VIEW sec_filing_daily_stats AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as filings_created,
    COUNT(*) FILTER (WHERE status = 'complete') as filings_completed,
    SUM(chunks_count) as chunks_indexed
FROM sec_filings
GROUP BY DATE(created_at)
ORDER BY date DESC;
