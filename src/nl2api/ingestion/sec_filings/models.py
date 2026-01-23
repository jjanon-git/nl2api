"""
SEC Filing Data Models

Data models for SEC EDGAR filing ingestion pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class FilingType(str, Enum):
    """Types of SEC filings supported for ingestion."""

    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_10K_A = "10-K/A"  # Amended 10-K
    FORM_10Q_A = "10-Q/A"  # Amended 10-Q


class FilingSection(str, Enum):
    """Standard sections in 10-K and 10-Q filings."""

    # 10-K Sections (Part I)
    BUSINESS = "business"  # Item 1
    RISK_FACTORS = "risk_factors"  # Item 1A
    UNRESOLVED_STAFF_COMMENTS = "unresolved_staff_comments"  # Item 1B
    PROPERTIES = "properties"  # Item 2
    LEGAL_PROCEEDINGS = "legal_proceedings"  # Item 3
    MINE_SAFETY = "mine_safety"  # Item 4

    # 10-K Sections (Part II)
    MARKET_INFO = "market_info"  # Item 5
    SELECTED_FINANCIAL = "selected_financial"  # Item 6
    MDA = "mda"  # Item 7 - Management's Discussion and Analysis
    QUANTITATIVE_QUALITATIVE = "quantitative_qualitative"  # Item 7A
    FINANCIAL_STATEMENTS = "financial_statements"  # Item 8
    CHANGES_ACCOUNTANTS = "changes_accountants"  # Item 9
    CONTROLS_PROCEDURES = "controls_procedures"  # Item 9A
    OTHER = "other"  # Item 9B

    # 10-Q Specific Sections
    LEGAL_PROCEEDINGS_Q = "legal_proceedings_q"  # Part II, Item 1
    RISK_FACTORS_Q = "risk_factors_q"  # Part II, Item 1A


@dataclass(frozen=True)
class Filing:
    """
    Represents a single SEC filing.

    Contains metadata about the filing and download location.
    """

    accession_number: str  # "0001234567-23-000001" (unique ID)
    cik: str  # 10-digit CIK
    ticker: str | None  # Stock ticker if available
    company_name: str
    filing_type: FilingType
    filing_date: datetime  # Date filed with SEC
    period_of_report: datetime  # Period end date
    primary_document: str  # Filename of main document
    filing_url: str  # Full URL to filing

    @property
    def accession_number_no_dashes(self) -> str:
        """Return accession number without dashes (for URLs)."""
        return self.accession_number.replace("-", "")

    @property
    def filing_directory_url(self) -> str:
        """Return URL to the filing directory on EDGAR."""
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession_number_no_dashes}"

    @property
    def primary_document_url(self) -> str:
        """Return URL to the primary document."""
        return f"{self.filing_directory_url}/{self.primary_document}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accession_number": self.accession_number,
            "cik": self.cik,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_type": self.filing_type.value,
            "filing_date": self.filing_date.isoformat(),
            "period_of_report": self.period_of_report.isoformat(),
            "primary_document": self.primary_document,
            "filing_url": self.filing_url,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Filing":
        """Create from dictionary."""
        return cls(
            accession_number=d["accession_number"],
            cik=d["cik"],
            ticker=d.get("ticker"),
            company_name=d["company_name"],
            filing_type=FilingType(d["filing_type"]),
            filing_date=datetime.fromisoformat(d["filing_date"]),
            period_of_report=datetime.fromisoformat(d["period_of_report"]),
            primary_document=d["primary_document"],
            filing_url=d["filing_url"],
        )


@dataclass
class FilingChunk:
    """
    A chunk of text from a parsed SEC filing.

    Contains the text content and metadata for RAG indexing.
    """

    chunk_id: str  # Unique ID: {accession_number}_{section}_{chunk_index}
    filing_accession: str  # Parent filing accession number
    section: FilingSection | str  # Section this chunk came from
    chunk_index: int  # Index within section (0-based)
    content: str  # The actual text content
    char_start: int  # Start position in original section text
    char_end: int  # End position in original section text

    # Inherited metadata from filing
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_length(self) -> int:
        """Return length of content in characters."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "filing_accession": self.filing_accession,
            "section": self.section if isinstance(self.section, str) else self.section.value,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": self.metadata,
        }


@dataclass
class FilingCheckpoint:
    """
    Checkpoint for SEC filing ingestion progress.

    Tracks which companies and filings have been processed
    to support resume capability.
    """

    # Identifiers
    source: str = "sec_edgar"
    started_at: datetime = field(default_factory=datetime.now)

    # Company-level progress
    current_company_index: int = 0  # Index in S&P 500 list
    current_company_cik: str | None = None
    total_companies: int = 0

    # Filing-level progress within current company
    current_filing_index: int = 0
    filings_for_current_company: int = 0

    # Overall counters
    companies_processed: int = 0
    filings_downloaded: int = 0
    filings_parsed: int = 0
    chunks_indexed: int = 0

    # Error tracking
    error_count: int = 0
    skipped_count: int = 0
    last_error: str | None = None

    # State
    state: str = "initialized"  # initialized, downloading, parsing, indexing, complete, failed

    # Timestamps
    last_updated_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "started_at": self.started_at.isoformat(),
            "current_company_index": self.current_company_index,
            "current_company_cik": self.current_company_cik,
            "total_companies": self.total_companies,
            "current_filing_index": self.current_filing_index,
            "filings_for_current_company": self.filings_for_current_company,
            "companies_processed": self.companies_processed,
            "filings_downloaded": self.filings_downloaded,
            "filings_parsed": self.filings_parsed,
            "chunks_indexed": self.chunks_indexed,
            "error_count": self.error_count,
            "skipped_count": self.skipped_count,
            "last_error": self.last_error,
            "state": self.state,
            "last_updated_at": self.last_updated_at.isoformat() if self.last_updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FilingCheckpoint":
        """Create from dictionary."""
        return cls(
            source=d.get("source", "sec_edgar"),
            started_at=datetime.fromisoformat(d["started_at"]),
            current_company_index=d.get("current_company_index", 0),
            current_company_cik=d.get("current_company_cik"),
            total_companies=d.get("total_companies", 0),
            current_filing_index=d.get("current_filing_index", 0),
            filings_for_current_company=d.get("filings_for_current_company", 0),
            companies_processed=d.get("companies_processed", 0),
            filings_downloaded=d.get("filings_downloaded", 0),
            filings_parsed=d.get("filings_parsed", 0),
            chunks_indexed=d.get("chunks_indexed", 0),
            error_count=d.get("error_count", 0),
            skipped_count=d.get("skipped_count", 0),
            last_error=d.get("last_error"),
            state=d.get("state", "initialized"),
            last_updated_at=(
                datetime.fromisoformat(d["last_updated_at"]) if d.get("last_updated_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None
            ),
        )

    @property
    def is_complete(self) -> bool:
        """Check if ingestion is complete."""
        return self.state == "complete"

    @property
    def is_failed(self) -> bool:
        """Check if ingestion failed."""
        return self.state == "failed"

    @property
    def is_resumable(self) -> bool:
        """Check if this checkpoint can be resumed."""
        return self.state not in ("complete", "failed", "initialized")

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage based on companies processed."""
        if self.total_companies == 0:
            return 0.0
        return (self.companies_processed / self.total_companies) * 100

    def mark_downloading(self, company_cik: str, company_index: int) -> None:
        """Mark state as downloading for a company."""
        self.state = "downloading"
        self.current_company_cik = company_cik
        self.current_company_index = company_index
        self.current_filing_index = 0
        self.last_updated_at = datetime.now()

    def mark_parsing(self) -> None:
        """Mark state as parsing filings."""
        self.state = "parsing"
        self.last_updated_at = datetime.now()

    def mark_indexing(self) -> None:
        """Mark state as indexing chunks."""
        self.state = "indexing"
        self.last_updated_at = datetime.now()

    def update_filing_progress(
        self,
        filings_downloaded: int = 0,
        filings_parsed: int = 0,
        chunks_indexed: int = 0,
        errors: int = 0,
    ) -> None:
        """Update filing-level progress counters."""
        self.filings_downloaded += filings_downloaded
        self.filings_parsed += filings_parsed
        self.chunks_indexed += chunks_indexed
        self.error_count += errors
        self.last_updated_at = datetime.now()

    def complete_company(self) -> None:
        """Mark current company as complete."""
        self.companies_processed += 1
        self.current_filing_index = 0
        self.last_updated_at = datetime.now()

    def mark_complete(self) -> None:
        """Mark ingestion as complete."""
        self.state = "complete"
        self.completed_at = datetime.now()
        self.last_updated_at = self.completed_at

    def mark_failed(self, error_message: str) -> None:
        """Mark ingestion as failed."""
        self.state = "failed"
        self.last_error = error_message
        self.last_updated_at = datetime.now()
