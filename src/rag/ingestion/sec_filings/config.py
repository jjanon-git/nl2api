"""
SEC Filing Ingestion Configuration

Defines settings for SEC EDGAR filing ingestion pipeline.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.rag.ingestion.sec_filings.models import FilingType


class SECFilingConfig(BaseSettings):
    """
    Configuration for SEC EDGAR filing ingestion pipeline.

    SEC EDGAR Fair Access Policy:
    - Maximum 10 requests per second
    - User-Agent header MUST include company/app name and contact email
    - Prefer bulk data downloads when available
    - Use off-peak hours (nights/weekends) for large batch jobs

    Reference: https://www.sec.gov/os/accessing-edgar-data
    """

    model_config = SettingsConfigDict(
        env_prefix="SEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SEC EDGAR API settings
    # IMPORTANT: Update this with your actual contact info before running
    user_agent: str = Field(
        default="NL2API Research Bot contact@example.com",
        description="User-Agent header required by SEC EDGAR (must include contact info)",
    )
    # Default to 5 req/sec (conservative, half of SEC's 10 req/sec limit)
    # This leaves headroom and is more respectful of SEC resources
    rate_limit_per_second: float = Field(
        default=5.0,
        description="SEC EDGAR rate limit (max 10/sec, default 5 for safety margin)",
        ge=1.0,
        le=10.0,
    )

    # Filing selection
    filing_types: list[str] = Field(
        default=["10-K", "10-Q"],
        description="Filing types to download",
    )
    years_back: int = Field(
        default=2,
        description="Number of years of filings to download",
        ge=1,
        le=10,
    )

    # Processing settings
    batch_size: int = Field(
        default=50,
        description="Number of chunks per batch for embedding/indexing",
        ge=10,
        le=500,
    )
    # Conservative: 1 concurrent download to avoid hammering SEC servers
    # The rate limiter handles request spacing, but serial downloads are safer
    max_concurrent_downloads: int = Field(
        default=1,
        description="Maximum concurrent filing downloads (keep low to respect SEC servers)",
        ge=1,
        le=5,
    )
    # Add delay between companies to further reduce load
    inter_company_delay_seconds: float = Field(
        default=2.0,
        description="Delay between processing different companies (seconds)",
        ge=0.0,
        le=60.0,
    )
    checkpoint_interval: int = Field(
        default=10,
        description="Save checkpoint every N companies",
        ge=1,
    )
    max_errors: int = Field(
        default=100,
        description="Maximum errors before aborting ingestion",
        ge=1,
    )

    # Chunking settings
    chunk_size: int = Field(
        default=4000,
        description="Target chunk size in characters",
        ge=500,
        le=10000,
    )
    chunk_overlap: int = Field(
        default=800,
        description="Overlap between chunks in characters",
        ge=0,
        le=2000,
    )
    min_chunk_size: int = Field(
        default=200,
        description="Minimum chunk size (smaller chunks are merged or dropped)",
        ge=50,
        le=1000,
    )
    contextual_chunking: bool = Field(
        default=True,
        description="Prepend document context (company, filing type, section) to each chunk",
    )

    # Directory paths
    data_dir: Path = Field(
        default=Path("data/sec_filings"),
        description="Directory for downloaded files and checkpoints",
    )

    # Download settings
    download_timeout_seconds: int = Field(
        default=120,
        description="Timeout for downloading individual filings",
        ge=30,
    )

    # Embedder settings
    embedder_type: str = Field(
        default="local",
        description="Embedder type: 'local' (sentence-transformers) or 'openai'",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
        ge=1,
        le=100,
    )

    def ensure_data_dir(self) -> Path:
        """Create data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoint files."""
        path = self.data_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def download_dir(self) -> Path:
        """Directory for downloaded filings."""
        path = self.data_dir / "downloads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def parsed_dir(self) -> Path:
        """Directory for parsed filing sections."""
        path = self.data_dir / "parsed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def error_log_dir(self) -> Path:
        """Directory for error logs."""
        path = self.data_dir / "errors"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def rate_limit_delay(self) -> float:
        """Delay between requests in seconds to respect rate limit."""
        return 1.0 / self.rate_limit_per_second

    def get_filing_types(self) -> list[FilingType]:
        """Get filing types as enum values."""
        result = []
        for ft in self.filing_types:
            try:
                result.append(FilingType(ft))
            except ValueError:
                # Try with amendments
                if not ft.endswith("/A"):
                    try:
                        result.append(FilingType(f"{ft}/A"))
                    except ValueError:
                        pass
        return result


# SEC EDGAR API endpoints
SEC_EDGAR_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
