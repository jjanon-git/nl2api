"""
Entity Ingestion Configuration

Defines settings for entity data ingestion from external sources.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntityIngestionConfig(BaseSettings):
    """Configuration for entity ingestion pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="ENTITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GLEIF source URLs
    gleif_download_url: str = Field(
        default="https://leidata-preview.gleif.org/api/v2/golden-copies/publishes/latest",
        description="GLEIF golden copy metadata API endpoint",
    )

    # SEC EDGAR source URLs
    sec_tickers_url: str = Field(
        default="https://www.sec.gov/files/company_tickers.json",
        description="SEC ticker-CIK mapping file",
    )
    sec_exchanges_url: str = Field(
        default="https://www.sec.gov/files/company_tickers_exchange.json",
        description="SEC ticker-exchange mapping file",
    )

    # Processing settings
    batch_size: int = Field(
        default=50000,
        description="Number of records per batch for COPY protocol",
        ge=1000,
        le=500000,
    )
    max_errors: int = Field(
        default=1000,
        description="Maximum errors before aborting ingestion",
        ge=1,
    )
    checkpoint_interval: int = Field(
        default=100000,
        description="Save checkpoint every N records",
        ge=10000,
    )

    # Directory paths
    data_dir: Path = Field(
        default=Path("data/entity_ingestion"),
        description="Directory for downloaded files and checkpoints",
    )

    # Refresh settings (disabled by default)
    refresh_enabled: bool = Field(
        default=False,
        description="Enable automatic refresh of entity data",
    )
    refresh_interval_days: int = Field(
        default=30,
        description="Days between automatic refreshes",
        ge=1,
    )

    # Download settings
    download_timeout_seconds: int = Field(
        default=600,
        description="Timeout for downloading large files (10 minutes)",
        ge=60,
    )
    chunk_size: int = Field(
        default=65536,
        description="Chunk size for streaming downloads (64KB)",
        ge=4096,
    )

    # RIC generation settings
    default_exchange_suffix: str = Field(
        default=".O",
        description="Default exchange suffix when unknown (NASDAQ)",
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
        """Directory for downloaded files."""
        path = self.data_dir / "downloads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def error_log_dir(self) -> Path:
        """Directory for error logs."""
        path = self.data_dir / "errors"
        path.mkdir(parents=True, exist_ok=True)
        return path


# Exchange suffix mapping for RIC generation
EXCHANGE_SUFFIX_MAP = {
    # US exchanges
    "NYSE": ".N",
    "NASDAQ": ".O",
    "AMEX": ".A",
    "BATS": ".Z",
    "ARCA": ".P",
    "N": ".N",
    "Q": ".O",
    "A": ".A",
    "Z": ".Z",
    "P": ".P",
    # International exchanges
    "LSE": ".L",
    "L": ".L",
    "TSE": ".T",
    "T": ".T",
    "XETRA": ".DE",
    "DE": ".DE",
    "EURONEXT": ".PA",
    "PA": ".PA",
    "HKSE": ".HK",
    "HK": ".HK",
    "SGX": ".SI",
    "SI": ".SI",
    "ASX": ".AX",
    "AX": ".AX",
    "TSX": ".TO",
    "TO": ".TO",
}


def generate_ric(ticker: str, exchange: str | None, default_suffix: str = ".O") -> str:
    """
    Generate RIC from ticker and exchange.

    Args:
        ticker: Stock ticker symbol
        exchange: Exchange code or name
        default_suffix: Default suffix if exchange unknown

    Returns:
        RIC (e.g., "AAPL.O" for Apple on NASDAQ)
    """
    if not ticker:
        return ""

    ticker = ticker.upper().strip()

    if not exchange:
        return f"{ticker}{default_suffix}"

    exchange = exchange.upper().strip()
    suffix = EXCHANGE_SUFFIX_MAP.get(exchange, default_suffix)

    return f"{ticker}{suffix}"
