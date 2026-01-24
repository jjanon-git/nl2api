"""
SEC EDGAR API Client

Rate-limited client for accessing SEC EDGAR filings.
Respects SEC's 10 requests/second rate limit.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from src.nl2api.ingestion.errors import DownloadError
from src.nl2api.ingestion.sec_filings.config import (
    SEC_ARCHIVES_BASE_URL,
    SEC_COMPANY_SUBMISSIONS_URL,
    SECFilingConfig,
)
from src.nl2api.ingestion.sec_filings.models import Filing, FilingType

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Rate limiter for async HTTP requests."""

    def __init__(self, rate_per_second: float):
        """
        Initialize rate limiter.

        Args:
            rate_per_second: Maximum requests per second
        """
        self._rate = rate_per_second
        self._min_interval = 1.0 / rate_per_second
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made within rate limit."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request = asyncio.get_event_loop().time()


class SECEdgarClient:
    """
    Client for SEC EDGAR API.

    Handles:
    - Company filing index retrieval
    - Filing download with streaming
    - Rate limiting (10 req/sec)
    - Retry with exponential backoff
    """

    def __init__(self, config: SECFilingConfig | None = None):
        """
        Initialize SEC EDGAR client.

        Args:
            config: Configuration (uses defaults if not provided)
        """
        self._config = config or SECFilingConfig()
        self._rate_limiter = AsyncRateLimiter(self._config.rate_limit_per_second)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SECEdgarClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self._config.user_agent},
            timeout=httpx.Timeout(self._config.download_timeout_seconds),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with SECEdgarClient()' context manager.")
        return self._client

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        **kwargs,
    ) -> httpx.Response:
        """
        Make HTTP request with rate limiting and retry.

        Args:
            method: HTTP method
            url: Request URL
            max_retries: Maximum retry attempts
            **kwargs: Additional request arguments

        Returns:
            HTTP response

        Raises:
            DownloadError: If request fails after retries
        """
        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(max_retries):
            await self._rate_limiter.acquire()

            try:
                response = await client.request(method, url, **kwargs)

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited by SEC EDGAR. Waiting {wait_time}s before retry.")
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    # Server error - retry with backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"SEC EDGAR server error {response.status_code}. Waiting {wait_time}s.")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response

            except httpx.TimeoutException as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Request timeout. Retry {attempt + 1}/{max_retries} after {wait_time}s.")
                await asyncio.sleep(wait_time)

            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    last_error = e
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise DownloadError(
                        f"HTTP error {e.response.status_code} for {url}",
                        source="sec_edgar",
                        url=url,
                        status_code=e.response.status_code,
                    ) from e

        raise DownloadError(
            f"Failed to fetch {url} after {max_retries} retries",
            source="sec_edgar",
            url=url,
        ) from last_error

    async def get_company_filings(
        self,
        cik: str,
        filing_types: list[FilingType] | None = None,
        after_date: datetime | None = None,
    ) -> list[Filing]:
        """
        Get list of filings for a company.

        Args:
            cik: Company CIK (will be zero-padded to 10 digits)
            filing_types: Filter by filing types (default: 10-K, 10-Q)
            after_date: Only return filings after this date

        Returns:
            List of Filing objects
        """
        # Zero-pad CIK to 10 digits
        cik_padded = cik.zfill(10)

        # Default filing types
        if filing_types is None:
            filing_types = self._config.get_filing_types()
        filing_type_values = {ft.value for ft in filing_types}

        # Fetch company submissions
        url = SEC_COMPANY_SUBMISSIONS_URL.format(cik=cik_padded)
        logger.debug(f"Fetching company submissions: {url}")

        response = await self._request_with_retry("GET", url)
        data = response.json()

        # Extract company info
        company_name = data.get("name", "Unknown")
        tickers = data.get("tickers", [])
        ticker = tickers[0] if tickers else None

        # Parse filings from recent filings
        filings = []
        recent = data.get("filings", {}).get("recent", {})

        if not recent:
            logger.warning(f"No filings found for CIK {cik}")
            return []

        # Extract arrays from recent filings
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        forms = recent.get("form", [])
        primary_documents = recent.get("primaryDocument", [])

        for i in range(len(accession_numbers)):
            form = forms[i] if i < len(forms) else None
            if form not in filing_type_values:
                continue

            filing_date_str = filing_dates[i] if i < len(filing_dates) else None
            if not filing_date_str:
                continue

            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")

            # Filter by date
            if after_date and filing_date < after_date:
                continue

            report_date_str = report_dates[i] if i < len(report_dates) else filing_date_str
            try:
                period_of_report = datetime.strptime(report_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                period_of_report = filing_date

            accession = accession_numbers[i]
            primary_doc = primary_documents[i] if i < len(primary_documents) else None

            if not primary_doc:
                continue

            # Build filing URL
            accession_no_dashes = accession.replace("-", "")
            filing_url = f"{SEC_ARCHIVES_BASE_URL}/{cik_padded}/{accession_no_dashes}/{primary_doc}"

            try:
                filing_type = FilingType(form)
            except ValueError:
                continue

            filings.append(
                Filing(
                    accession_number=accession,
                    cik=cik_padded,
                    ticker=ticker,
                    company_name=company_name,
                    filing_type=filing_type,
                    filing_date=filing_date,
                    period_of_report=period_of_report,
                    primary_document=primary_doc,
                    filing_url=filing_url,
                )
            )

        logger.info(f"Found {len(filings)} filings for {company_name} (CIK: {cik})")
        return filings

    async def download_filing(
        self,
        filing: Filing,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Download a filing's primary document.

        Args:
            filing: Filing to download
            output_dir: Output directory (default: config download_dir)

        Returns:
            Path to downloaded file
        """
        output_dir = output_dir or self._config.download_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for company
        company_dir = output_dir / filing.cik
        company_dir.mkdir(parents=True, exist_ok=True)

        # Output filename
        output_file = company_dir / f"{filing.accession_number}_{filing.primary_document}"

        # Skip if already downloaded
        if output_file.exists():
            logger.debug(f"Filing already downloaded: {output_file}")
            return output_file

        logger.info(f"Downloading filing: {filing.filing_url}")

        # Download with streaming
        response = await self._request_with_retry("GET", filing.filing_url)

        # Write to file
        output_file.write_bytes(response.content)
        logger.info(f"Downloaded filing to: {output_file}")

        return output_file

    async def download_filing_streaming(
        self,
        filing: Filing,
        output_dir: Path | None = None,
        chunk_size: int = 65536,
    ) -> Path:
        """
        Download a filing's primary document with streaming.

        Better for large files (reduces memory usage).

        Args:
            filing: Filing to download
            output_dir: Output directory
            chunk_size: Download chunk size in bytes

        Returns:
            Path to downloaded file
        """
        output_dir = output_dir or self._config.download_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for company
        company_dir = output_dir / filing.cik
        company_dir.mkdir(parents=True, exist_ok=True)

        # Output filename
        output_file = company_dir / f"{filing.accession_number}_{filing.primary_document}"

        # Skip if already downloaded
        if output_file.exists():
            logger.debug(f"Filing already downloaded: {output_file}")
            return output_file

        logger.info(f"Downloading filing (streaming): {filing.filing_url}")

        await self._rate_limiter.acquire()
        client = self._get_client()

        async with client.stream("GET", filing.filing_url) as response:
            response.raise_for_status()

            with open(output_file, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    f.write(chunk)

        logger.info(f"Downloaded filing to: {output_file}")
        return output_file


def load_sp500_companies(data_path: Path | None = None) -> list[dict[str, Any]]:
    """
    Load S&P 500 company list from data file.

    Args:
        data_path: Path to sp500.json (default: data/tickers/sp500.json)

    Returns:
        List of company dicts with ticker, cik, name
    """
    if data_path is None:
        data_path = Path("data/tickers/sp500.json")

    if not data_path.exists():
        raise FileNotFoundError(f"S&P 500 data file not found: {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    return data.get("companies", [])


def filter_filings_by_date(
    filings: list[Filing],
    years_back: int = 2,
    reference_date: datetime | None = None,
) -> list[Filing]:
    """
    Filter filings to only include those within the specified time range.

    Args:
        filings: List of filings to filter
        years_back: Number of years back to include
        reference_date: Reference date (default: today)

    Returns:
        Filtered list of filings
    """
    if reference_date is None:
        reference_date = datetime.now()

    cutoff_date = reference_date - timedelta(days=365 * years_back)

    return [f for f in filings if f.filing_date >= cutoff_date]
