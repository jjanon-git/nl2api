"""Unit tests for SEC EDGAR client."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.nl2api.ingestion.sec_filings.client import (
    AsyncRateLimiter,
    SECEdgarClient,
    filter_filings_by_date,
    load_sp500_companies,
)
from src.nl2api.ingestion.sec_filings.config import SECFilingConfig
from src.nl2api.ingestion.sec_filings.models import Filing, FilingType


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter can be initialized."""
        limiter = AsyncRateLimiter(rate_per_second=10.0)
        assert limiter._rate == 10.0
        assert limiter._min_interval == 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test acquiring rate limit slot."""
        limiter = AsyncRateLimiter(rate_per_second=100.0)  # Fast for testing

        # First acquire should be immediate
        await limiter.acquire()

        # Second acquire should also work
        await limiter.acquire()


class TestSECEdgarClient:
    """Tests for SECEdgarClient class."""

    @pytest.fixture
    def config(self) -> SECFilingConfig:
        """Create test configuration."""
        return SECFilingConfig(
            user_agent="Test Bot test@example.com",
            rate_limit_per_second=10.0,
        )

    @pytest.fixture
    def mock_company_response(self) -> dict:
        """Create mock company submissions response."""
        return {
            "cik": "0000320193",
            "name": "Apple Inc.",
            "tickers": ["AAPL"],
            "filings": {
                "recent": {
                    "accessionNumber": [
                        "0000320193-23-000077",
                        "0000320193-23-000064",
                    ],
                    "filingDate": [
                        "2023-11-03",
                        "2023-08-04",
                    ],
                    "reportDate": [
                        "2023-09-30",
                        "2023-07-01",
                    ],
                    "form": [
                        "10-K",
                        "10-Q",
                    ],
                    "primaryDocument": [
                        "aapl-20230930.htm",
                        "aapl-20230701.htm",
                    ],
                }
            },
        }

    @pytest.mark.asyncio
    async def test_client_context_manager(self, config: SECFilingConfig):
        """Test client can be used as context manager."""
        async with SECEdgarClient(config) as client:
            assert client._client is not None
        # After exit, client should be closed
        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_company_filings_parsing(
        self, config: SECFilingConfig, mock_company_response: dict
    ):
        """Test parsing of company filings response."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_company_response
            mock_response.raise_for_status = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_response)

            async with SECEdgarClient(config) as client:
                client._client = mock_client  # Inject mock

                filings = await client.get_company_filings(
                    cik="320193",
                    filing_types=[FilingType.FORM_10K, FilingType.FORM_10Q],
                )

                assert len(filings) == 2
                assert filings[0].ticker == "AAPL"
                assert filings[0].filing_type == FilingType.FORM_10K
                assert filings[1].filing_type == FilingType.FORM_10Q


class TestFilterFilingsByDate:
    """Tests for filter_filings_by_date function."""

    @pytest.fixture
    def sample_filings(self) -> list[Filing]:
        """Create sample filings for testing."""
        return [
            Filing(
                accession_number="0000320193-23-000077",
                cik="0000320193",
                ticker="AAPL",
                company_name="Apple Inc.",
                filing_type=FilingType.FORM_10K,
                filing_date=datetime(2023, 11, 3),
                period_of_report=datetime(2023, 9, 30),
                primary_document="aapl-20230930.htm",
                filing_url="https://example.com/filing1.htm",
            ),
            Filing(
                accession_number="0000320193-22-000077",
                cik="0000320193",
                ticker="AAPL",
                company_name="Apple Inc.",
                filing_type=FilingType.FORM_10K,
                filing_date=datetime(2022, 11, 4),
                period_of_report=datetime(2022, 9, 24),
                primary_document="aapl-20220924.htm",
                filing_url="https://example.com/filing2.htm",
            ),
            Filing(
                accession_number="0000320193-21-000077",
                cik="0000320193",
                ticker="AAPL",
                company_name="Apple Inc.",
                filing_type=FilingType.FORM_10K,
                filing_date=datetime(2021, 10, 29),
                period_of_report=datetime(2021, 9, 25),
                primary_document="aapl-20210925.htm",
                filing_url="https://example.com/filing3.htm",
            ),
        ]

    def test_filter_by_years(self, sample_filings: list[Filing]):
        """Test filtering filings by years back."""
        # Filter to 2 years from 2024-01-01
        filtered = filter_filings_by_date(
            sample_filings,
            years_back=2,
            reference_date=datetime(2024, 1, 1),
        )

        # Should include 2023 and 2022 filings
        assert len(filtered) == 2
        assert filtered[0].filing_date.year == 2023
        assert filtered[1].filing_date.year == 2022

    def test_filter_by_one_year(self, sample_filings: list[Filing]):
        """Test filtering to last year only."""
        filtered = filter_filings_by_date(
            sample_filings,
            years_back=1,
            reference_date=datetime(2024, 1, 1),
        )

        # Should only include 2023 filing
        assert len(filtered) == 1
        assert filtered[0].filing_date.year == 2023


class TestLoadSP500Companies:
    """Tests for load_sp500_companies function."""

    def test_load_from_existing_file(self, tmp_path: Path):
        """Test loading companies from file."""
        # Create test file
        data = {
            "companies": [
                {"ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc."},
                {"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corporation"},
            ]
        }
        test_file = tmp_path / "sp500.json"
        test_file.write_text(json.dumps(data))

        companies = load_sp500_companies(test_file)

        assert len(companies) == 2
        assert companies[0]["ticker"] == "AAPL"
        assert companies[1]["ticker"] == "MSFT"

    def test_load_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_sp500_companies(Path("/nonexistent/sp500.json"))

    def test_load_empty_companies(self, tmp_path: Path):
        """Test loading file with empty companies list."""
        data = {"companies": []}
        test_file = tmp_path / "sp500.json"
        test_file.write_text(json.dumps(data))

        companies = load_sp500_companies(test_file)
        assert companies == []
