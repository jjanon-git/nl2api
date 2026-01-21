"""Unit tests for OpenFIGI integration."""

import pytest
from unittest.mock import patch, MagicMock
from src.nl2api.resolution.openfigi import resolve_via_openfigi


@pytest.mark.asyncio
async def test_resolve_via_openfigi_success():
    """Test successful resolution via OpenFIGI."""
    mock_response_data = [
        {
            "data": [
                {
                    "ticker": "AAPL",
                    "exchCode": "US",
                    "name": "APPLE INC",
                }
            ]
        }
    ]
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = pytest.importorskip("unittest.mock").AsyncMock(return_value=mock_response_data)
        mock_resp.__aenter__.return_value = mock_resp
        mock_post.return_value = mock_resp
        
        result = await resolve_via_openfigi("AAPL")
        
        assert result is not None
        assert result["found"] is True
        assert result["identifier"] == "AAPL.O"
        assert result["source"] == "openfigi"


@pytest.mark.asyncio
async def test_resolve_via_openfigi_not_found():
    """Test resolution when entity is not found."""
    mock_response_data = [{"error": "No identifier found"}]
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = pytest.importorskip("unittest.mock").AsyncMock(return_value=mock_response_data)
        mock_resp.__aenter__.return_value = mock_resp
        mock_post.return_value = mock_resp
        
        result = await resolve_via_openfigi("UNKNOWN")
        
        assert result is None


@pytest.mark.asyncio
async def test_resolve_via_openfigi_error():
    """Test resolution when API returns an error."""
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__aenter__.return_value = mock_resp
        mock_post.return_value = mock_resp
        
        result = await resolve_via_openfigi("AAPL")
        
        assert result is None
