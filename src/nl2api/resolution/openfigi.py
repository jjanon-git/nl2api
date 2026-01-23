"""
OpenFIGI API integration for entity resolution.

OpenFIGI is a free API for mapping securities identifiers.
https://www.openfigi.com/api
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"


async def resolve_via_openfigi(
    query: str,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """
    Resolve a company/ticker via OpenFIGI API.

    Args:
        query: Company name or ticker
        api_key: Optional API key (higher rate limits)
        timeout: Request timeout

    Returns:
        Dict with identifier info if found
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key

    # Try as ticker first
    payload = [{"idType": "TICKER", "idValue": query.upper()}]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENFIGI_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and data[0].get("data"):
                        figi_data = data[0]["data"][0]
                        return {
                            "found": True,
                            "identifier": _convert_to_ric(figi_data),
                            "type": "equity",
                            "confidence": 0.9,
                            "source": "openfigi",
                            "raw": figi_data,
                        }
                else:
                    logger.debug(f"OpenFIGI API returned status {response.status}")
    except Exception as e:
        logger.warning(f"OpenFIGI lookup failed: {e}")

    return None


def _convert_to_ric(figi_data: dict[str, Any]) -> str:
    """Convert OpenFIGI data to Reuters RIC format."""
    ticker = figi_data.get("ticker", "")
    exchange = figi_data.get("exchCode", "")

    # Map common exchange codes to RIC suffixes
    # This is a simplified mapping
    exchange_map = {
        "US": ".O",  # NASDAQ
        "UN": ".N",  # NYSE
        "UW": ".O",  # NASDAQ
        "LN": ".L",  # London
        "JP": ".T",  # Tokyo
        "HK": ".HK",  # Hong Kong
    }

    suffix = exchange_map.get(exchange, ".O")
    return f"{ticker}{suffix}"
