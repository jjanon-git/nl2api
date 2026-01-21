"""Company name to RIC mappings loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAPPINGS_FILE = Path(__file__).parent / "data" / "company_mappings.json"
_mappings_cache: dict[str, Any] | None = None


def load_mappings() -> dict[str, Any]:
    """Load company mappings from JSON file."""
    global _mappings_cache

    if _mappings_cache is not None:
        return _mappings_cache

    if not _MAPPINGS_FILE.exists():
        logger.warning(f"Mappings file not found: {_MAPPINGS_FILE}")
        return {"mappings": {}, "tickers": {}}

    try:
        with open(_MAPPINGS_FILE) as f:
            _mappings_cache = json.load(f)
    except Exception as e:
        logger.error(f"Error loading mappings file: {e}")
        return {"mappings": {}, "tickers": {}}

    logger.info(f"Loaded {_mappings_cache.get('company_count', 0)} company mappings")
    return _mappings_cache


def get_ric_for_company(name: str) -> str | None:
    """
    Get RIC for a company name.

    Checks primary name, then aliases.

    Args:
        name: Normalized company name (lowercase, stripped)

    Returns:
        RIC if found, None otherwise
    """
    mappings = load_mappings()

    # Direct lookup
    if name in mappings["mappings"]:
        return mappings["mappings"][name]["ric"]

    # Check aliases
    for company, data in mappings["mappings"].items():
        if name in data.get("aliases", []):
            return data["ric"]

    return None


def get_ric_for_ticker(ticker: str) -> str | None:
    """
    Get RIC for a ticker symbol.

    Args:
        ticker: Ticker symbol (e.g., "AAPL")

    Returns:
        RIC if found, None otherwise
    """
    mappings = load_mappings()
    return mappings.get("tickers", {}).get(ticker.upper())


def get_all_known_names() -> set[str]:
    """Get all known company names and aliases for pattern matching."""
    mappings = load_mappings()
    names = set()

    for company, data in mappings.get("mappings", {}).items():
        names.add(company)
        names.update(data.get("aliases", []))

    return names
