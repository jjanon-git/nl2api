"""Unit tests for company mappings loader."""

import pytest
from src.nl2api.resolution.mappings import (
    load_mappings,
    get_ric_for_company,
    get_ric_for_ticker,
    get_all_known_names,
)


def test_load_mappings():
    """Test that mappings are loaded correctly."""
    mappings = load_mappings()
    assert "mappings" in mappings
    assert "tickers" in mappings
    assert len(mappings["mappings"]) > 0
    assert len(mappings["tickers"]) > 0


def test_get_ric_for_company():
    """Test RIC lookup by company name and aliases."""
    # Test primary name
    assert get_ric_for_company("apple") == "AAPL.O"
    
    # Test alias
    assert get_ric_for_company("google") == "GOOGL.O"
    assert get_ric_for_company("jpmorgan chase") == "JPM.N"
    
    # Test unknown
    assert get_ric_for_company("nonexistent company") is None


def test_get_ric_for_ticker():
    """Test RIC lookup by ticker."""
    assert get_ric_for_ticker("AAPL") == "AAPL.O"
    assert get_ric_for_ticker("MSFT") == "MSFT.O"
    assert get_ric_for_ticker("JPM") == "JPM.N"
    
    # Case insensitivity
    assert get_ric_for_ticker("aapl") == "AAPL.O"
    
    # Test unknown
    assert get_ric_for_ticker("XYZABC") is None


def test_get_all_known_names():
    """Test retrieving all known names and aliases."""
    names = get_all_known_names()
    assert "apple" in names
    assert "apple inc" in names
    assert "google" in names
    assert len(names) > 100
