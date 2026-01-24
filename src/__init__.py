"""NL2API - Natural Language to API translation for LSEG financial data."""

__version__ = "0.1.0"

# Import evalkit to make it available as an attribute (needed for patching)
# Note: Other submodules are imported lazily to avoid circular imports
from src import evalkit  # noqa: F401
