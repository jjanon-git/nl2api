"""
OpenFIGI API - Re-export from canonical location.

The canonical implementation lives in src/evalkit/common/entity_resolution/openfigi.py.
This module re-exports for backwards compatibility.
"""

from src.evalkit.common.entity_resolution.openfigi import (
    OPENFIGI_URL,
    resolve_via_openfigi,
)

__all__ = ["OPENFIGI_URL", "resolve_via_openfigi"]
