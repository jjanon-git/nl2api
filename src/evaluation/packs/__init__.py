"""
Compatibility shim for src.evaluation.packs -> application packs migration.

TODO: Remove after validation period (Stage 2 of codebase separation).

Packs have been reorganized:
- NL2APIPack: src.nl2api.evaluation.pack
- RAGPack: src.rag.evaluation.pack
- Registry: src.evalkit.packs
"""

# Re-export packs from their new application locations
# Re-export registry from evalkit
from src.evalkit.packs import get_pack
from src.nl2api.evaluation import NL2APIPack
from src.rag.evaluation import RAGPack

__all__ = [
    "NL2APIPack",
    "RAGPack",
    "get_pack",
    "PACKS",
]

# Registry of available packs (for backwards compatibility)
PACKS: dict[str, type] = {
    "nl2api": NL2APIPack,
    "rag": RAGPack,
}
