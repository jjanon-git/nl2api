"""
Evalkit Packs

Registry and factory for evaluation packs.

Available packs:
- NL2APIPack: Tool-calling LLM evaluation (syntax, logic, execution, semantics)
- RAGPack: RAG system evaluation (retrieval, faithfulness, relevance, citations)

Usage:
    from src.evalkit.packs import get_pack

    # Factory function (recommended)
    pack = get_pack("nl2api")
    pack = get_pack("rag")

    # Or import packs directly from their application modules
    from src.nl2api.evaluation import NL2APIPack
    from src.rag.evaluation import RAGPack
"""

from __future__ import annotations

from typing import Any


def _get_nl2api_pack():
    """Lazy import NL2APIPack to avoid circular imports."""
    from src.nl2api.evaluation import NL2APIPack

    return NL2APIPack


def _get_rag_pack():
    """Lazy import RAGPack to avoid circular imports."""
    from src.rag.evaluation import RAGPack

    return RAGPack


# Registry of available packs (lazy loaded)
_PACK_FACTORIES: dict[str, callable] = {
    "nl2api": _get_nl2api_pack,
    "rag": _get_rag_pack,
}


def get_pack(name: str, **kwargs) -> Any:
    """
    Get an evaluation pack by name.

    Args:
        name: Pack name ("nl2api" or "rag")
        **kwargs: Configuration passed to pack constructor

    Returns:
        Instantiated evaluation pack

    Raises:
        ValueError: If pack name is not recognized
    """
    if name not in _PACK_FACTORIES:
        available = ", ".join(_PACK_FACTORIES.keys())
        raise ValueError(f"Unknown pack: {name}. Available: {available}")

    pack_class = _PACK_FACTORIES[name]()
    return pack_class(**kwargs)


def get_available_packs() -> dict[str, type]:
    """
    Get registry of available pack classes.

    Returns:
        Dict mapping pack names to their classes.
    """
    return {name: factory() for name, factory in _PACK_FACTORIES.items()}


# For backwards compatibility
PACKS = get_available_packs  # noqa: N816


__all__ = [
    "get_pack",
    "get_available_packs",
]
