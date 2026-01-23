"""
Evaluation Packs

Domain-specific evaluation logic for different use cases.

Available packs:
- NL2APIPack: Tool-calling LLM evaluation (syntax, logic, execution, semantics)
- RAGPack: RAG system evaluation (retrieval, faithfulness, relevance, citations)

Usage:
    from src.evaluation.packs import NL2APIPack, RAGPack, get_pack

    # Direct instantiation
    pack = RAGPack()
    evaluator = Evaluator(pack=pack)
    results = await evaluator.evaluate(test_case, system_output)

    # Factory function
    pack = get_pack("rag")
"""

from src.evaluation.packs.nl2api import NL2APIPack
from src.evaluation.packs.rag import RAGPack

__all__ = [
    "NL2APIPack",
    "RAGPack",
    "get_pack",
    "PACKS",
]

# Registry of available packs
PACKS: dict[str, type] = {
    "nl2api": NL2APIPack,
    "rag": RAGPack,
}


def get_pack(name: str, **kwargs):
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
    if name not in PACKS:
        available = ", ".join(PACKS.keys())
        raise ValueError(f"Unknown pack: {name}. Available: {available}")
    return PACKS[name](**kwargs)
