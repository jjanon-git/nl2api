"""
HyDE (Hypothetical Document Embeddings) Query Expansion

Generates hypothetical answers to queries to improve retrieval.
The hypothetical answer's embedding often aligns better with
actual documents than the query's embedding.

Reference: https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from src.evalkit.common.telemetry import get_tracer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client used by HyDE."""

    async def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text completion."""
        ...


class HyDEExpander:
    """
    Generate hypothetical document for query expansion.

    HyDE improves retrieval by embedding a hypothetical answer
    instead of the raw query. This helps when query vocabulary
    differs from document vocabulary.

    Example:
        Query: "What risks does Apple face?"
        Hypothetical: "Apple faces risks including supply chain disruption,
                      regulatory changes in key markets, and currency fluctuations..."

    The hypothetical answer's embedding is closer to actual SEC filing text.
    """

    HYDE_PROMPT = """Generate a brief, factual answer to this SEC filing question.
Write as if quoting from an actual SEC filing (10-K, 10-Q).
Be specific and use financial terminology. Keep it under 150 words.

Question: {query}

Hypothetical answer from SEC filing:"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str | None = None,
    ):
        """
        Initialize HyDE expander.

        Args:
            llm_client: LLM client for generating hypothetical answers
            prompt_template: Optional custom prompt template (must have {query} placeholder)
        """
        self._client = llm_client
        self._prompt_template = prompt_template or self.HYDE_PROMPT

    async def expand(self, query: str) -> str:
        """
        Generate hypothetical answer to use for embedding.

        Args:
            query: Original user query

        Returns:
            Hypothetical answer text to embed instead of query
        """
        with tracer.start_as_current_span("hyde.expand") as span:
            span.set_attribute("hyde.query_length", len(query))

            prompt = self._prompt_template.format(query=query)
            hypothetical = await self._client.generate(prompt, max_tokens=200)

            span.set_attribute("hyde.hypothetical_length", len(hypothetical))
            logger.debug(f"HyDE expansion: '{query[:50]}...' -> '{hypothetical[:50]}...'")

            return hypothetical


class SimpleAnthropicClient:
    """Simple Anthropic client for HyDE generation."""

    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        self._model = model
        self._client = None

    async def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import os

            from dotenv import load_dotenv

            load_dotenv()  # Ensure API key is loaded
            import anthropic

            # Support both ANTHROPIC_API_KEY and NL2API_ANTHROPIC_API_KEY
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("NL2API_ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Anthropic API key required for HyDE. "
                    "Set ANTHROPIC_API_KEY or NL2API_ANTHROPIC_API_KEY"
                )
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text completion using Anthropic."""
        client = await self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class SimpleOpenAIClient:
    """Simple OpenAI client for HyDE generation."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._client = None

    async def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            import os

            from dotenv import load_dotenv

            load_dotenv()  # Ensure API key is loaded
            import openai

            # Support both OPENAI_API_KEY and NL2API_OPENAI_API_KEY
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("NL2API_OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OpenAI API key required for HyDE. Set OPENAI_API_KEY or NL2API_OPENAI_API_KEY"
                )
            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    async def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text completion using OpenAI."""
        client = await self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


def create_hyde_expander(
    provider: str = "anthropic",
    model: str | None = None,
) -> HyDEExpander:
    """
    Factory to create HyDE expander with appropriate LLM client.

    Args:
        provider: LLM provider ("anthropic" or "openai")
        model: Optional model override

    Returns:
        Configured HyDEExpander
    """
    if provider == "anthropic":
        client = SimpleAnthropicClient(model=model or "claude-3-5-haiku-latest")
    elif provider == "openai":
        client = SimpleOpenAIClient(model=model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")

    return HyDEExpander(llm_client=client)
