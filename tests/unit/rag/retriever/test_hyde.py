"""Tests for HyDE (Hypothetical Document Embeddings) module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.retriever.hyde import HyDEExpander, create_hyde_expander


class TestHyDEExpander:
    """Tests for HyDEExpander class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.generate = AsyncMock(
            return_value="Apple faces risks including supply chain disruption."
        )
        return client

    @pytest.fixture
    def expander(self, mock_llm_client):
        """Create HyDEExpander with mock client."""
        return HyDEExpander(llm_client=mock_llm_client)

    @pytest.mark.asyncio
    async def test_expand_generates_hypothetical(self, expander, mock_llm_client):
        """Test that expand generates a hypothetical answer."""
        query = "What risks does Apple face?"
        result = await expander.expand(query)

        assert result == "Apple faces risks including supply chain disruption."
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        assert "What risks does Apple face?" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_expand_uses_custom_prompt(self, mock_llm_client):
        """Test that custom prompt template is used."""
        custom_prompt = "Answer this financial question: {query}\nAnswer:"
        expander = HyDEExpander(llm_client=mock_llm_client, prompt_template=custom_prompt)

        await expander.expand("What is revenue?")

        call_args = mock_llm_client.generate.call_args
        assert "Answer this financial question: What is revenue?" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_expand_passes_max_tokens(self, mock_llm_client):
        """Test that max_tokens is passed to LLM."""
        expander = HyDEExpander(llm_client=mock_llm_client)
        await expander.expand("Test query")

        call_args = mock_llm_client.generate.call_args
        assert call_args[1]["max_tokens"] == 200


class TestCreateHyDEExpander:
    """Tests for create_hyde_expander factory."""

    def test_create_anthropic_expander(self, monkeypatch):
        """Test creating Anthropic-based expander."""
        # Just test that the factory doesn't raise
        expander = create_hyde_expander(provider="anthropic")
        assert expander is not None
        assert expander._prompt_template is not None

    def test_create_openai_expander(self, monkeypatch):
        """Test creating OpenAI-based expander."""
        expander = create_hyde_expander(provider="openai")
        assert expander is not None

    def test_create_with_custom_model(self):
        """Test creating expander with custom model."""
        expander = create_hyde_expander(provider="anthropic", model="claude-3-opus")
        assert expander is not None

    def test_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_hyde_expander(provider="unknown")
