"""
Unit tests for LLMJudge.

Tests the LLM-as-judge abstraction used by reference-free stages.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.contracts import LLMJudgeConfig
from src.evaluation.packs.rag.llm_judge import (
    ClaimVerificationResult,
    JudgeResult,
    LLMJudge,
)


@pytest.fixture
def config():
    """Create a LLMJudgeConfig."""
    return LLMJudgeConfig(
        model="claude-3-5-haiku-20241022",
        temperature=0.0,
        max_tokens=512,
        pass_threshold=0.7,
    )


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


class TestJudgeResultDataclass:
    """Tests for JudgeResult dataclass."""

    def test_judge_result_creation(self):
        """Create JudgeResult with all fields."""
        result = JudgeResult(
            score=0.85,
            passed=True,
            reasoning="Good response",
            raw_response='{"score": 0.85}',
            metrics={"claims": 3},
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.reasoning == "Good response"
        assert result.metrics["claims"] == 3

    def test_judge_result_default_metrics(self):
        """Default metrics is empty dict."""
        result = JudgeResult(
            score=0.5,
            passed=False,
            reasoning="Test",
            raw_response="",
        )

        assert result.metrics == {}


class TestClaimVerificationResult:
    """Tests for ClaimVerificationResult dataclass."""

    def test_claim_verification_supported(self):
        """Create supported claim result."""
        result = ClaimVerificationResult(
            claim="Python was created by Guido",
            supported=True,
            evidence="Guido van Rossum created Python",
            confidence=0.95,
        )

        assert result.claim == "Python was created by Guido"
        assert result.supported is True
        assert result.confidence == 0.95

    def test_claim_verification_unsupported(self):
        """Create unsupported claim result."""
        result = ClaimVerificationResult(
            claim="Python was created in 1492",
            supported=False,
        )

        assert result.supported is False
        assert result.evidence is None
        assert result.confidence == 1.0  # Default


class TestLLMJudgeInit:
    """Tests for LLMJudge initialization."""

    def test_init_with_config(self, config):
        """Initialize with config."""
        judge = LLMJudge(config=config)

        assert judge.config == config

    def test_init_default_config(self):
        """Initialize with default config."""
        judge = LLMJudge()

        assert judge.config is not None
        assert isinstance(judge.config, LLMJudgeConfig)

    def test_init_with_client(self, mock_anthropic_client, config):
        """Initialize with provided client."""
        judge = LLMJudge(config=config, llm_client=mock_anthropic_client)

        assert judge._llm_client is mock_anthropic_client


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_judge_response_valid_json(self, config):
        """Parse valid JSON response."""
        judge = LLMJudge(config=config)

        response = '{"score": 0.85, "reasoning": "Good answer"}'
        result = judge._parse_judge_response(response)

        assert result.score == 0.85
        assert result.reasoning == "Good answer"
        assert result.passed is True  # 0.85 > 0.7 threshold

    def test_parse_judge_response_json_in_text(self, config):
        """Extract JSON from surrounding text."""
        judge = LLMJudge(config=config)

        response = 'Here is my evaluation:\n{"score": 0.6, "reasoning": "Partial answer"}\nEnd.'
        result = judge._parse_judge_response(response)

        assert result.score == 0.6
        assert result.passed is False  # 0.6 < 0.7 threshold

    def test_parse_judge_response_invalid_json(self, config):
        """Handle invalid JSON gracefully."""
        judge = LLMJudge(config=config)

        response = "This is not JSON at all"
        result = judge._parse_judge_response(response)

        assert result.score == 0.5
        assert result.passed is False

    def test_parse_judge_response_clamps_score(self, config):
        """Clamp score to [0, 1] range."""
        judge = LLMJudge(config=config)

        response = '{"score": 1.5, "reasoning": "Too high"}'
        result = judge._parse_judge_response(response)

        assert result.score == 1.0

        response = '{"score": -0.5, "reasoning": "Too low"}'
        result = judge._parse_judge_response(response)

        assert result.score == 0.0


class TestClaimExtraction:
    """Tests for claim extraction parsing."""

    def test_parse_claims_json_array(self, config):
        """Parse JSON array of claims."""
        judge = LLMJudge(config=config)

        response = '["Claim one", "Claim two", "Claim three"]'
        claims = judge._parse_claims(response)

        assert len(claims) == 3
        assert "Claim one" in claims

    def test_parse_claims_json_in_text(self, config):
        """Extract JSON array from surrounding text."""
        judge = LLMJudge(config=config)

        response = 'Here are the claims:\n["Claim A", "Claim B"]\nDone.'
        claims = judge._parse_claims(response)

        assert len(claims) == 2

    def test_parse_claims_fallback_newlines(self, config):
        """Fall back to newline splitting."""
        judge = LLMJudge(config=config)

        response = "1. First claim about something\n2. Second claim about another thing\n3. Third claim here"
        claims = judge._parse_claims(response)

        assert len(claims) >= 2
        assert any("claim" in c.lower() for c in claims)

    def test_parse_claims_filters_short(self, config):
        """Filter out very short claims."""
        judge = LLMJudge(config=config)

        response = "- Yes\n- This is a longer claim that should be kept\n- No"
        claims = judge._parse_claims(response)

        # Short entries should be filtered
        assert all(len(c) > 10 for c in claims)


class TestClaimVerificationParsing:
    """Tests for claim verification response parsing."""

    def test_parse_verification_supported(self, config):
        """Parse supported verification."""
        judge = LLMJudge(config=config)

        response = '{"supported": true, "evidence": "Source text", "confidence": 0.9}'
        result = judge._parse_verification("Test claim", response)

        assert result.claim == "Test claim"
        assert result.supported is True
        assert result.evidence == "Source text"
        assert result.confidence == 0.9

    def test_parse_verification_unsupported(self, config):
        """Parse unsupported verification."""
        judge = LLMJudge(config=config)

        response = '{"supported": false, "confidence": 0.8}'
        result = judge._parse_verification("Invalid claim", response)

        assert result.claim == "Invalid claim"
        assert result.supported is False
        assert result.evidence is None

    def test_parse_verification_invalid(self, config):
        """Handle invalid verification response."""
        judge = LLMJudge(config=config)

        response = "Not valid JSON"
        result = judge._parse_verification("Some claim", response)

        assert result.claim == "Some claim"
        assert result.supported is False  # Default to unsupported
        assert result.confidence == 0.5


class TestRelevancePromptBuilding:
    """Tests for relevance prompt construction."""

    def test_build_context_relevance_prompt(self, config):
        """Build context relevance prompt."""
        judge = LLMJudge(config=config)

        prompt = judge._build_relevance_prompt(
            query="What is Python?",
            text="Python is a programming language.",
            context_type="context",
        )

        assert "What is Python?" in prompt
        assert "Python is a programming language." in prompt
        assert "context" in prompt.lower()
        assert "JSON" in prompt

    def test_build_answer_relevance_prompt(self, config):
        """Build answer relevance prompt."""
        judge = LLMJudge(config=config)

        prompt = judge._build_relevance_prompt(
            query="What is Python?",
            text="Python is a versatile language.",
            context_type="answer",
        )

        assert "answer" in prompt.lower()
        assert "addresses the question" in prompt.lower()


class TestFaithfulnessReasoning:
    """Tests for faithfulness reasoning generation."""

    def test_build_faithfulness_reasoning_all_supported(self, config):
        """Reasoning for all claims supported."""
        judge = LLMJudge(config=config)

        claims = ["Claim 1", "Claim 2"]
        results = [
            ClaimVerificationResult(claim="Claim 1", supported=True),
            ClaimVerificationResult(claim="Claim 2", supported=True),
        ]

        reasoning = judge._build_faithfulness_reasoning(claims, results)

        assert "Verified 2 claims" in reasoning
        assert "Supported" in reasoning

    def test_build_faithfulness_reasoning_mixed(self, config):
        """Reasoning for mixed support."""
        judge = LLMJudge(config=config)

        claims = ["True claim", "False claim"]
        results = [
            ClaimVerificationResult(claim="True claim", supported=True),
            ClaimVerificationResult(claim="False claim", supported=False),
        ]

        reasoning = judge._build_faithfulness_reasoning(claims, results)

        assert "Supported" in reasoning
        assert "Unsupported" in reasoning


class TestLLMJudgeWithMockedClient:
    """Tests with mocked LLM client."""

    @pytest.mark.asyncio
    async def test_evaluate_relevance(self, config, mock_anthropic_client):
        """Evaluate relevance with mocked client."""
        mock_anthropic_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"score": 0.8, "reasoning": "Good relevance"}')]
        )

        judge = LLMJudge(config=config, llm_client=mock_anthropic_client)

        result = await judge.evaluate_relevance(
            query="What is Python?",
            text="Python is a programming language.",
            context_type="context",
        )

        assert result.score == 0.8
        assert result.passed is True
        mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_claims(self, config, mock_anthropic_client):
        """Extract claims with mocked client."""
        mock_anthropic_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='["Python was created by Guido", "Python is open source"]')]
        )

        judge = LLMJudge(config=config, llm_client=mock_anthropic_client)

        claims = await judge.extract_claims("Python was created by Guido. Python is open source.")

        assert len(claims) == 2
        assert "Python was created by Guido" in claims

    @pytest.mark.asyncio
    async def test_extract_claims_empty_text(self, config, mock_anthropic_client):
        """Empty text returns empty claims list."""
        judge = LLMJudge(config=config, llm_client=mock_anthropic_client)

        claims = await judge.extract_claims("")

        assert claims == []
        mock_anthropic_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_claim(self, config, mock_anthropic_client):
        """Verify claim with mocked client."""
        mock_anthropic_client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text='{"supported": true, "evidence": "Found in source", "confidence": 0.95}'
                )
            ]
        )

        judge = LLMJudge(config=config, llm_client=mock_anthropic_client)

        result = await judge.verify_claim(
            claim="Python is open source",
            context="Python is an open source programming language.",
        )

        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness(self, config, mock_anthropic_client):
        """Evaluate faithfulness with mocked client."""
        # First call: extract claims
        # Second+ calls: verify each claim
        mock_anthropic_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(text='["Claim 1", "Claim 2"]')]),
            MagicMock(content=[MagicMock(text='{"supported": true, "confidence": 0.9}')]),
            MagicMock(content=[MagicMock(text='{"supported": true, "confidence": 0.8}')]),
        ]

        judge = LLMJudge(config=config, llm_client=mock_anthropic_client)

        result = await judge.evaluate_faithfulness(
            response="Claim 1. Claim 2.",
            context="Context supporting both claims.",
        )

        assert result.score == 1.0  # Both claims supported
        assert result.passed is True
        assert result.metrics["num_claims"] == 2
        assert result.metrics["supported_claims"] == 2

    @pytest.mark.asyncio
    async def test_llm_call_error_handling(self, config, mock_anthropic_client):
        """Handle LLM call errors gracefully."""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        judge = LLMJudge(config=config, llm_client=mock_anthropic_client)

        # Should return graceful degradation, not raise
        result = await judge.evaluate_relevance(
            query="test",
            text="test",
            context_type="context",
        )

        assert result.score == 0.5  # Default degraded score
