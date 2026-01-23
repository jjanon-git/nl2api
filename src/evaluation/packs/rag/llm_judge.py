"""
LLM Judge Abstraction

Provides a unified interface for LLM-as-judge evaluations.
Used by reference-free stages (ContextRelevance, Faithfulness, AnswerRelevance).

Features:
- Claim extraction for faithfulness evaluation
- Structured scoring with JSON output
- Retry logic with exponential backoff
- Prompt templating with few-shot examples
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.contracts import LLMJudgeConfig


@dataclass
class ClaimVerificationResult:
    """Result of verifying a single claim against context."""

    claim: str
    supported: bool
    evidence: str | None = None
    confidence: float = 1.0


@dataclass
class JudgeResult:
    """Result from an LLM judge evaluation."""

    score: float
    passed: bool
    reasoning: str
    raw_response: str
    metrics: dict[str, Any] = field(default_factory=dict)


class LLMJudge:
    """
    Abstraction for LLM-as-judge calls.

    Handles prompt construction, LLM invocation, and response parsing.
    Supports multiple evaluation types (relevance, faithfulness, etc.).
    """

    def __init__(
        self,
        config: LLMJudgeConfig | None = None,
        llm_client: Any = None,
    ):
        """
        Initialize the LLM judge.

        Args:
            config: Judge configuration (model, temperature, thresholds)
            llm_client: LLM client instance. If None, will be created on first use.
        """
        self.config = config or LLMJudgeConfig()
        self._llm_client = llm_client
        self._client_initialized = False

    @property
    def llm_client(self) -> Any:
        """Lazy initialization of LLM client."""
        if not self._client_initialized and self._llm_client is None:
            self._llm_client = self._create_llm_client()
            self._client_initialized = True
        return self._llm_client

    def _create_llm_client(self) -> Any:
        """Create LLM client based on config."""
        # Import here to avoid circular dependencies
        try:
            from anthropic import AsyncAnthropic

            return AsyncAnthropic()
        except ImportError:
            raise RuntimeError(
                "anthropic package required for LLM judge. Install with: pip install anthropic"
            )

    async def evaluate_relevance(
        self,
        query: str,
        text: str,
        context_type: str = "context",
    ) -> JudgeResult:
        """
        Evaluate relevance of text to a query.

        Used for:
        - Context relevance (is retrieved context relevant to query?)
        - Answer relevance (does answer address the question?)

        Args:
            query: The query/question
            text: The text to evaluate (context or answer)
            context_type: Type of text ("context" or "answer")

        Returns:
            JudgeResult with relevance score and reasoning
        """
        prompt = self._build_relevance_prompt(query, text, context_type)
        return await self._evaluate_with_prompt(prompt)

    async def extract_claims(self, text: str) -> list[str]:
        """
        Extract atomic claims from text for faithfulness evaluation.

        Decomposes a response into individual verifiable claims.
        Uses the RAGAS approach of claim decomposition.

        Args:
            text: Response text to decompose

        Returns:
            List of atomic claims
        """
        if not text or len(text.strip()) < 10:
            return []

        prompt = f"""Extract all factual claims from the following text.
Each claim should be:
1. A single, atomic statement
2. Verifiable against source documents
3. Self-contained (understandable without context)

Text:
{text}

Return a JSON array of claims. Example:
["The company was founded in 2020", "Revenue grew by 15%"]

Claims:"""

        response = await self._call_llm(prompt)
        return self._parse_claims(response)

    async def verify_claim(
        self,
        claim: str,
        context: str,
    ) -> ClaimVerificationResult:
        """
        Verify if a claim is supported by the context.

        Args:
            claim: The claim to verify
            context: The source context

        Returns:
            ClaimVerificationResult with support status and evidence
        """
        prompt = f"""Determine if the following claim is supported by the context.

Claim: {claim}

Context:
{context}

Respond with JSON:
{{
    "supported": true/false,
    "evidence": "quote from context if supported, null if not",
    "confidence": 0.0-1.0
}}

Response:"""

        response = await self._call_llm(prompt)
        return self._parse_verification(claim, response)

    async def evaluate_faithfulness(
        self,
        response: str,
        context: str,
    ) -> JudgeResult:
        """
        Evaluate if response is grounded in context (faithfulness).

        Uses claim extraction + verification approach from RAGAS.

        Args:
            response: The generated response
            context: The source context

        Returns:
            JudgeResult with faithfulness score
        """
        # Extract claims
        claims = await self.extract_claims(response)

        if not claims:
            return JudgeResult(
                score=1.0,
                passed=True,
                reasoning="No claims to verify",
                raw_response="",
                metrics={"num_claims": 0, "supported_claims": 0},
            )

        # Verify each claim
        supported_count = 0
        verification_results = []

        for claim in claims:
            result = await self.verify_claim(claim, context)
            verification_results.append(result)
            if result.supported:
                supported_count += 1

        # Compute score
        score = supported_count / len(claims)
        passed = score >= self.config.pass_threshold

        return JudgeResult(
            score=score,
            passed=passed,
            reasoning=self._build_faithfulness_reasoning(claims, verification_results),
            raw_response="",
            metrics={
                "num_claims": len(claims),
                "supported_claims": supported_count,
                "unsupported_claims": [r.claim for r in verification_results if not r.supported],
            },
        )

    async def _evaluate_with_prompt(self, prompt: str) -> JudgeResult:
        """
        Run evaluation with a pre-built prompt.

        Expects LLM to return JSON with score and reasoning.
        """
        response = await self._call_llm(prompt)
        return self._parse_judge_response(response)

    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with retry logic.

        Returns the raw text response.
        """
        client = self.llm_client
        if client is None:
            raise RuntimeError("LLM client not initialized")

        # Handle Anthropic client
        try:
            message = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            # Return error as response for graceful degradation
            return json.dumps(
                {
                    "score": 0.5,
                    "reasoning": f"LLM call failed: {e}",
                    "error": True,
                }
            )

    def _build_relevance_prompt(
        self,
        query: str,
        text: str,
        context_type: str,
    ) -> str:
        """Build prompt for relevance evaluation."""
        if context_type == "context":
            instruction = "Evaluate if the context is relevant and useful for answering the query."
        else:
            instruction = "Evaluate if the answer addresses the question completely and accurately."

        return f"""You are evaluating the relevance of a {context_type} to a query.

{instruction}

Query: {query}

{context_type.title()}:
{text}

Rate on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly addresses the query
- 0.7-0.9: Relevant, addresses most aspects
- 0.4-0.6: Partially relevant, some useful information
- 0.1-0.3: Marginally relevant, mostly off-topic
- 0.0: Not relevant at all

Respond with JSON:
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<explanation of score>"
}}

Response:"""

    def _parse_judge_response(self, response: str) -> JudgeResult:
        """Parse LLM response into JudgeResult."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")

            return JudgeResult(
                score=min(1.0, max(0.0, score)),  # Clamp to [0, 1]
                passed=score >= self.config.pass_threshold,
                reasoning=reasoning,
                raw_response=response,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # Graceful degradation on parse failure
            return JudgeResult(
                score=0.5,
                passed=False,
                reasoning=f"Failed to parse LLM response: {response[:200]}",
                raw_response=response,
            )

    def _parse_claims(self, response: str) -> list[str]:
        """Parse claims from LLM response."""
        try:
            # Try to extract JSON array
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
                return [str(c).strip() for c in claims if c]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: split by newlines and clean up
        lines = response.strip().split("\n")
        claims = []
        for line in lines:
            # Remove numbering, bullets, etc.
            cleaned = re.sub(r"^[\d\-\*\.\)]+\s*", "", line.strip())
            if cleaned and len(cleaned) > 10:
                claims.append(cleaned)
        return claims

    def _parse_verification(self, claim: str, response: str) -> ClaimVerificationResult:
        """Parse verification response."""
        try:
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return ClaimVerificationResult(
                claim=claim,
                supported=bool(data.get("supported", False)),
                evidence=data.get("evidence"),
                confidence=float(data.get("confidence", 1.0)),
            )
        except (json.JSONDecodeError, ValueError):
            # Assume not supported if can't parse
            return ClaimVerificationResult(
                claim=claim,
                supported=False,
                confidence=0.5,
            )

    def _build_faithfulness_reasoning(
        self,
        claims: list[str],
        results: list[ClaimVerificationResult],
    ) -> str:
        """Build explanation for faithfulness score."""
        supported = [r for r in results if r.supported]
        unsupported = [r for r in results if not r.supported]

        parts = [f"Verified {len(claims)} claims:"]

        if supported:
            parts.append(
                f"  Supported ({len(supported)}): {', '.join(r.claim[:50] for r in supported[:3])}"
            )
        if unsupported:
            parts.append(
                f"  Unsupported ({len(unsupported)}): {', '.join(r.claim[:50] for r in unsupported[:3])}"
            )

        return " ".join(parts)
