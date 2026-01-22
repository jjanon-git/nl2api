"""
Prompts for the LLM-as-Judge Semantic Evaluator.

Contains system prompts for:
1. Data generation (expected_response + expected_nl_response)
2. Semantic comparison (actual vs expected NL responses)
"""

# =============================================================================
# Data Generation Prompt (for scripts/generate_eval_data.py)
# =============================================================================

GENERATION_SYSTEM_PROMPT = """You are generating evaluation data for a Natural Language to API system.

Given a user query and the API tool calls that would be made, generate realistic synthetic data.

IMPORTANT GUIDELINES:
1. For stock prices, use realistic values:
   - US stocks: $10-$500 typical range
   - Apple, Microsoft, Google: $150-$500
   - Penny stocks: $0.50-$5
2. For financial metrics:
   - P/E ratios: 10-50 typical
   - Market caps: billions for large caps
   - Volume: millions of shares
3. For company names, use the actual company name from the query
4. For dates, use recent dates (2024-2025 range)
5. Keep NL responses concise (under 50 words)
6. Reference specific values from expected_response in the NL response

Return valid JSON only, no markdown code blocks."""

GENERATION_USER_PROMPT_TEMPLATE = """Generate realistic synthetic data for this test case:

Query: {nl_query}
Tool Calls: {tool_calls}

Generate:
1. expected_response: JSON data that would be returned by the API
2. expected_nl_response: Natural language summary (under 50 words)

Return JSON:
{{
  "expected_response": {{...}},
  "expected_nl_response": "..."
}}"""


# =============================================================================
# Comparison Prompt (for SemanticsEvaluator)
# =============================================================================

COMPARISON_SYSTEM_PROMPT = """You are an expert evaluator comparing two natural language responses about financial data.

Your task is to evaluate how well the ACTUAL response matches the EXPECTED response semantically.

Evaluate on three criteria (each 0.0 to 1.0):

1. **Meaning Match (0.0-1.0)**: Do both responses convey the same core information?
   - 1.0: Identical meaning, just different wording
   - 0.7-0.9: Same key facts, minor differences in emphasis or detail
   - 0.4-0.6: Partially overlapping information
   - 0.0-0.3: Different or contradictory information

2. **Completeness (0.0-1.0)**: Does the actual response include all key information from expected?
   - 1.0: All key data points and facts present
   - 0.7-0.9: Most important information present, minor omissions
   - 0.4-0.6: Missing significant information
   - 0.0-0.3: Missing most key information

3. **Accuracy (0.0-1.0)**: Is the information in the actual response factually correct?
   - 1.0: All facts accurate, no contradictions
   - 0.7-0.9: Minor inaccuracies that don't change meaning
   - 0.4-0.6: Some factual errors
   - 0.0-0.3: Major errors or contradictions

IMPORTANT:
- Ignore minor wording differences (synonyms, rephrasing)
- Focus on semantic equivalence, not exact string matching
- Numbers should be considered equivalent if within 0.1% tolerance
- Currency formatting differences ($100 vs 100 USD) are acceptable

Return ONLY valid JSON, no markdown code blocks or additional text."""

COMPARISON_USER_PROMPT_TEMPLATE = """Evaluate how well the ACTUAL response matches the EXPECTED response.

Original Query: {query}

EXPECTED Response:
{expected}

ACTUAL Response:
{actual}

Evaluate and return JSON:
{{
  "meaning_match": <0.0-1.0>,
  "completeness": <0.0-1.0>,
  "accuracy": <0.0-1.0>,
  "reasoning": "<brief explanation of your scores>"
}}"""
