"""
SEC RAG Question Generator.

Generates evaluation questions for SEC 10-K/10-Q filings RAG system.
Uses Claude 3.5 Haiku for question variations (Sonnet for complex queries).

Usage:
    # Test run (10 questions)
    python -m scripts.generators.sec_rag_question_generator --limit 10

    # Full generation
    python -m scripts.generators.sec_rag_question_generator
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class SECTestCase:
    """Represents a SEC RAG test case (Phase 1: questions only)."""

    id: str
    input: dict[str, str]  # {"query": "..."}
    expected: dict[str, Any]  # {"behavior": "answer"|"reject", ...}
    tags: list[str]
    category: str
    subcategory: str
    complexity: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# LLM Prompts (approved by user)
VARIATION_SYSTEM_PROMPT = """You are a financial analyst generating questions about SEC filings (10-K and 10-Q reports).
Given a template question, generate 3 natural variations that ask the same thing but with different phrasing.

Rules:
- Keep the same semantic meaning - the question should retrieve the same information
- Vary the formality (formal analyst vs. casual inquiry)
- Vary the structure (direct question, "Can you tell me...", "I need to know...")
- Use natural financial terminology
- Do NOT change the company name, ticker, or fiscal year - use them exactly as provided
- Do NOT add information not in the original template

Output format: JSON array of 3 strings, one per line:
["variation 1", "variation 2", "variation 3"]"""

REJECTION_SYSTEM_PROMPT = """You are generating test questions that a SEC filing Q&A system should REJECT.

Given a rejection category and company context, generate a natural-sounding question that falls into that category.

Rejection categories:
- investment_advice: Questions asking whether to buy/sell/hold, price predictions, portfolio recommendations
- out_of_scope: Questions about topics not in SEC filings (personal info, non-public data, unrelated topics)
- speculation: Questions asking for predictions or opinions beyond filed facts
- confidential: Questions seeking non-public or insider information

Rules:
- Make questions sound natural, like a real user might ask
- Include company names/tickers when relevant to make them realistic
- Vary the phrasing and directness

Output format: JSON with "question" and "rejection_reason":
{
  "question": "The question text",
  "rejection_reason": "Brief explanation of why this should be rejected"
}"""


class SECRAGQuestionGenerator:
    """Generates SEC filing questions for RAG evaluation."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        limit: int | None = None,
        use_sonnet_for_complex: bool = True,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.limit = limit
        self.use_sonnet_for_complex = use_sonnet_for_complex

        # Load data files
        self.companies = self._load_json("sec/sp500_companies.json")
        self.templates = self._load_json("sec/question_templates.json")

        # Anthropic client (use NL2API_ANTHROPIC_API_KEY from .env)
        api_key = os.getenv("NL2API_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

        # Track generated IDs for deduplication
        self.generated_ids: set[str] = set()

        # Stats
        self.stats = {
            "templates_processed": 0,
            "llm_calls": 0,
            "questions_generated": 0,
            "duplicates_skipped": 0,
        }

    def _load_json(self, relative_path: str) -> dict:
        """Load JSON file from data directory."""
        file_path = self.data_dir / relative_path
        if file_path.exists():
            with open(file_path) as f:
                return json.load(f)
        raise FileNotFoundError(f"Required file not found: {file_path}")

    def _generate_id(self, query: str, category: str) -> str:
        """Generate unique test case ID."""
        content = f"sec-rag:{category}:{query}"
        hash_str = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"sec-rag-{category}-{hash_str}"

    def _get_random_company(self) -> dict:
        """Get a random company from the S&P 500 list."""
        return random.choice(self.companies["companies"])

    def _get_random_year(self) -> str:
        """Get a random fiscal year."""
        return random.choice(self.companies["fiscal_years"])

    def _get_random_quarter(self) -> str:
        """Get a random quarter."""
        return random.choice(self.companies["quarters"])

    def _fill_template(self, template: str, company: dict, year: str) -> str:
        """Fill template placeholders with actual values."""
        result = template
        result = result.replace("{company}", company["name"])
        result = result.replace("{ticker}", company["ticker"])
        result = result.replace("{year}", year)
        result = result.replace("{year1}", str(int(year) - 1))
        result = result.replace("{year2}", year)
        result = result.replace("{quarter}", self._get_random_quarter())
        result = result.replace("{segment}", "Services")  # Default segment
        return result

    async def _generate_variations(
        self, filled_template: str, use_sonnet: bool = False
    ) -> list[str]:
        """Generate 3 natural variations of a question using LLM."""
        model = "claude-sonnet-4-20250514" if use_sonnet else "claude-3-5-haiku-latest"

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=500,
                system=VARIATION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f'Template question: "{filled_template}"\n\nGenerate 3 natural variations:',
                    }
                ],
            )
            self.stats["llm_calls"] += 1

            # Parse JSON response
            content = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            variations = json.loads(content)

            if isinstance(variations, list) and len(variations) >= 3:
                return variations[:3]
            return [filled_template]  # Fallback to original

        except Exception as e:
            print(f"Warning: LLM variation failed: {e}")
            return [filled_template]  # Fallback to original

    async def _generate_rejection_question(self, category: str, company: dict) -> dict | None:
        """Generate a rejection case question using LLM."""
        try:
            response = await self.client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=300,
                system=REJECTION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Category: {category}\nCompany: {company['name']} ({company['ticker']})\n\nGenerate a question that should be rejected:",
                    }
                ],
            )
            self.stats["llm_calls"] += 1

            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)

        except Exception as e:
            print(f"Warning: Rejection generation failed: {e}")
            return None

    def _create_test_case(
        self,
        query: str,
        subcategory: str,
        complexity: int,
        tags: list[str],
        metadata: dict,
        is_rejection: bool = False,
        rejection_reason: str | None = None,
    ) -> SECTestCase | None:
        """Create a test case, checking for duplicates."""
        test_id = self._generate_id(query, subcategory)

        if test_id in self.generated_ids:
            self.stats["duplicates_skipped"] += 1
            return None

        self.generated_ids.add(test_id)

        expected: dict[str, Any] = {
            "behavior": "reject" if is_rejection else "answer",
            "requires_citations": not is_rejection,
        }
        if is_rejection and rejection_reason:
            expected["rejection_reason"] = rejection_reason

        return SECTestCase(
            id=test_id,
            input={"query": query},
            expected=expected,
            tags=tags,
            category="rag",
            subcategory=subcategory,
            complexity=complexity,
            metadata=metadata,
        )

    async def _process_category(self, category_name: str, category_data: dict) -> list[SECTestCase]:
        """Process a single category of templates."""
        test_cases: list[SECTestCase] = []
        templates = category_data.get("templates", [])
        use_sonnet = category_data.get("use_sonnet", False) and self.use_sonnet_for_complex

        for template_info in templates:
            if self.limit and len(test_cases) >= self.limit // 7:  # ~equal per category
                break

            template = template_info["template"]
            company = self._get_random_company()
            year = self._get_random_year()
            filled = self._fill_template(template, company, year)

            # Generate variations
            variations = await self._generate_variations(filled, use_sonnet=use_sonnet)
            self.stats["templates_processed"] += 1

            for variation in variations:
                tc = self._create_test_case(
                    query=variation,
                    subcategory=category_name,
                    complexity=template_info.get("complexity", 1),
                    tags=template_info.get("tags", []) + [category_name],
                    metadata={
                        "template_id": template_info["id"],
                        "company": company["name"],
                        "ticker": company["ticker"],
                        "year": year,
                    },
                )
                if tc:
                    test_cases.append(tc)
                    self.stats["questions_generated"] += 1

        return test_cases

    async def _process_rejection_category(
        self, rejection_cat: str, templates: list[dict]
    ) -> list[SECTestCase]:
        """Process rejection case templates."""
        test_cases: list[SECTestCase] = []

        for template_info in templates:
            if self.limit and len(test_cases) >= 10:  # Cap rejections in test mode
                break

            company = self._get_random_company()

            # Use template directly or generate via LLM
            if "{company}" in template_info["template"]:
                query = template_info["template"].replace("{company}", company["name"])
                rejection_reason = template_info["rejection_reason"]
            else:
                result = await self._generate_rejection_question(rejection_cat, company)
                if not result:
                    continue
                query = result["question"]
                rejection_reason = result["rejection_reason"]

            tc = self._create_test_case(
                query=query,
                subcategory=f"rejection_{rejection_cat}",
                complexity=1,
                tags=["rejection", rejection_cat],
                metadata={
                    "template_id": template_info.get("id", "generated"),
                    "company": company["name"],
                    "ticker": company["ticker"],
                    "rejection_category": rejection_cat,
                },
                is_rejection=True,
                rejection_reason=rejection_reason,
            )
            if tc:
                test_cases.append(tc)
                self.stats["questions_generated"] += 1

        return test_cases

    async def generate(self) -> list[SECTestCase]:
        """Generate all SEC RAG questions."""
        all_test_cases: list[SECTestCase] = []
        categories = self.templates.get("categories", {})

        # Process regular categories
        for cat_name, cat_data in categories.items():
            if cat_name == "rejection_cases":
                continue  # Handle separately

            print(f"Processing category: {cat_name}")
            cases = await self._process_category(cat_name, cat_data)
            all_test_cases.extend(cases)
            print(f"  Generated {len(cases)} questions")

        # Process rejection cases
        rejection_data = categories.get("rejection_cases", {})
        rejection_categories = rejection_data.get("categories", {})

        for rej_cat, rej_data in rejection_categories.items():
            if rej_cat.startswith("_"):  # Skip placeholders
                continue

            print(f"Processing rejection category: {rej_cat}")
            templates = rej_data.get("templates", [])
            cases = await self._process_rejection_category(rej_cat, templates)
            all_test_cases.extend(cases)
            print(f"  Generated {len(cases)} rejection questions")

        return all_test_cases

    def save(self, test_cases: list[SECTestCase]):
        """Save test cases to JSON file."""
        output_path = self.output_dir / "questions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "_meta": {
                "name": "sec_filings_rag_questions",
                "capability": "rag_evaluation",
                "schema_version": "1.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "phase": "questions_only",
                "awaiting_answers": True,
                "generator": "scripts/generators/sec_rag_question_generator.py",
            },
            "metadata": {
                "total_questions": len(test_cases),
                "stats": self.stats,
            },
            "test_cases": [tc.to_dict() for tc in test_cases],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(test_cases)} questions to {output_path}")
        print(f"Stats: {json.dumps(self.stats, indent=2)}")


async def main():
    parser = argparse.ArgumentParser(description="Generate SEC RAG evaluation questions")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for test runs)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("scripts/data"),
        help="Data directory path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures/rag/sec_filings"),
        help="Output directory path",
    )
    parser.add_argument(
        "--no-sonnet",
        action="store_true",
        help="Use Haiku for all categories (skip Sonnet for complex)",
    )
    args = parser.parse_args()

    generator = SECRAGQuestionGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        use_sonnet_for_complex=not args.no_sonnet,
    )

    print(f"Starting SEC RAG question generation (limit={args.limit})")
    print("=" * 60)

    test_cases = await generator.generate()
    generator.save(test_cases)


if __name__ == "__main__":
    asyncio.run(main())
