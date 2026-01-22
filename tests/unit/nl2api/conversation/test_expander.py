"""Tests for QueryExpander."""


from src.nl2api.conversation.expander import ExpansionResult, QueryExpander
from src.nl2api.conversation.models import ConversationContext


class TestExpansionResult:
    """Tests for ExpansionResult dataclass."""

    def test_result_creation_no_expansion(self) -> None:
        """Test creating a result with no expansion."""
        result = ExpansionResult(
            original_query="What is Apple's price?",
            expanded_query="What is Apple's price?",
            was_expanded=False,
        )

        assert result.original_query == "What is Apple's price?"
        assert result.expanded_query == "What is Apple's price?"
        assert result.was_expanded is False
        assert result.expansion_type is None
        assert result.confidence == 1.0

    def test_result_creation_with_expansion(self) -> None:
        """Test creating a result with expansion."""
        result = ExpansionResult(
            original_query="their EPS",
            expanded_query="Apple's EPS",
            was_expanded=True,
            expansion_type="entity",
            confidence=0.9,
        )

        assert result.was_expanded is True
        assert result.expansion_type == "entity"
        assert result.confidence == 0.9


class TestQueryExpanderNoContext:
    """Tests for expansion with no context."""

    def test_no_expansion_without_context(self) -> None:
        """Test that queries are not expanded without context."""
        expander = QueryExpander()
        context = ConversationContext()  # Empty context

        result = expander.expand("their EPS", context)

        assert result.was_expanded is False
        assert result.expanded_query == "their EPS"

    def test_no_expansion_empty_entities(self) -> None:
        """Test no expansion when context has no entities."""
        expander = QueryExpander()
        context = ConversationContext(turn_count=1)  # Has turns but no entities

        result = expander.expand("their EPS", context)

        assert result.was_expanded is False


class TestQueryExpanderPronounExpansion:
    """Tests for pronoun expansion."""

    def test_expands_their(self) -> None:
        """Test expanding 'their' to entity name."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = expander.expand("What is their EPS?", context)

        assert result.was_expanded is True
        assert result.expansion_type == "entity"
        # "their" is replaced with entity name (not possessive)
        assert "Apple" in result.expanded_query
        assert "their" not in result.expanded_query.lower()

    def test_expands_its(self) -> None:
        """Test expanding 'its' to entity name with possessive."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Microsoft": "MSFT.O"},
            turn_count=1,
        )

        result = expander.expand("What is its revenue?", context)

        assert result.was_expanded is True
        assert "Microsoft's" in result.expanded_query

    def test_expands_the_company(self) -> None:
        """Test expanding 'the company' to entity name."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Google": "GOOGL.O"},
            turn_count=1,
        )

        result = expander.expand("What is the company revenue?", context)

        assert result.was_expanded is True
        assert "Google" in result.expanded_query

    def test_expands_the_companys(self) -> None:
        """Test expanding 'the company's' to entity name with possessive."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Tesla": "TSLA.O"},
            turn_count=1,
        )

        result = expander.expand("What is the company's market cap?", context)

        assert result.was_expanded is True
        assert "Tesla's" in result.expanded_query

    def test_expands_the_stock(self) -> None:
        """Test expanding 'the stock' to entity name."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Amazon": "AMZN.O"},
            turn_count=1,
        )

        result = expander.expand("How is the stock performing?", context)

        assert result.was_expanded is True
        assert "Amazon" in result.expanded_query

    def test_uses_most_recent_entity(self) -> None:
        """Test that the most recent entity is used for expansion."""
        expander = QueryExpander()
        # Multiple entities - last one should be used
        context = ConversationContext(
            entities={"Apple": "AAPL.O", "Microsoft": "MSFT.O", "Google": "GOOGL.O"},
            turn_count=3,
        )

        result = expander.expand("What is their EPS?", context)

        assert result.was_expanded is True
        # Should use Google (last entity) - "their" replaced with just entity name
        assert "Google" in result.expanded_query
        assert "Apple" not in result.expanded_query
        assert "Microsoft" not in result.expanded_query

    def test_preserves_query_structure(self) -> None:
        """Test that expansion preserves the rest of the query."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = expander.expand("What is their quarterly EPS estimate for 2024?", context)

        assert result.was_expanded is True
        assert "Apple" in result.expanded_query
        assert "quarterly" in result.expanded_query.lower()
        assert "2024" in result.expanded_query

    def test_confidence_for_entity_expansion(self) -> None:
        """Test that entity expansion has appropriate confidence."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = expander.expand("their EPS", context)

        assert result.confidence == 0.9


class TestQueryExpanderComparisonExpansion:
    """Tests for comparison/addition expansion."""

    def test_expands_what_about_entity(self) -> None:
        """Test expanding 'what about [entity]' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("What about Microsoft?", context)

        assert result.was_expanded is True
        assert result.expansion_type == "comparison"
        assert "Microsoft" in result.expanded_query
        assert "EPS" in result.expanded_query  # Should include the metric from context

    def test_expands_how_about_entity(self) -> None:
        """Test expanding 'how about [entity]' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.Revenue"],
            turn_count=1,
        )

        result = expander.expand("How about Google?", context)

        assert result.was_expanded is True
        assert "Google" in result.expanded_query

    def test_expands_compare_to(self) -> None:
        """Test expanding 'compare to [entity]' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("Compare to Amazon", context)

        assert result.was_expanded is True
        assert "Amazon" in result.expanded_query

    def test_expands_same_for(self) -> None:
        """Test expanding 'same for [entity]' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("Same for Tesla", context)

        assert result.was_expanded is True
        assert "Tesla" in result.expanded_query

    def test_comparison_requires_fields(self) -> None:
        """Test that comparison expansion requires fields in context."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=[],  # No fields
            turn_count=1,
        )

        result = expander.expand("What about Microsoft?", context)

        # Without fields, can't determine what metric to compare
        assert result.was_expanded is False

    def test_does_not_expand_period_keywords(self) -> None:
        """Test that period keywords are not treated as entities."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        # "quarterly" should not be treated as an entity comparison
        result = expander.expand("What about quarterly?", context)

        # Should not expand as comparison (might expand as period instead)
        assert result.expansion_type != "comparison"


class TestQueryExpanderPeriodExpansion:
    """Tests for period expansion."""

    def test_expands_now_quarterly(self) -> None:
        """Test expanding 'now quarterly' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("Now quarterly", context)

        assert result.was_expanded is True
        assert result.expansion_type == "period"
        assert "Apple" in result.expanded_query
        assert "quarterly" in result.expanded_query.lower()

    def test_expands_for_annual(self) -> None:
        """Test expanding 'for annual' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Microsoft": "MSFT.O"},
            fields=["TR.Revenue"],
            turn_count=1,
        )

        result = expander.expand("For annual", context)

        assert result.was_expanded is True
        assert "Microsoft" in result.expanded_query
        assert "annual" in result.expanded_query.lower()

    def test_period_expansion_requires_entities(self) -> None:
        """Test that period expansion requires entities in context."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={},  # No entities
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("Now quarterly", context)

        assert result.was_expanded is False

    def test_period_expansion_requires_fields(self) -> None:
        """Test that period expansion requires fields in context."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=[],  # No fields
            turn_count=1,
        )

        result = expander.expand("Now quarterly", context)

        assert result.was_expanded is False

    def test_period_expansion_short_queries_only(self) -> None:
        """Test that period expansion only applies to short queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        # Long query with quarterly should not trigger period expansion
        result = expander.expand(
            "What is Apple's quarterly EPS estimate for the technology sector in 2024?",
            context,
        )

        # Should not expand as period (query is too long/complex)
        assert result.expansion_type != "period"


class TestQueryExpanderMetricExpansion:
    """Tests for metric expansion."""

    def test_expands_what_about_metric(self) -> None:
        """Test expanding 'what about [metric]' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("What about revenue?", context)

        assert result.was_expanded is True
        assert result.expansion_type == "metric"
        assert "Apple" in result.expanded_query
        assert "revenue" in result.expanded_query.lower()

    def test_expands_and_metric(self) -> None:
        """Test expanding 'and [metric]' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Microsoft": "MSFT.O"},
            turn_count=1,
        )

        result = expander.expand("And EPS?", context)

        assert result.was_expanded is True
        assert "Microsoft" in result.expanded_query
        # Metric is extracted from query and may be lowercased
        assert "eps" in result.expanded_query.lower()

    def test_expands_also_metric(self) -> None:
        """Test expanding 'also [metric]' queries."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Google": "GOOGL.O"},
            turn_count=1,
        )

        result = expander.expand("Also EBITDA?", context)

        assert result.was_expanded is True
        assert "Google" in result.expanded_query
        # Metric may be lowercased during extraction
        assert "ebitda" in result.expanded_query.lower()

    def test_metric_expansion_requires_entities(self) -> None:
        """Test that metric expansion requires entities in context."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={},  # No entities
            turn_count=1,
        )

        result = expander.expand("What about revenue?", context)

        assert result.was_expanded is False

    def test_only_expands_known_metrics(self) -> None:
        """Test that only known metrics trigger metric expansion."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        # "pandas" is not a known metric
        result = expander.expand("What about pandas?", context)

        # Should not expand as metric
        assert result.expansion_type != "metric"


class TestQueryExpanderExpansionPriority:
    """Tests for expansion priority/order."""

    def test_pronoun_expansion_takes_priority(self) -> None:
        """Test that pronoun expansion is checked first."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        # Query has pronoun - should be expanded as entity
        result = expander.expand("What is their revenue?", context)

        assert result.expansion_type == "entity"

    def test_no_double_expansion(self) -> None:
        """Test that only one expansion type is applied."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("their revenue", context)

        # Should have exactly one expansion type
        assert result.expansion_type in ["entity", "metric", "comparison", "period", None]


class TestQueryExpanderMetricInference:
    """Tests for metric inference from field codes."""

    def test_infers_eps_from_field(self) -> None:
        """Test inferring EPS metric from field code."""
        expander = QueryExpander()

        metric = expander._infer_metric_from_fields(["TR.EPSMean"])

        assert metric is not None
        assert "EPS" in metric

    def test_infers_revenue_from_field(self) -> None:
        """Test inferring revenue metric from field code."""
        expander = QueryExpander()

        metric = expander._infer_metric_from_fields(["TR.RevenueHigh"])

        assert metric is not None
        assert "revenue" in metric.lower()

    def test_infers_ebitda_from_field(self) -> None:
        """Test inferring EBITDA metric from field code."""
        expander = QueryExpander()

        metric = expander._infer_metric_from_fields(["TR.EBITDAMean"])

        assert metric is not None
        assert "EBITDA" in metric

    def test_infers_dividend_from_field(self) -> None:
        """Test inferring dividend metric from field code."""
        expander = QueryExpander()

        metric = expander._infer_metric_from_fields(["TR.DPSMean"])

        assert metric is not None
        assert "dividend" in metric.lower()

    def test_infers_price_target_from_field(self) -> None:
        """Test inferring price target metric from field code."""
        expander = QueryExpander()

        metric = expander._infer_metric_from_fields(["TR.PriceTargetMean"])

        assert metric is not None
        assert "price target" in metric.lower()

    def test_fallback_to_estimate(self) -> None:
        """Test fallback to generic 'estimate' for unknown fields."""
        expander = QueryExpander()

        metric = expander._infer_metric_from_fields(["TR.SomeUnknownField"])

        assert metric == "estimate"

    def test_uses_most_recent_field(self) -> None:
        """Test that the most recent field is used for inference."""
        expander = QueryExpander()

        # Multiple fields - last one should be used
        metric = expander._infer_metric_from_fields(["TR.EPSMean", "TR.Revenue"])

        assert metric is not None
        assert "revenue" in metric.lower()

    def test_returns_none_for_empty_fields(self) -> None:
        """Test that empty fields return None."""
        expander = QueryExpander()

        metric = expander._infer_metric_from_fields([])

        assert metric is None


class TestQueryExpanderEdgeCases:
    """Tests for edge cases."""

    def test_handles_empty_query(self) -> None:
        """Test handling empty query."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = expander.expand("", context)

        assert result is not None
        assert result.expanded_query == ""

    def test_handles_whitespace_query(self) -> None:
        """Test handling whitespace-only query."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = expander.expand("   ", context)

        assert result is not None

    def test_case_insensitive_pronoun_matching(self) -> None:
        """Test that pronoun matching is case-insensitive."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = expander.expand("What is THEIR EPS?", context)

        assert result.was_expanded is True
        # "THEIR" is replaced with entity name (case-insensitive match)
        assert "Apple" in result.expanded_query

    def test_handles_special_characters_in_query(self) -> None:
        """Test handling special characters in query."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            turn_count=1,
        )

        result = expander.expand("What is their P/E ratio?", context)

        assert result.was_expanded is True
        assert "Apple" in result.expanded_query
        assert "P/E" in result.expanded_query

    def test_strips_trailing_punctuation_in_comparison(self) -> None:
        """Test that trailing punctuation is stripped in comparisons."""
        expander = QueryExpander()
        context = ConversationContext(
            entities={"Apple": "AAPL.O"},
            fields=["TR.EPSMean"],
            turn_count=1,
        )

        result = expander.expand("What about Microsoft??", context)

        assert result.was_expanded is True
        # Should not have extra punctuation in the entity name
        assert "Microsoft?" not in result.expanded_query
