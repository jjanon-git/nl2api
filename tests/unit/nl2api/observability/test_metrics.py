"""Tests for RequestMetrics dataclass."""

from __future__ import annotations

import json
import pytest
from datetime import datetime

from src.nl2api.observability.metrics import RequestMetrics


class TestRequestMetricsInit:
    """Test RequestMetrics initialization."""

    def test_default_initialization(self) -> None:
        """Test metrics with default values."""
        metrics = RequestMetrics()

        assert metrics.request_id  # Auto-generated UUID
        assert metrics.timestamp  # Auto-generated timestamp
        assert metrics.query == ""
        assert metrics.query_length == 0
        assert metrics.success is True
        assert metrics.total_tokens == 0

    def test_query_length_computed(self) -> None:
        """Test query_length is computed from query."""
        metrics = RequestMetrics(query="What is Apple's EPS?")

        assert metrics.query_length == 20

    def test_custom_request_id(self) -> None:
        """Test custom request_id is preserved."""
        metrics = RequestMetrics(request_id="custom-123")

        assert metrics.request_id == "custom-123"

    def test_session_id_set(self) -> None:
        """Test session_id is set correctly."""
        metrics = RequestMetrics(session_id="session-456")

        assert metrics.session_id == "session-456"


class TestRequestMetricsSetters:
    """Test RequestMetrics setter methods."""

    def test_set_routing_result(self) -> None:
        """Test routing result setter."""
        metrics = RequestMetrics()
        metrics.set_routing_result(
            domain="estimates",
            confidence=0.95,
            cached=True,
            cache_type="exact",
            model="claude-haiku",
            latency_ms=50,
        )

        assert metrics.routing_domain == "estimates"
        assert metrics.routing_confidence == 0.95
        assert metrics.routing_cached is True
        assert metrics.routing_cache_type == "exact"
        assert metrics.routing_model == "claude-haiku"
        assert metrics.routing_latency_ms == 50

    def test_set_entity_resolution(self) -> None:
        """Test entity resolution setter."""
        metrics = RequestMetrics()
        metrics.set_entity_resolution(
            extracted=["Apple", "Microsoft", "Unknown Corp"],
            resolved={"Apple": "AAPL.O", "Microsoft": "MSFT.O"},
            method="fuzzy",
            latency_ms=100,
        )

        assert metrics.entities_extracted == ["Apple", "Microsoft", "Unknown Corp"]
        assert metrics.entities_resolved == {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}
        assert metrics.entities_unresolved == ["Unknown Corp"]
        assert metrics.entity_resolution_method == "fuzzy"
        assert metrics.entity_resolution_latency_ms == 100

    def test_set_context_retrieval(self) -> None:
        """Test context retrieval setter."""
        metrics = RequestMetrics()
        metrics.set_context_retrieval(
            mode="hybrid",
            field_codes_count=15,
            examples_count=5,
            latency_ms=200,
        )

        assert metrics.context_mode == "hybrid"
        assert metrics.context_field_codes_count == 15
        assert metrics.context_examples_count == 5
        assert metrics.context_latency_ms == 200

    def test_set_agent_result(self) -> None:
        """Test agent result setter."""
        metrics = RequestMetrics()
        metrics.set_agent_result(
            domain="estimates",
            used_llm=True,
            rule_matched=None,
            llm_model="claude-sonnet",
            tokens_prompt=500,
            tokens_completion=100,
            latency_ms=300,
        )

        assert metrics.agent_domain == "estimates"
        assert metrics.agent_used_llm is True
        assert metrics.agent_rule_matched is None
        assert metrics.agent_llm_model == "claude-sonnet"
        assert metrics.agent_tokens_prompt == 500
        assert metrics.agent_tokens_completion == 100
        assert metrics.agent_latency_ms == 300
        assert metrics.total_tokens == 600

    def test_set_agent_result_rule_based(self) -> None:
        """Test agent result for rule-based processing."""
        metrics = RequestMetrics()
        metrics.set_agent_result(
            domain="datastream",
            used_llm=False,
            rule_matched="price_pattern",
            llm_model=None,
            tokens_prompt=0,
            tokens_completion=0,
            latency_ms=5,
        )

        assert metrics.agent_domain == "datastream"
        assert metrics.agent_used_llm is False
        assert metrics.agent_rule_matched == "price_pattern"
        assert metrics.agent_llm_model is None
        assert metrics.total_tokens == 0

    def test_set_tool_calls_with_objects(self) -> None:
        """Test tool calls setter with ToolCall-like objects."""

        class MockToolCall:
            tool_name = "get_estimates"
            arguments = {"ric": "AAPL.O", "fields": ["TR.EPSMean"]}

        metrics = RequestMetrics()
        metrics.set_tool_calls([MockToolCall(), MockToolCall()])

        assert metrics.tool_calls_count == 2
        assert metrics.tool_names == ["get_estimates", "get_estimates"]
        assert len(metrics.tool_calls) == 2
        assert metrics.tool_calls[0]["tool_name"] == "get_estimates"

    def test_set_tool_calls_with_dicts(self) -> None:
        """Test tool calls setter with dictionaries."""
        metrics = RequestMetrics()
        metrics.set_tool_calls([
            {"tool_name": "get_prices", "arguments": {"ric": "AAPL.O"}},
            {"tool_name": "get_fundamentals", "arguments": {"ric": "MSFT.O"}},
        ])

        assert metrics.tool_calls_count == 2
        assert metrics.tool_names == ["get_prices", "get_fundamentals"]

    def test_set_error(self) -> None:
        """Test error setter."""
        metrics = RequestMetrics()
        metrics.set_error(ValueError("Invalid query"))

        assert metrics.success is False
        assert metrics.error_type == "ValueError"
        assert metrics.error_message == "Invalid query"

    def test_set_clarification(self) -> None:
        """Test clarification setter."""
        metrics = RequestMetrics()
        metrics.set_clarification("domain")

        assert metrics.needs_clarification is True
        assert metrics.clarification_type == "domain"


class TestRequestMetricsFinalize:
    """Test RequestMetrics finalization."""

    def test_finalize_sets_total_latency(self) -> None:
        """Test finalize sets total latency."""
        metrics = RequestMetrics()
        metrics.finalize(total_latency_ms=500)

        assert metrics.total_latency_ms == 500

    def test_finalize_computes_total_tokens(self) -> None:
        """Test finalize computes total tokens."""
        metrics = RequestMetrics()
        metrics.agent_tokens_prompt = 200
        metrics.agent_tokens_completion = 50
        metrics.finalize(total_latency_ms=500)

        assert metrics.total_tokens == 250


class TestRequestMetricsSerialization:
    """Test RequestMetrics serialization methods."""

    def test_to_dict(self) -> None:
        """Test to_dict produces valid dictionary."""
        metrics = RequestMetrics(
            query="Test query",
            routing_domain="estimates",
            routing_confidence=0.9,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["query"] == "Test query"
        assert result["routing_domain"] == "estimates"
        assert result["routing_confidence"] == 0.9

    def test_to_json(self) -> None:
        """Test to_json produces valid JSON string."""
        metrics = RequestMetrics(
            query="Test query",
            routing_domain="estimates",
        )

        json_str = metrics.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["query"] == "Test query"
        assert parsed["routing_domain"] == "estimates"

    def test_to_log_summary(self) -> None:
        """Test to_log_summary produces concise summary."""
        metrics = RequestMetrics()
        metrics.set_routing_result(
            domain="estimates",
            confidence=0.95,
            cached=True,
        )
        metrics.set_agent_result(
            domain="estimates",
            used_llm=True,
            tokens_prompt=100,
            tokens_completion=50,
        )
        metrics.set_tool_calls([{"tool_name": "test"}])
        metrics.finalize(total_latency_ms=250)

        summary = metrics.to_log_summary()

        assert "domain=estimates" in summary
        assert "conf=0.95" in summary
        assert "cached=True" in summary
        assert "llm=True" in summary
        assert "tools=1" in summary
        assert "latency=250ms" in summary
        assert "tokens=150" in summary

    def test_to_log_summary_with_error(self) -> None:
        """Test log summary includes error information."""
        metrics = RequestMetrics()
        metrics.set_error(ValueError("Test error"))
        metrics.finalize(total_latency_ms=100)

        summary = metrics.to_log_summary()

        assert "error=ValueError" in summary


class TestRequestMetricsPostInit:
    """Test RequestMetrics __post_init__ behavior."""

    def test_unresolved_entities_computed(self) -> None:
        """Test unresolved entities are computed in __post_init__."""
        metrics = RequestMetrics(
            entities_extracted=["Apple", "Unknown"],
            entities_resolved={"Apple": "AAPL.O"},
        )

        assert metrics.entities_unresolved == ["Unknown"]

    def test_total_tokens_computed(self) -> None:
        """Test total_tokens is computed in __post_init__."""
        metrics = RequestMetrics(
            agent_tokens_prompt=100,
            agent_tokens_completion=50,
        )

        assert metrics.total_tokens == 150
