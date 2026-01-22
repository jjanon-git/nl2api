"""
Unit tests for the agent factory.
"""

from unittest.mock import MagicMock

import pytest

from src.nl2api.agents import (
    AGENT_REGISTRY,
    DatastreamAgent,
    EstimatesAgent,
    FundamentalsAgent,
    OfficersAgent,
    ScreeningAgent,
    get_agent_by_name,
    list_available_agents,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return MagicMock()


class TestAgentRegistry:
    """Tests for the AGENT_REGISTRY constant."""

    def test_registry_has_five_agents(self):
        """Test that registry contains exactly 5 agents."""
        assert len(AGENT_REGISTRY) == 5

    def test_registry_contains_datastream(self):
        """Test that registry contains datastream agent."""
        assert "datastream" in AGENT_REGISTRY
        assert AGENT_REGISTRY["datastream"] == DatastreamAgent

    def test_registry_contains_estimates(self):
        """Test that registry contains estimates agent."""
        assert "estimates" in AGENT_REGISTRY
        assert AGENT_REGISTRY["estimates"] == EstimatesAgent

    def test_registry_contains_fundamentals(self):
        """Test that registry contains fundamentals agent."""
        assert "fundamentals" in AGENT_REGISTRY
        assert AGENT_REGISTRY["fundamentals"] == FundamentalsAgent

    def test_registry_contains_officers(self):
        """Test that registry contains officers agent."""
        assert "officers" in AGENT_REGISTRY
        assert AGENT_REGISTRY["officers"] == OfficersAgent

    def test_registry_contains_screening(self):
        """Test that registry contains screening agent."""
        assert "screening" in AGENT_REGISTRY
        assert AGENT_REGISTRY["screening"] == ScreeningAgent


class TestGetAgentByName:
    """Tests for get_agent_by_name function."""

    def test_get_datastream_agent(self, mock_llm):
        """Test creating datastream agent by name."""
        agent = get_agent_by_name("datastream", llm=mock_llm)
        assert isinstance(agent, DatastreamAgent)
        assert agent.domain_name == "datastream"

    def test_get_estimates_agent(self, mock_llm):
        """Test creating estimates agent by name."""
        agent = get_agent_by_name("estimates", llm=mock_llm)
        assert isinstance(agent, EstimatesAgent)
        assert agent.domain_name == "estimates"

    def test_get_fundamentals_agent(self, mock_llm):
        """Test creating fundamentals agent by name."""
        agent = get_agent_by_name("fundamentals", llm=mock_llm)
        assert isinstance(agent, FundamentalsAgent)
        assert agent.domain_name == "fundamentals"

    def test_get_officers_agent(self, mock_llm):
        """Test creating officers agent by name."""
        agent = get_agent_by_name("officers", llm=mock_llm)
        assert isinstance(agent, OfficersAgent)
        assert agent.domain_name == "officers"

    def test_get_screening_agent(self, mock_llm):
        """Test creating screening agent by name."""
        agent = get_agent_by_name("screening", llm=mock_llm)
        assert isinstance(agent, ScreeningAgent)
        assert agent.domain_name == "screening"

    def test_case_insensitive_lookup(self, mock_llm):
        """Test that agent lookup is case insensitive."""
        agent1 = get_agent_by_name("DATASTREAM", llm=mock_llm)
        agent2 = get_agent_by_name("Datastream", llm=mock_llm)
        agent3 = get_agent_by_name("datastream", llm=mock_llm)

        assert isinstance(agent1, DatastreamAgent)
        assert isinstance(agent2, DatastreamAgent)
        assert isinstance(agent3, DatastreamAgent)

    def test_unknown_agent_raises_value_error(self, mock_llm):
        """Test that unknown agent name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_by_name("unknown_agent", llm=mock_llm)

        assert "Unknown agent" in str(exc_info.value)
        assert "unknown_agent" in str(exc_info.value)
        assert "datastream" in str(exc_info.value)  # Lists available agents

    def test_with_rag_retriever(self, mock_llm):
        """Test creating agent with RAG retriever."""
        mock_rag = MagicMock()
        agent = get_agent_by_name("datastream", llm=mock_llm, rag=mock_rag)

        assert isinstance(agent, DatastreamAgent)
        assert agent._rag == mock_rag


class TestListAvailableAgents:
    """Tests for list_available_agents function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = list_available_agents()
        assert isinstance(result, list)

    def test_returns_five_agents(self):
        """Test that function returns exactly 5 agent names."""
        result = list_available_agents()
        assert len(result) == 5

    def test_contains_all_agent_names(self):
        """Test that result contains all expected agent names."""
        result = list_available_agents()
        expected = {"datastream", "estimates", "fundamentals", "officers", "screening"}
        assert set(result) == expected

    def test_matches_registry_keys(self):
        """Test that result matches AGENT_REGISTRY keys."""
        result = list_available_agents()
        assert set(result) == set(AGENT_REGISTRY.keys())
