from unittest.mock import Mock

import pytest

from core.career_agent import CareerAgent


@pytest.fixture
def mock_services():
    """Create mock services."""
    llm_service = Mock()
    rag_service = Mock()
    tool_registry = Mock()
    tool_registry.to_openai_format.return_value = []

    return llm_service, rag_service, tool_registry


def test_career_agent_init(mock_services):
    """Test career agent initialization."""
    llm, rag, registry = mock_services

    agent = CareerAgent(llm, rag, registry, "LinkedIn", "Summary")

    assert agent.llm_service is llm
    assert agent.rag_service is rag
    assert agent.tool_registry is registry


def test_career_agent_chat_simple(mock_services):
    """Test simple chat without tools."""
    llm, rag, registry = mock_services

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message.content = "Test response"
    llm.chat_completion.return_value = mock_response

    agent = CareerAgent(llm, rag, registry, "LinkedIn", "Summary")
    response = agent.chat("Hello", [])

    assert response == "Test response"
