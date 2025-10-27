from unittest.mock import Mock, patch

import pytest

from services.llm_service import LLMService


@pytest.fixture
def llm_service():
    """Create LLM service with mocked client."""
    with patch("services.llm_service.OpenAI"):
        return LLMService()


def test_chat_completion_basic(llm_service):
    """Test basic chat completion."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"

    llm_service.client.chat.completions.create = Mock(return_value=mock_response)

    messages = [{"role": "user", "content": "Hello"}]
    response = llm_service.chat_completion(messages)

    assert response is not None
    llm_service.client.chat.completions.create.assert_called_once()


def test_chat_completion_with_tools(llm_service):
    """Test chat completion with tools."""
    mock_response = Mock()
    llm_service.client.chat.completions.create = Mock(return_value=mock_response)

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test"}}]

    llm_service.chat_completion(messages, tools=tools)

    call_args = llm_service.client.chat.completions.create.call_args
    assert "tools" in call_args.kwargs
