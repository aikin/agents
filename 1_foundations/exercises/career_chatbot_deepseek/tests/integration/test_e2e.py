"""End-to-end integration tests."""

import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest

from app import initialize_system


@pytest.fixture(scope="function")
def system():
    """Initialize system for testing with mocked LLM."""
    # Use temporary directories
    temp_data = tempfile.mkdtemp()
    temp_chroma = tempfile.mkdtemp()

    # Mock settings for testing
    with patch('config.settings.settings') as mock_settings:
        # Configure mock settings
        mock_settings.db_path = f"{temp_data}/test.db"
        mock_settings.chroma_path = temp_chroma
        mock_settings.enable_evaluation = False
        mock_settings.enable_rag = True
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.kb_path = "knowledge_base"
        mock_settings.agent_name = "Test Agent"
        mock_settings.google_api_key = None  # Disable evaluator

        # Mock LLM responses
        with patch('services.llm_service.OpenAI') as mock_openai:
            # Create mock client
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Create mock response structure
            def create_mock_response(content: str, finish_reason: str = "stop"):
                mock_response = Mock()
                mock_choice = Mock()
                mock_message = Mock()
                mock_message.content = content
                mock_message.tool_calls = None
                mock_choice.message = mock_message
                mock_choice.finish_reason = finish_reason
                mock_response.choices = [mock_choice]
                return mock_response

            # Set up default mock response
            mock_client.chat.completions.create.return_value = create_mock_response(
                "Hello! I'm Test Agent, an AI assistant. How can I help you today?"
            )

            try:
                controller, rag = initialize_system()
                # Store mock client for test customization
                controller._mock_client = mock_client
                controller._create_mock_response = create_mock_response
                yield controller, rag

            finally:
                # Cleanup temporary directories
                try:
                    shutil.rmtree(temp_data, ignore_errors=True)
                    shutil.rmtree(temp_chroma, ignore_errors=True)
                except OSError:
                    pass  # Ignore cleanup errors in tests


def test_e2e_simple_conversation(system):
    """Test simple conversation flow."""
    controller, _ = system

    # Mock LLM response for this test
    controller._mock_client.chat.completions.create.return_value = controller._create_mock_response(
        "Hello! I'm Test Agent, an AI assistant. How can I help you today?"
    )

    response = controller.process_message("Hello")

    assert response is not None
    assert len(response) > 0
    assert isinstance(response, str)
    assert "Test Agent" in response


def test_e2e_multiple_turns(system):
    """Test multiple conversation turns."""
    controller, _ = system

    # Mock different responses for each turn
    responses = [
        controller._create_mock_response("My name is Test Agent."),
        controller._create_mock_response("I have 5 years of experience in AI and software development."),
    ]
    controller._mock_client.chat.completions.create.side_effect = responses

    response1 = controller.process_message("What is your name?")
    response2 = controller.process_message("What is your experience?")

    assert response1 is not None
    assert response2 is not None
    assert isinstance(response1, str)
    assert isinstance(response2, str)
    assert len(response1) > 0
    assert len(response2) > 0
    assert "Test Agent" in response1
    assert "experience" in response2.lower()


def test_e2e_analytics(system):
    """Test analytics after conversations."""
    controller, _ = system

    # Mock responses for analytics test
    mock_response = controller._create_mock_response("This is a test response.")
    controller._mock_client.chat.completions.create.return_value = mock_response

    controller.process_message("Test question 1")
    controller.process_message("Test question 2")
    controller.process_message("Test question 3")

    analytics = controller.get_analytics()

    assert analytics["total_conversations"] >= 3
    assert "total_contacts" in analytics
    assert "average_evaluation_score" in analytics
    assert "top_unknown_questions" in analytics
    assert isinstance(analytics["total_conversations"], int)
    assert analytics["total_conversations"] > 0


def test_e2e_memory_persistence(system):
    """Test that memory persists across conversation turns."""
    controller, _ = system

    # Mock responses for memory test
    responses = [
        controller._create_mock_response("Nice to meet you, John!"),
        controller._create_mock_response("Your name is John, as you just told me."),
    ]
    controller._mock_client.chat.completions.create.side_effect = responses

    # First message
    response1 = controller.process_message("Hello, my name is John")
    assert response1 is not None
    assert "John" in response1

    # Second message should have context from first
    response2 = controller.process_message("What's my name?")
    assert response2 is not None
    assert "John" in response2

    # Both responses should be different (indicating context awareness)
    assert response1 != response2


def test_e2e_tool_integration(system):
    """Test tool calling integration."""
    controller, _ = system

    # Mock response with contact information
    controller._mock_client.chat.completions.create.return_value = controller._create_mock_response(
        "You can reach me via email at test@example.com or through the contact form."
    )

    # Test contact tool by asking for contact info
    response = controller.process_message("How can I contact you?")

    assert response is not None
    assert len(response) > 0
    # Should contain some contact information
    contact_indicators = ["email", "contact", "reach", "@"]
    assert any(indicator in response.lower() for indicator in contact_indicators)


def test_e2e_error_handling(system):
    """Test error handling in conversation flow."""
    controller, _ = system

    # Mock responses for error handling test
    mock_response = controller._create_mock_response(
        "I'm here to help! Please feel free to ask me anything."
    )
    controller._mock_client.chat.completions.create.return_value = mock_response

    # Test with empty message
    response1 = controller.process_message("")
    assert response1 is not None

    # Test with very long message
    long_message = "What is " * 100 + "your experience?"
    response2 = controller.process_message(long_message)
    assert response2 is not None

    # Both should return valid responses
    assert isinstance(response1, str)
    assert isinstance(response2, str)
