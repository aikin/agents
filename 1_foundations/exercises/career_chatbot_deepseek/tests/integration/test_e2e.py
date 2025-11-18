"""End-to-end integration tests."""

import os
import tempfile
import shutil

import pytest

from app import initialize_system


@pytest.fixture
def system():
    """Initialize system for testing."""
    # Use temporary directories
    temp_data = tempfile.mkdtemp()
    temp_chroma = tempfile.mkdtemp()

    # Set environment variables for testing
    original_env = {}
    test_env = {
        "DB_PATH": f"{temp_data}/test.db",
        "CHROMA_PATH": temp_chroma,
        "ENABLE_EVALUATION": "false",  # Disable for faster tests
        "ENABLE_RAG": "true",
        "DEEPSEEK_API_KEY": "test-key",  # Mock key for testing
        "KB_PATH": "knowledge_base",
    }

    # Save original environment
    for key in test_env:
        original_env[key] = os.environ.get(key)

    # Set test environment
    os.environ.update(test_env)

    try:
        controller, rag = initialize_system()

        yield controller, rag

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

        # Cleanup temporary directories
        try:
            shutil.rmtree(temp_data)
            shutil.rmtree(temp_chroma)
        except OSError:
            pass  # Ignore cleanup errors in tests


def test_e2e_simple_conversation(system):
    """Test simple conversation flow."""
    controller, _ = system

    response = controller.process_message("Hello")

    assert response is not None
    assert len(response) > 0
    assert isinstance(response, str)


def test_e2e_multiple_turns(system):
    """Test multiple conversation turns."""
    controller, _ = system

    response1 = controller.process_message("What is your name?")
    response2 = controller.process_message("What is your experience?")

    assert response1 is not None
    assert response2 is not None
    assert isinstance(response1, str)
    assert isinstance(response2, str)
    assert len(response1) > 0
    assert len(response2) > 0


def test_e2e_analytics(system):
    """Test analytics after conversations."""
    controller, _ = system

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

    # First message
    response1 = controller.process_message("Hello, my name is John")
    assert response1 is not None

    # Second message should have context from first
    response2 = controller.process_message("What's my name?")
    assert response2 is not None

    # Both responses should be different (indicating context awareness)
    assert response1 != response2


def test_e2e_tool_integration(system):
    """Test tool calling integration."""
    controller, _ = system

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
