"""Integration tests for controller."""

from unittest.mock import Mock, patch

import pytest

from core.controller import ChatbotController
from models.schemas import Evaluation


@pytest.fixture
def mock_components():
    """Create mock components."""
    career_agent = Mock()
    career_agent.chat.return_value = "Test response"

    evaluator = Mock()
    evaluator.evaluate.return_value = Evaluation(
        acceptable=True, feedback="Good", score=8
    )

    db_service = Mock()
    db_service.save_conversation.return_value = 1

    memory_service = Mock()
    memory_service.build_context.return_value = []

    return career_agent, evaluator, db_service, memory_service


def test_controller_init(mock_components):
    """Test controller initialization."""
    agent, evaluator, db, memory = mock_components

    controller = ChatbotController(agent, evaluator, db, memory)

    assert controller.career_agent is agent
    assert controller.evaluator_agent is evaluator
    assert controller.database_service is db
    assert controller.memory_service is memory


def test_controller_process_message(mock_components):
    """Test processing message."""
    agent, evaluator, db, memory = mock_components

    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("Hello")

    assert response == "Test response"
    agent.chat.assert_called_once()
    db.save_conversation.assert_called_once()
    memory.store_interaction.assert_called_once_with("Hello", "Test response")


@patch("core.controller.settings")
def test_controller_with_evaluation_enabled(mock_settings, mock_components):
    """Test processing with evaluation enabled."""
    agent, evaluator, db, memory = mock_components
    mock_settings.enable_evaluation = True

    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("Test")

    assert response == "Test response"
    evaluator.evaluate.assert_called_once_with("Test", "Test response")
    db.save_conversation.assert_called_once()


@patch("core.controller.settings")
def test_controller_with_evaluation_disabled(mock_settings, mock_components):
    """Test processing with evaluation disabled."""
    agent, evaluator, db, memory = mock_components
    mock_settings.enable_evaluation = False

    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("Test")

    assert response == "Test response"
    evaluator.evaluate.assert_not_called()


@patch("core.controller.settings")
def test_controller_regenerate_on_bad_evaluation(mock_settings, mock_components):
    """Test regeneration when evaluation fails."""
    agent, evaluator, db, memory = mock_components
    mock_settings.enable_evaluation = True

    # First evaluation fails, second passes
    evaluator.evaluate.side_effect = [
        Evaluation(acceptable=False, feedback="Too short", score=3),
        Evaluation(acceptable=True, feedback="Better", score=7),
    ]

    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("Test")

    assert response == "Test response"
    assert agent.chat.call_count == 2  # Original + regeneration
    assert evaluator.evaluate.call_count == 2

    # Verify second call includes feedback
    second_call_args = agent.chat.call_args_list[1]
    assert "Too short" in second_call_args[0][0]
    assert "Previous response feedback" in second_call_args[0][0]


@patch("core.controller.settings")
def test_controller_stores_evaluation_score(mock_settings, mock_components):
    """Test that evaluation score is stored in database."""
    agent, evaluator, db, memory = mock_components
    mock_settings.enable_evaluation = True

    evaluator.evaluate.return_value = Evaluation(
        acceptable=True, feedback="Excellent", score=9
    )

    controller = ChatbotController(agent, evaluator, db, memory)
    controller.process_message("Test")

    # Verify save_conversation was called with evaluation score
    db.save_conversation.assert_called_once_with(
        user_message="Test", agent_response="Test response", evaluation_score=9
    )


@patch("core.controller.settings")
def test_controller_no_evaluation_score_when_disabled(mock_settings, mock_components):
    """Test that evaluation score is None when evaluation disabled."""
    agent, evaluator, db, memory = mock_components
    mock_settings.enable_evaluation = False

    controller = ChatbotController(agent, evaluator, db, memory)
    controller.process_message("Test")

    # Verify save_conversation was called with None evaluation score
    db.save_conversation.assert_called_once_with(
        user_message="Test", agent_response="Test response", evaluation_score=None
    )


def test_controller_with_no_evaluator(mock_components):
    """Test controller with None evaluator."""
    agent, _, db, memory = mock_components

    controller = ChatbotController(agent, None, db, memory)
    response = controller.process_message("Test")

    assert response == "Test response"
    assert controller.enable_evaluation is False


def test_controller_memory_integration(mock_components):
    """Test memory service integration."""
    agent, evaluator, db, memory = mock_components

    # Mock memory context
    memory.build_context.return_value = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    controller = ChatbotController(agent, evaluator, db, memory)
    controller.process_message("Follow-up question")

    # Verify memory was queried
    memory.build_context.assert_called_once_with("Follow-up question")

    # Verify history was passed to agent
    agent.chat.assert_called_once()
    call_args = agent.chat.call_args
    assert call_args[0][1] == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]


def test_controller_get_analytics(mock_components):
    """Test get_analytics method."""
    agent, evaluator, db, memory = mock_components

    db.get_analytics.return_value = {
        "total_conversations": 10,
        "average_score": 8.5,
    }

    controller = ChatbotController(agent, evaluator, db, memory)
    analytics = controller.get_analytics()

    assert analytics["total_conversations"] == 10
    assert analytics["average_score"] == 8.5
    db.get_analytics.assert_called_once()


@patch("core.controller.settings")
def test_controller_full_flow(mock_settings, mock_components):
    """Test full controller flow with all steps."""
    agent, evaluator, db, memory = mock_components
    mock_settings.enable_evaluation = True

    # Setup mock returns
    memory.build_context.return_value = [{"role": "user", "content": "Hi"}]
    agent.chat.return_value = "Complete response"
    evaluator.evaluate.return_value = Evaluation(
        acceptable=True, feedback="Perfect", score=10
    )

    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("What are your skills?")

    # Verify full flow
    assert response == "Complete response"

    # 1. Memory context was built
    memory.build_context.assert_called_once_with("What are your skills?")

    # 2. Agent generated response
    agent.chat.assert_called_once()

    # 3. Response was evaluated
    evaluator.evaluate.assert_called_once_with(
        "What are your skills?", "Complete response"
    )

    # 4. Conversation was stored
    db.save_conversation.assert_called_once_with(
        user_message="What are your skills?",
        agent_response="Complete response",
        evaluation_score=10,
    )

    # 5. Memory was updated
    memory.store_interaction.assert_called_once_with(
        "What are your skills?", "Complete response"
    )

