

from unittest.mock import Mock, patch

import pytest

from core.evaluator_agent import EvaluatorAgent
from models.schemas import Evaluation


@pytest.fixture
def evaluator():
  """create evaluator with mocked Gemini."""
  with patch('core.evaluator_agent.genai'):
    return EvaluatorAgent()


def test_evaluator_evaluate_success(evaluator):

  mock_response = Mock()
  mock_response.text = '{"acceptable": true, "feedback": "Good answer", "score": 8}'
  evaluator.model.generate_content = Mock(return_value=mock_response)

  eval = evaluator.evaluate("What is AI?", "AI is artificial intelligence")

  assert isinstance(eval, Evaluation)
  assert eval.acceptable is True
  assert eval.score == 8



def test_evaluator_evaluate_with_markdown(evaluator):
  mock_response = Mock()
  mock_response.text = '```json\n{"acceptable": true, "feedback": "Great", "score": 9}\n```'
  evaluator.model.generate_content = Mock(return_value=mock_response)

  eval = evaluator.evaluate("Test", "Test answer")

  assert eval.acceptable is True
  assert eval.score == 9


def test_evaluator_evaluate_fallback(evaluator):
  evaluator.model.generate_content = Mock(side_effect=Exception("API Error"))

  eval = evaluator.evaluate("Test", "Test")

  assert isinstance(eval, Evaluation)
  assert eval.acceptable is True
  assert eval.score == 5
