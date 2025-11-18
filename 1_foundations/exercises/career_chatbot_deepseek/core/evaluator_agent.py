"""Evaluator agent for response quality assessment."""

from textwrap import dedent

import google.generativeai as genai

from config.settings import settings
from models.schemas import Evaluation


class EvaluatorAgent:
  """Evaluator agent for response quality assessment."""


  def __init__(self, api_key: str | None = None):
    """Initialize evaluator agent."""
    genai.configure(api_key=api_key or settings.google_api_key)
    self.model = genai.GenerativeModel(settings.evaluator_model)

  def evaluate(self, question: str, answer: str) -> Evaluation:
    """Evaluate response quality.
    Args:
      question: User's question
      answer: Agent's answer
    Returns:
      Evaluation object with acceptable, feedback, score
    """

    prompt = dedent(f"""
        Evaluate the following Q&A exchange for a career chatbot.

        Question: {question}
        Answer: {answer}

        Evaluate based on: 
        1. Relevance: Does the answer address the question?
        2. Accuracy: Is the information correct?
        3. Completeness: Is the answer thorought?
        4. Professionalism: Is the tone appropriate?

        Response in JSON format:
        {{
            "acceptable": true/false,
            "feedback": "specific feedback",
            "score": 1-10
        }}
    """).strip()

    try:
      response =  self.model.generate_content(prompt)

      import json
      import re

      text = response.text
      json_match = re.search(r'```json\s*(.*?)\s```', text, re.DOTALL)
      if json_match:
        json_str = json_match.group(1)
      else:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        json_str = json_match.group(0) if json_match else text

      data = json.loads(json_str)
      return Evaluation(**data)

    except Exception as e:
      return Evaluation(
        acceptable=True,
        feedback=f"Evaluation failed: {str(e)}",
        score=5
      )
