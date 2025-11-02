from typing import Any

from services.database_service import DatabaseService
from tools.base import BaseTool


class QuestionTool(BaseTool):

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    @property
    def name(self) -> str:
        return "record_unknown_question"

    @property
    def description(self) -> str:
        return "Record questions that cannot be answered"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The unknown question"}
            },
            "required": ["question"]
        }

    def execute(self, **kwargs) -> dict[str, Any]:
      question = kwargs.get("question", "")
      try:
        self.db_service.record_unknown_question(question)
        return {"success": True, "message": "Question recorded"}
      except Exception as e:
          return {"success": False, "message": str(e)}
