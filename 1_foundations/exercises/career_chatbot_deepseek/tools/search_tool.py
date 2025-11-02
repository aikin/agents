from typing import Any

from services.rag_service import RAGService
from tools.base import BaseTool


class SearchTool(BaseTool):
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return "Search the knowledge base for relevant information"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }

    def execute(self, **kwargs) -> dict[str, Any]:
        query = kwargs.get("query", "")
        try:
            results = self.rag_service.search(query)
            return {"success": True, "message": f"Found {len(results)} results", "results": results}
        except Exception as e:
            return {"success": False, "message": str(e), "results": []}
