

from tools.base import BaseTool


class ToolRegistry:

  def __init__(self):
    self._tools: dict[str, BaseTool] = {}


  def register(self, tool: BaseTool):
    self._tools[tool.name] = tool


  def get(self, name: str) -> BaseTool | None:
    return self._tools.get(name)

  def get_all(self) -> list[BaseTool]:
    return list(self._tools.values())

  def to_openai_format(self) -> list[dict]:
    return [tool.to_openai_format() for tool in self._tools.values()]
