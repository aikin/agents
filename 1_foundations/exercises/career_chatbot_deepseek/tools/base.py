"""Base tool class for function calling."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Base class for all tools.

    Subclasses must implement:
    - name: Tool name
    - description: Tool description
    - parameters: JSON schema for parameters
    - execute: Tool execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> dict[str, Any]:
        """Execute tool with given parameters.

        Returns:
            Result dictionary with 'success' and 'message' keys
        """
        pass

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
