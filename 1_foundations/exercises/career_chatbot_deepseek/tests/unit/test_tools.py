from tools.base import BaseTool
from tools.registry import ToolRegistry


class MockTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            }
        }

    def execute(self, **kwargs):
        return {"success": True, "message": "Mock executed"}



def test_tool_to_openai_format():
    """Test tool conversion to OpenAI format."""
    tool = MockTool()
    format = tool.to_openai_format()

    assert format["type"] == "function"
    assert format["function"]["name"] == "mock_tool"


def test_registry_register():
    """Test registering tool."""
    registry = ToolRegistry()
    tool = MockTool()

    registry.register(tool)
    assert registry.get("mock_tool") is not None


def test_registry_get_all():
    """Test getting all tools."""
    registry = ToolRegistry()
    tool1 = MockTool()

    registry.register(tool1)
    tools = registry.get_all()

    assert len(tools) == 1


def test_registry_to_openai_format():
    """Test converting registry to OpenAI format."""
    registry = ToolRegistry()
    registry.register(MockTool())

    formats = registry.to_openai_format()
    assert len(formats) == 1
    assert formats[0]["type"] == "function"
