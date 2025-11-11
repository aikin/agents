
from memory.memory_service import MemoryService
from memory.working_memory import WorkingMemory


def test_working_memory_add_turn():
    """Test adding turn to working memory."""
    memory = WorkingMemory(max_turns=5)

    memory.add_turn("Hello", "Hi!")
    context = memory.get_context()

    assert len(context) == 2
    assert context[0]["role"] == "user"
    assert context[1]["role"] == "assistant"


def test_working_memory_max_turns():
    """Test max turns limit."""
    memory = WorkingMemory(max_turns=2)

    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")
    memory.add_turn("Q3", "A3")

    context = memory.get_context()
    assert len(context) == 4  # 2 turns * 2 messages


def test_working_memory_clear():
    """Test clearing memory."""
    memory = WorkingMemory()
    memory.add_turn("Hello", "Hi!")
    memory.clear()

    assert len(memory.get_context()) == 0


def test_memory_service_store_interaction():
    """Test storing interaction."""
    service = MemoryService()

    service.store_interaction("Hello", "Hi!")
    context = service.build_context("How are you?")

    assert len(context) == 2


def test_memory_service_clear():
    """Test clearing memory service."""
    service = MemoryService()
    service.store_interaction("Hello", "Hi!")
    service.clear()

    assert len(service.build_context("Test")) == 0
