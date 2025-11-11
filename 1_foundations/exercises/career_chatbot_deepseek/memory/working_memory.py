
from config.settings import settings


class WorkingMemory:
    """Manages recent conversation turns.

    Example:
        >>> memory = WorkingMemory()
        >>> memory.add_turn("Hello", "Hi!")
        >>> context = memory.get_context()
    """

    def __init__(self, max_turns: int = None):
        self.max_runs = max_turns or settings.working_memory_max_turns
        self.turns: list[dict] = []

    def add_turn(self, user_message: str, agent_response: str):
        """Add conversation turn to memory.

        Args:
            user_message: User's message
            agent_response: Agent's response
        """
        self.turns.append({"role": "user", "content": user_message})
        self.turns.append({"role": "assistant", "content": agent_response})

        if len(self.turns) > self.max_runs * 2:
            self.turns = self.turns[-(self.max_runs * 2) :]

    def get_context(self) -> list[dict]:
        """Get conversation context for LLM.

        Returns:
            List of message dictionaries
        """
        return self.turns.copy()

    def clear(self):
        """Clear all turns from memory."""
        self.turns = []
