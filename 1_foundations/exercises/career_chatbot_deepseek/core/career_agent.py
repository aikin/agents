
from config.settings import settings
from services.llm_service import LLMService
from services.rag_service import RAGService
from tools.registry import ToolRegistry
from utils.prompt_builder import PromptBuilder


class CareerAgent:

    def __init__(
        self,
        llm_service: LLMService,
        rag_service: RAGService,
        tool_registry: ToolRegistry,
        linkedin_profile: str,
        career_summary: str,
    ):
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.tool_registry = tool_registry
        self.system_prompt = PromptBuilder.build_system_prompt(
            name=settings.agent_name,
            linkedin_profile=linkedin_profile,
            summary=career_summary,
            enable_rag=settings.enable_rag,
        )

    def chat(self, message: str, history: list[dict]) -> str:
        """Process user message and generate response.

        Args:
            message: User's message
            history: Conversation history

        Returns:
            Agent's response
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": message},
        ]

        # Agentic loop: allow multiple tool calls
        max_iterations = 5
        for _ in range(max_iterations):
            response = self.llm_service.chat_completion(
                messages=messages,
                tools=self.tool_registry.to_openai_format()
            )

            finish_reason = response.choices[0].finish_reason
            assistant_message = response.choices[0].message

            if finish_reason == "tool_calls":
                messages.append(assistant_message)
                tool_results = self._execute_tools(assistant_message.tool_calls)
                messages.extend(tool_results)
            else:
                content = assistant_message.content
                return content if content is not None else ""

        return "I apologize, but I'm having trouble processing your request."


    def _execute_tools(self, tool_calls) -> list[dict]:
        """Execute tool calls and format results.

        Args:
            tool_calls: List of tool calls from LLM

        Returns:
            List of tool result messages
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool = self.tool_registry.get(tool_name)

            if tool:
                import json
                args = json.loads(tool_call.function.arguments)
                result = tool.execute(**args)

                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            else:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "success": False,
                        "message": "Tool not found"
                    })
                })

        return results
