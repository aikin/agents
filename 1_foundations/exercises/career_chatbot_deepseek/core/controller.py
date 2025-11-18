from config.settings import settings
from core.career_agent import CareerAgent
from core.evaluator_agent import EvaluatorAgent
from memory.memory_service import MemoryService
from services.database_service import DatabaseService


class ChatbotController:
    """Main controller orchestrating all components.

    Impletements the Evaluator-Optimizer patterns:
    1. Generate response (Career Agent)
    2. Evaluate response (Evaluator Agent)
    3. Regenerate if needed (Optimizer)
    4. Store results (Database)
    """

    def __init__(
        self,
        career_agent: CareerAgent,
        evaluator_agent: EvaluatorAgent | None,
        database_service: DatabaseService,
        memory_service: MemoryService,
    ):
        self.career_agent = career_agent
        self.evaluator_agent = evaluator_agent
        self.database_service = database_service
        self.memory_service = memory_service
        self.enable_evaluation = settings.enable_evaluation and evaluator_agent is not None

    def process_message(self, message: str) -> str:
        """
        Impletements the full agentic flow:
        1. Get converstation content from memory
        2. Generate response with career agent
        3. Evaluate response (if enabled)
        4. Regenerate if not acceptable (one retry)
        5. Store conversation in database
        6. Update memory
        """

        history = self.memory_service.build_context(message)

        response = self.career_agent.chat(message, history)

        evaluation_score = None

        if self.enable_evaluation:
            evaluation = self.evaluator_agent.evaluate(message, response)
            evaluation_score = evaluation.score

            if not evaluation.acceptable:
                response = self.career_agent.chat(
                    f"{message}\n\nPrevious response feedback: {evaluation.feedback}", history
                )
                evaluation = self.evaluator_agent.evaluate(message, response)
                evaluation_score = evaluation.score

        self.database_service.save_conversation(
            user_message=message, agent_response=response, evaluation_score=evaluation_score
        )

        self.memory_service.store_interaction(message, response)

        return response


    def get_analytics(self) -> dict:
      return self.database_service.get_analytics()
