"""Gradio application for Career Chatbot MVP."""

import os

import gradio as gr
from dotenv import load_dotenv

from config.settings import settings
from core.career_agent import CareerAgent
from core.controller import ChatbotController
from core.evaluator_agent import EvaluatorAgent
from memory.memory_service import MemoryService
from services.database_service import DatabaseService
from services.llm_service import LLMService
from services.notification_service import NotificationService
from services.rag_service import RAGService
from tools.contact_tool import ContactTool
from tools.question_tool import QuestionTool
from tools.registry import ToolRegistry
from tools.search_tool import SearchTool
from utils.file_loader import FileLoader

load_dotenv(override=True)

def initialize_system():
    """Initialize all system components.

    Returns:
        Tuple of (controller, rag_service)
    """
    db_service = DatabaseService()
    llm_service = LLMService()
    rag_service = RAGService()
    notif_service = NotificationService()
    memory_service = MemoryService()

    loader = FileLoader()
    linkedin = loader.load_pdf(f"{settings.kb_path}/linkedin.pdf")
    summary = loader.load_text(f"{settings.kb_path}/summary.md")

    if settings.enable_rag:
        rag_service.ingest_text(linkedin, "linkedin_profile")
        rag_service.ingest_text(summary, "career_summary")

    tool_registry = ToolRegistry()
    tool_registry.register(ContactTool(db_service, notif_service))
    tool_registry.register(QuestionTool(db_service))
    if settings.enable_rag:
        tool_registry.register(SearchTool(rag_service))

    career_agent = CareerAgent(
        llm_service=llm_service,
        rag_service=rag_service,
        tool_registry=tool_registry,
        linkedin_profile=linkedin,
        career_summary=summary
    )

    evaluator_agent = None
    if settings.enable_evaluation and settings.google_api_key:
        try:
            evaluator_agent = EvaluatorAgent()
        except Exception as e:
            print(f"Warning: Could not initialize evaluator: {e}")

    controller = ChatbotController(
        career_agent=career_agent,
        evaluator_agent=evaluator_agent,
        database_service=db_service,
        memory_service=memory_service
    )

    return controller, rag_service


print("Initializing Career Chatbot...")
controller, rag_service = initialize_system()
print("✓ System initialized successfully!")


def chat(message: str, history: list) -> str:
    """Gradio chat function.

    Args:
        message: User's message
        history: Gradio chat history (not used, we use our own memory)

    Returns:
        Agent's response
    """
    try:
        response = controller.process_message(message)
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title=f"💼 {settings.agent_name} - Career Assistant",
    description=f"""
    Hi! I'm {settings.agent_name}'s AI career assistant. I can answer questions about:
    - Professional experience and background
    - Skills and expertise
    - Career achievements
    - How to get in touch

    Feel free to ask me anything!
    """,
    examples=[
        "What is your professional background?",
        "What are your key skills?",
        "What projects have you worked on?",
        "How can I contact you?",
    ],
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
