# Sprint 5: Integration & Deployment

**Sprint Goal:** Wire everything together and deploy to HuggingFace Spaces  
**Deliverable:** Deployed MVP on HuggingFace Spaces  
**Duration:** ~1-2 weeks

---

## Overview

This final sprint brings everything together:
- **Controller:** Orchestrates all components
- **app.py:** Gradio interface with dependency injection
- **Integration Tests:** End-to-end testing
- **Deployment:** HuggingFace Spaces deployment

**Why this matters:** This is where all the pieces come together into a working MVP.

---

## Task 5.1: Controller Implementation

**Estimated Time:** 3-4 hours  
**Priority:** P0

### Goal
Create a controller that orchestrates all components following the Evaluator-Optimizer pattern.

### Implementation

Create `core/controller.py`:

```python
"""Controller for orchestrating the career chatbot system."""

from typing import List, Dict, Optional
from core.career_agent import CareerAgent
from core.evaluator_agent import EvaluatorAgent
from services.database_service import DatabaseService
from memory.memory_service import MemoryService
from config.settings import settings


class ChatbotController:
    """Main controller orchestrating all components.
    
    Implements the Evaluator-Optimizer pattern:
    1. Generate response (Career Agent)
    2. Evaluate response (Evaluator Agent)
    3. Regenerate if needed (Optimizer)
    4. Store results (Database)
    
    Example:
        >>> controller = ChatbotController(agent, evaluator, db, memory)
        >>> response = controller.process_message("What is your experience?")
    """
    
    def __init__(
        self,
        career_agent: CareerAgent,
        evaluator_agent: Optional[EvaluatorAgent],
        database_service: DatabaseService,
        memory_service: MemoryService
    ):
        """Initialize controller.
        
        Args:
            career_agent: Career agent instance
            evaluator_agent: Optional evaluator agent
            database_service: Database service instance
            memory_service: Memory service instance
        """
        self.career_agent = career_agent
        self.evaluator_agent = evaluator_agent
        self.database_service = database_service
        self.memory_service = memory_service
        self.enable_evaluation = settings.enable_evaluation and evaluator_agent is not None
    
    def process_message(self, message: str) -> str:
        """Process user message and return response.
        
        Implements the full agentic flow:
        1. Get conversation context from memory
        2. Generate response with career agent
        3. Evaluate response (if enabled)
        4. Regenerate if not acceptable (one retry)
        5. Store conversation in database
        6. Update memory
        
        Args:
            message: User's message
            
        Returns:
            Agent's response
        """
        # 1. Get context from memory
        history = self.memory_service.build_context(message)
        
        # 2. Generate response
        response = self.career_agent.chat(message, history)
        
        # 3. Evaluate response
        evaluation_score = None
        if self.enable_evaluation:
            evaluation = self.evaluator_agent.evaluate(message, response)
            evaluation_score = evaluation.score
            
            # 4. Regenerate if not acceptable (one retry)
            if not evaluation.acceptable:
                response = self.career_agent.chat(
                    f"{message}\n\nPrevious response feedback: {evaluation.feedback}",
                    history
                )
                # Re-evaluate
                evaluation = self.evaluator_agent.evaluate(message, response)
                evaluation_score = evaluation.score
        
        # 5. Store conversation
        self.database_service.save_conversation(
            user_message=message,
            agent_response=response,
            evaluation_score=evaluation_score
        )
        
        # 6. Update memory
        self.memory_service.store_interaction(message, response)
        
        return response
    
    def get_analytics(self) -> Dict:
        """Get usage analytics.
        
        Returns:
            Analytics dictionary
        """
        return self.database_service.get_analytics()
```

### Tests

Create `tests/integration/test_controller.py`:

```python
"""Integration tests for controller."""

import pytest
from unittest.mock import Mock
from core.controller import ChatbotController
from models.schemas import Evaluation


@pytest.fixture
def mock_components():
    """Create mock components."""
    career_agent = Mock()
    career_agent.chat.return_value = "Test response"
    
    evaluator = Mock()
    evaluator.evaluate.return_value = Evaluation(
        acceptable=True,
        feedback="Good",
        score=8
    )
    
    db_service = Mock()
    db_service.save_conversation.return_value = 1
    
    memory_service = Mock()
    memory_service.build_context.return_value = []
    
    return career_agent, evaluator, db_service, memory_service


def test_controller_process_message(mock_components):
    """Test processing message."""
    agent, evaluator, db, memory = mock_components
    
    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("Hello")
    
    assert response == "Test response"
    agent.chat.assert_called_once()
    db.save_conversation.assert_called_once()
    memory.store_interaction.assert_called_once()


def test_controller_with_evaluation(mock_components):
    """Test processing with evaluation."""
    agent, evaluator, db, memory = mock_components
    
    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("Test")
    
    evaluator.evaluate.assert_called_once()


def test_controller_regenerate_on_bad_evaluation(mock_components):
    """Test regeneration when evaluation fails."""
    agent, evaluator, db, memory = mock_components
    
    # First evaluation fails, second passes
    evaluator.evaluate.side_effect = [
        Evaluation(acceptable=False, feedback="Too short", score=3),
        Evaluation(acceptable=True, feedback="Better", score=7)
    ]
    
    controller = ChatbotController(agent, evaluator, db, memory)
    response = controller.process_message("Test")
    
    assert agent.chat.call_count == 2  # Original + regeneration
    assert evaluator.evaluate.call_count == 2
```

### Test & Commit

```bash
pytest tests/integration/test_controller.py -v

git add core/controller.py tests/integration/test_controller.py
git commit -m "feat(core): implement controller with Evaluator-Optimizer pattern

- Add ChatbotController orchestrating all components
- Implement evaluation and regeneration flow
- Add integration tests

Relates to Sprint 5, Task 5.1"
```

---

## Task 5.2: App.py Integration

**Estimated Time:** 2-3 hours  
**Priority:** P0

### Goal
Create the Gradio interface that wires everything together.

### Implementation

Create `app.py`:

```python
"""Gradio application for Career Chatbot MVP."""

import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Import components
from config.settings import settings
from services.database_service import DatabaseService
from services.llm_service import LLMService
from services.rag_service import RAGService
from services.notification_service import NotificationService
from tools.registry import ToolRegistry
from tools.contact_tool import ContactTool
from tools.question_tool import QuestionTool
from tools.search_tool import SearchTool
from core.career_agent import CareerAgent
from core.evaluator_agent import EvaluatorAgent
from core.controller import ChatbotController
from memory.memory_service import MemoryService
from utils.file_loader import FileLoader


def initialize_system():
    """Initialize all system components.
    
    Returns:
        Tuple of (controller, rag_service)
    """
    # 1. Initialize services
    db_service = DatabaseService()
    llm_service = LLMService()
    rag_service = RAGService()
    notif_service = NotificationService()
    memory_service = MemoryService()
    
    # 2. Load knowledge base
    loader = FileLoader()
    linkedin = loader.load_pdf(f"{settings.kb_path}/linkedin.pdf")
    summary = loader.load_text(f"{settings.kb_path}/summary.txt")
    
    # 3. Ingest into RAG
    if settings.enable_rag:
        rag_service.ingest_text(linkedin, "linkedin_profile")
        rag_service.ingest_text(summary, "career_summary")
    
    # 4. Initialize tools
    tool_registry = ToolRegistry()
    tool_registry.register(ContactTool(db_service, notif_service))
    tool_registry.register(QuestionTool(db_service))
    if settings.enable_rag:
        tool_registry.register(SearchTool(rag_service))
    
    # 5. Initialize agents
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
    
    # 6. Initialize controller
    controller = ChatbotController(
        career_agent=career_agent,
        evaluator_agent=evaluator_agent,
        database_service=db_service,
        memory_service=memory_service
    )
    
    return controller, rag_service


# Initialize system
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


# Create Gradio interface
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
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear Conversation",
)


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Launch app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

### Test Manually

```bash
# Run the app
python app.py

# Open browser to http://localhost:7860
# Test conversation flow
# Test tool calling (contact, unknown question)
# Test RAG search
```

### Commit

```bash
git add app.py
git commit -m "feat: implement Gradio app with full system integration

- Add app.py with dependency injection
- Initialize all components
- Ingest knowledge base into RAG
- Register all tools
- Create Gradio chat interface

Relates to Sprint 5, Task 5.2"
```

---

## Task 5.3: End-to-End Testing

**Estimated Time:** 2 hours  
**Priority:** P1

### Implementation

Create `tests/integration/test_e2e.py`:

```python
"""End-to-end integration tests."""

import pytest
import os
import tempfile
import shutil
from app import initialize_system


@pytest.fixture
def system():
    """Initialize system for testing."""
    # Use temporary directories
    temp_data = tempfile.mkdtemp()
    temp_chroma = tempfile.mkdtemp()
    
    os.environ["DB_PATH"] = f"{temp_data}/test.db"
    os.environ["CHROMA_PATH"] = temp_chroma
    os.environ["ENABLE_EVALUATION"] = "false"  # Disable for faster tests
    
    controller, rag = initialize_system()
    
    yield controller, rag
    
    # Cleanup
    shutil.rmtree(temp_data)
    shutil.rmtree(temp_chroma)


def test_e2e_simple_conversation(system):
    """Test simple conversation flow."""
    controller, _ = system
    
    response = controller.process_message("Hello")
    
    assert response is not None
    assert len(response) > 0


def test_e2e_multiple_turns(system):
    """Test multiple conversation turns."""
    controller, _ = system
    
    response1 = controller.process_message("What is your name?")
    response2 = controller.process_message("What is your experience?")
    
    assert response1 is not None
    assert response2 is not None


def test_e2e_analytics(system):
    """Test analytics after conversations."""
    controller, _ = system
    
    controller.process_message("Test question 1")
    controller.process_message("Test question 2")
    
    analytics = controller.get_analytics()
    
    assert analytics["total_conversations"] >= 2
```

### Test & Commit

```bash
pytest tests/integration/test_e2e.py -v

git add tests/integration/test_e2e.py
git commit -m "test: add end-to-end integration tests

- Add E2E tests for conversation flow
- Test multiple turns
- Test analytics

Relates to Sprint 5, Task 5.3"
```

---

## Task 5.4: HuggingFace Deployment

**Estimated Time:** 1-2 hours  
**Priority:** P0

### Goal
Deploy the MVP to HuggingFace Spaces.

### Steps

#### 1. Create requirements.txt for HF

Create `requirements.txt` in project root:

```txt
# Core dependencies
openai>=1.0.0
gradio>=4.0.0
pypdf>=3.0.0
python-dotenv>=1.0.0
requests>=2.31.0

# Data & Validation
pydantic>=2.0.0
pydantic-settings>=2.0.0
sqlalchemy>=2.0.0

# RAG
chromadb>=0.4.22
sentence-transformers>=2.2.0

# Evaluator (optional)
google-generativeai>=0.3.0

# Testing (for CI)
pytest>=7.4.0
pytest-cov>=4.1.0
```

#### 2. Create README.md

Create `README.md`:

```markdown
# Career Chatbot MVP

An intelligent career assistant powered by DeepSeek, with RAG, tool calling, and quality evaluation.

## Features

- 💬 **Conversational AI** - Natural career Q&A
- 🔍 **RAG** - Knowledge base retrieval with Chroma
- 🛠️ **Tool Calling** - Contact recording, unknown questions
- ✅ **Quality Evaluation** - Gemini-powered response assessment
- 💾 **Persistence** - SQLite database for conversations
- 🧠 **Memory** - Working memory for context

## Architecture

Built with clean code principles and modular design:
- **Config Layer** - Pydantic Settings
- **Services Layer** - Database, LLM, RAG, Notifications
- **Tools Layer** - Function calling capabilities
- **Core Layer** - Agents and controller
- **Memory Layer** - Conversation context management

## Setup

### Environment Variables

Create a `.env` file:

```bash
DEEPSEEK_API_KEY=your-deepseek-key
GOOGLE_API_KEY=your-google-key  # Optional for evaluation
PUSHOVER_USER=your-pushover-user  # Optional for notifications
PUSHOVER_TOKEN=your-pushover-token
```

### Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Deployment

Deployed on HuggingFace Spaces with Gradio.

## License

MIT
```

#### 3. Deploy to HuggingFace

```bash
# 1. Create Space on HuggingFace
# - Go to https://huggingface.co/new-space
# - Name: career-chatbot-mvp
# - SDK: Gradio
# - Hardware: CPU Basic (free)

# 2. Clone the Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/career-chatbot-mvp
cd career-chatbot-mvp

# 3. Copy files from your project
cp -r ../career_chatbot_deepseek/* .

# 4. Add .env secrets in HF Space settings
# - DEEPSEEK_API_KEY
# - GOOGLE_API_KEY (optional)
# - PUSHOVER_USER (optional)
# - PUSHOVER_TOKEN (optional)

# 5. Commit and push
git add .
git commit -m "Initial deployment"
git push
```

#### 4. Test Deployment

```bash
# Visit your Space URL
# Test all features:
# - Simple conversation
# - Contact recording
# - Unknown question recording
# - RAG search
# - Evaluation (if enabled)
```

### Commit

```bash
git add requirements.txt README.md
git commit -m "docs: add deployment files for HuggingFace Spaces

- Add requirements.txt for HF
- Add comprehensive README
- Document deployment process

Relates to Sprint 5, Task 5.4"
```

---

## Sprint 5 Completion

### Final Verification

```bash
# 1. Run all tests
pytest tests/ -v

# 2. Check coverage
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# 3. Run app locally
python app.py
# Test all features manually

# 4. Verify deployment
# Visit HuggingFace Space URL
# Test in production
```

### MVP Deliverables Checklist

- [x] **Sprint 1:** Foundation & Data Models
- [x] **Sprint 2:** Services Layer
- [x] **Sprint 3:** RAG & Tools
- [x] **Sprint 4:** Agents & Memory
- [x] **Sprint 5:** Integration & Deployment

**Feature Checklist:**
- [x] Core conversation with DeepSeek
- [x] Basic RAG with Chroma
- [x] Working Memory
- [x] Tool calling (contact, unknown question, search)
- [x] Pushover notifications
- [x] Gradio interface
- [x] SQLite persistence
- [x] Basic evaluator with Gemini
- [x] Configuration management
- [x] Unit tests (70%+ coverage)
- [x] Integration tests
- [x] Deployed to HuggingFace Spaces

### Success Metrics

- ✅ All existing functionality preserved
- ✅ Modular architecture achieved
- ✅ 70%+ test coverage
- ✅ Deployed to HuggingFace Spaces
- ✅ Response time < 3s
- ✅ Zero breaking changes

---

## 🎉 MVP Complete!

**Congratulations!** You've successfully built and deployed a production-ready career chatbot with:

- Clean, modular architecture
- Agentic AI patterns (Tool Use, Reflection, RAG)
- Comprehensive testing
- Production deployment

### What's Next?

**Post-MVP Enhancements:**
1. **Advanced Memory** - Episodic, Semantic, Procedural layers
2. **Multi-LLM Support** - Add Anthropic, more models
3. **Streaming** - Real-time response streaming
4. **Analytics Dashboard** - Visualize usage data
5. **Multi-user** - Session management
6. **Advanced RAG** - Better chunking, hybrid search

### Retrospective

**What went well:**
- 

**What could be improved:**
- 

**Lessons learned:**
- 

**Key achievements:**
- Built production-ready MVP in 5 sprints
- Followed clean code principles throughout
- Comprehensive test coverage
- Successfully deployed

---

**🚀 MVP DEPLOYED!** ✅

Thank you for following this implementation plan. Your career chatbot is now live and ready to serve users!

