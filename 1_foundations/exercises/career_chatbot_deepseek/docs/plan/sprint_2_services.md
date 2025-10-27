# Sprint 2: Services Layer

**Sprint Goal:** Implement core services (Database, LLM, Notification, Utils)  
**Deliverable:** Reusable service layer with tests  
**Duration:** ~1-2 weeks

---

## Overview

This sprint builds the service layer that encapsulates business logic:
- **Database Service:** CRUD operations and analytics
- **LLM Service:** DeepSeek API wrapper
- **Notification Service:** Pushover integration
- **Utils:** File loader and prompt builder

**Why this matters:** Services provide clean abstractions and make the codebase testable.

---

## Prerequisites

```bash
# Ensure Sprint 1 is complete
uv run pytest tests/unit/test_settings.py tests/unit/test_schemas.py tests/unit/test_database.py

# All tests should pass before starting Sprint 2
```

---

## Task 2.1: Database Service

**Estimated Time:** 3-4 hours  
**Priority:** P0

### Goal
Centralize all database operations in a single service class.

### Implementation

Create `services/database_service.py`:

```python
"""Database service for SQL operations.

Provides a clean interface for all database operations,
following the Repository pattern.
"""

from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import json

from models.database import Base, Conversation, Contact, UnknownQuestion
from config.settings import settings


class DatabaseService:
    """Centralized database operations.
    
    This service handles all SQL operations, providing a clean
    interface and automatic session management.
    
    Example:
        >>> db = DatabaseService()
        >>> conv_id = db.save_conversation("Hello", "Hi!")
        >>> analytics = db.get_analytics()
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize database service.
        
        Args:
            db_url: Optional database URL. Defaults to settings.db_path
        """
        self.db_url = db_url or f"sqlite:///{settings.db_path}"
        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions.
        
        Automatically commits on success, rolls back on error.
        
        Example:
            >>> with db.get_session() as session:
            ...     session.add(conversation)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def save_conversation(
        self,
        user_message: str,
        agent_response: str,
        evaluation_score: Optional[float] = None,
        context_used: Optional[List[str]] = None
    ) -> int:
        """Save conversation to database.
        
        Args:
            user_message: User's message
            agent_response: Agent's response
            evaluation_score: Optional quality score (1-10)
            context_used: Optional RAG context chunks
            
        Returns:
            Conversation ID
        """
        with self.get_session() as session:
            conv = Conversation(
                user_message=user_message,
                agent_response=agent_response,
                evaluation_score=evaluation_score,
                context_used=json.dumps(context_used) if context_used else None
            )
            session.add(conv)
            session.flush()
            return conv.id
    
    def save_contact(
        self,
        email: str,
        name: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """Save contact information.
        
        Args:
            email: Contact's email
            name: Contact's name
            notes: Additional notes
            
        Returns:
            Contact ID
        """
        with self.get_session() as session:
            contact = Contact(
                email=email,
                name=name or "Not provided",
                notes=notes or "Not provided"
            )
            session.add(contact)
            session.flush()
            return contact.id
    
    def record_unknown_question(self, question: str) -> None:
        """Record or increment unknown question count.
        
        If question already exists, increments count.
        Otherwise, creates new record.
        
        Args:
            question: The unknown question
        """
        with self.get_session() as session:
            existing = session.query(UnknownQuestion).filter_by(
                question=question
            ).first()
            
            if existing:
                existing.count += 1
                existing.last_asked = func.now()
            else:
                new_q = UnknownQuestion(question=question)
                session.add(new_q)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get usage analytics.
        
        Returns:
            Dictionary with analytics data
        """
        with self.get_session() as session:
            total_conversations = session.query(Conversation).count()
            total_contacts = session.query(Contact).count()
            
            top_unknown = session.query(UnknownQuestion).order_by(
                desc(UnknownQuestion.count)
            ).limit(10).all()
            
            avg_score = session.query(
                func.avg(Conversation.evaluation_score)
            ).scalar() or 0
            
            return {
                "total_conversations": total_conversations,
                "total_contacts": total_contacts,
                "average_evaluation_score": round(avg_score, 2),
                "top_unknown_questions": [
                    {"question": q.question, "count": q.count}
                    for q in top_unknown
                ]
            }
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation dictionaries
        """
        with self.get_session() as session:
            convs = session.query(Conversation).order_by(
                desc(Conversation.timestamp)
            ).limit(limit).all()
            
            return [
                {
                    "id": c.id,
                    "user_message": c.user_message,
                    "agent_response": c.agent_response,
                    "score": c.evaluation_score,
                    "timestamp": c.timestamp.isoformat()
                }
                for c in convs
            ]
```

### Tests

Create `tests/unit/test_database_service.py`:

```python
"""Tests for database service."""

import pytest
from services.database_service import DatabaseService


@pytest.fixture
def db_service():
    """Create in-memory database service for testing."""
    return DatabaseService(db_url="sqlite:///:memory:")


def test_save_conversation(db_service):
    """Test saving conversation."""
    conv_id = db_service.save_conversation(
        user_message="Hello",
        agent_response="Hi!",
        evaluation_score=8.5
    )
    assert conv_id is not None
    assert conv_id > 0


def test_save_contact(db_service):
    """Test saving contact."""
    contact_id = db_service.save_contact(
        email="test@example.com",
        name="Test User"
    )
    assert contact_id is not None


def test_record_unknown_question_new(db_service):
    """Test recording new unknown question."""
    db_service.record_unknown_question("What is AI?")
    
    analytics = db_service.get_analytics()
    assert len(analytics["top_unknown_questions"]) == 1
    assert analytics["top_unknown_questions"][0]["count"] == 1


def test_record_unknown_question_increment(db_service):
    """Test incrementing unknown question count."""
    question = "What is AI?"
    db_service.record_unknown_question(question)
    db_service.record_unknown_question(question)
    
    analytics = db_service.get_analytics()
    assert analytics["top_unknown_questions"][0]["count"] == 2


def test_get_analytics(db_service):
    """Test getting analytics."""
    db_service.save_conversation("Q1", "A1", 8.0)
    db_service.save_conversation("Q2", "A2", 9.0)
    db_service.save_contact("test@example.com")
    
    analytics = db_service.get_analytics()
    assert analytics["total_conversations"] == 2
    assert analytics["total_contacts"] == 1
    assert analytics["average_evaluation_score"] == 8.5


def test_get_recent_conversations(db_service):
    """Test getting recent conversations."""
    db_service.save_conversation("Q1", "A1")
    db_service.save_conversation("Q2", "A2")
    
    recent = db_service.get_recent_conversations(limit=5)
    assert len(recent) == 2
    assert recent[0]["user_message"] == "Q2"  # Most recent first
```

### Test & Commit

```bash
pytest tests/unit/test_database_service.py -v

git add services/database_service.py tests/unit/test_database_service.py
git commit -m "feat(services): implement database service with CRUD operations

- Add DatabaseService with context manager for sessions
- Implement save_conversation, save_contact, record_unknown_question
- Add analytics queries
- Add comprehensive unit tests

Relates to Sprint 2, Task 2.1"
```

---

## Task 2.2: LLM Service

**Estimated Time:** 1-2 hours  
**Priority:** P0

### Implementation

Create `services/llm_service.py`:

```python
"""LLM service wrapper for DeepSeek.

Provides a clean interface for LLM operations,
abstracting away API details.
"""

from openai import OpenAI
from typing import List, Dict, Optional, Any
from config.settings import settings


class LLMService:
    """Wrapper for LLM operations.
    
    Example:
        >>> llm = LLMService()
        >>> response = llm.chat_completion([
        ...     {"role": "user", "content": "Hello"}
        ... ])
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM client.
        
        Args:
            api_key: Optional API key. Defaults to settings
            model: Optional model name. Defaults to settings
        """
        self.client = OpenAI(
            api_key=api_key or settings.deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model or settings.primary_model
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Any:
        """Create chat completion.
        
        Args:
            messages: List of message dictionaries
            tools: Optional tool definitions
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            OpenAI completion response
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if tools:
            kwargs["tools"] = tools
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        return self.client.chat.completions.create(**kwargs)
```

### Tests

Create `tests/unit/test_llm_service.py`:

```python
"""Tests for LLM service."""

import pytest
from unittest.mock import Mock, patch
from services.llm_service import LLMService


@pytest.fixture
def llm_service():
    """Create LLM service with mocked client."""
    with patch('services.llm_service.OpenAI'):
        return LLMService()


def test_chat_completion_basic(llm_service):
    """Test basic chat completion."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    
    llm_service.client.chat.completions.create = Mock(return_value=mock_response)
    
    messages = [{"role": "user", "content": "Hello"}]
    response = llm_service.chat_completion(messages)
    
    assert response is not None
    llm_service.client.chat.completions.create.assert_called_once()


def test_chat_completion_with_tools(llm_service):
    """Test chat completion with tools."""
    mock_response = Mock()
    llm_service.client.chat.completions.create = Mock(return_value=mock_response)
    
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test"}}]
    
    llm_service.chat_completion(messages, tools=tools)
    
    call_args = llm_service.client.chat.completions.create.call_args
    assert "tools" in call_args.kwargs
```

### Test & Commit

```bash
pytest tests/unit/test_llm_service.py -v

git add services/llm_service.py tests/unit/test_llm_service.py
git commit -m "feat(services): implement LLM service wrapper for DeepSeek

- Add LLMService with chat_completion method
- Support tool calling
- Add unit tests with mocking

Relates to Sprint 2, Task 2.2"
```

---

## Task 2.3: Notification Service

**Estimated Time:** 1-2 hours  
**Priority:** P1

### Implementation

Create `services/notification_service.py`:

```python
"""Notification service for Pushover integration."""

import requests
from typing import Optional
from config.settings import settings


class NotificationService:
    """Pushover notification service.
    
    Example:
        >>> notif = NotificationService()
        >>> notif.send("New contact: test@example.com")
    """
    
    def __init__(self):
        """Initialize notification service."""
        self.user = settings.pushover_user
        self.token = settings.pushover_token
        self.enabled = bool(self.user and self.token)
    
    def send(self, message: str, title: Optional[str] = None) -> bool:
        """Send notification via Pushover.
        
        Args:
            message: Notification message
            title: Optional notification title
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": self.token,
                    "user": self.user,
                    "message": message,
                    "title": title or "Career Chatbot"
                },
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
```

### Tests

Create `tests/unit/test_notification_service.py`:

```python
"""Tests for notification service."""

import pytest
from unittest.mock import patch, Mock
from services.notification_service import NotificationService


@pytest.fixture
def notif_service():
    """Create notification service."""
    with patch('services.notification_service.settings') as mock_settings:
        mock_settings.pushover_user = "test_user"
        mock_settings.pushover_token = "test_token"
        return NotificationService()


def test_send_notification_success(notif_service):
    """Test successful notification send."""
    with patch('services.notification_service.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = notif_service.send("Test message")
        
        assert result is True
        mock_post.assert_called_once()


def test_send_notification_disabled():
    """Test notification when disabled."""
    with patch('services.notification_service.settings') as mock_settings:
        mock_settings.pushover_user = None
        mock_settings.pushover_token = None
        
        notif = NotificationService()
        result = notif.send("Test")
        
        assert result is False
```

### Test & Commit

```bash
pytest tests/unit/test_notification_service.py -v

git add services/notification_service.py tests/unit/test_notification_service.py
git commit -m "feat(services): implement notification service for Pushover

- Add NotificationService with send method
- Handle disabled state gracefully
- Add unit tests with mocking

Relates to Sprint 2, Task 2.3"
```

---

## Task 2.4: Utils (File Loader & Prompt Builder)

**Estimated Time:** 2 hours  
**Priority:** P0

### Implementation

Create `utils/file_loader.py`:

```python
"""File loading utilities."""

from pypdf import PdfReader
from typing import Optional


class FileLoader:
    """Utility for loading files.
    
    Example:
        >>> loader = FileLoader()
        >>> text = loader.load_pdf("linkedin.pdf")
    """
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            reader = PdfReader(file_path)
            text = "\n".join(
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            )
            return text
        except Exception as e:
            return f"Error loading PDF: {e}"
    
    @staticmethod
    def load_text(file_path: str, encoding: str = "utf-8") -> str:
        """Load text from text file.
        
        Args:
            file_path: Path to text file
            encoding: File encoding
            
        Returns:
            File contents
        """
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            return f"Error loading text file: {e}"
```

Create `utils/prompt_builder.py`:

```python
"""Prompt building utilities."""

from textwrap import dedent
from typing import Optional


class PromptBuilder:
    """Utility for building prompts.
    
    Example:
        >>> builder = PromptBuilder()
        >>> prompt = builder.build_system_prompt("Kin Lu", "...", "...")
    """
    
    @staticmethod
    def build_system_prompt(
        name: str,
        linkedin_profile: str,
        summary: str,
        enable_rag: bool = True
    ) -> str:
        """Build system prompt for career agent.
        
        Args:
            name: Agent name
            linkedin_profile: LinkedIn profile text
            summary: Career summary text
            enable_rag: Whether RAG is enabled
            
        Returns:
            System prompt string
        """
        rag_instruction = ""
        if enable_rag:
            rag_instruction = dedent("""
                When answering questions, you can use the search_knowledge_base tool
                to retrieve relevant information from the knowledge base.
            """).strip()
        
        return dedent(f"""
            You are acting as {name}. You are answering questions on {name}'s
            website, particularly questions related to {name}'s career, background,
            skills and experience.
            
            {rag_instruction}
            
            Here is {name}'s LinkedIn profile:
            {linkedin_profile}
            
            Here is {name}'s career summary:
            {summary}
            
            Answer questions naturally and professionally. If you don't know
            something, be honest about it and use the record_unknown_question tool.
            
            If someone wants to get in touch, use the record_user_details tool
            to save their contact information.
        """).strip()
```

### Tests

Create `tests/unit/test_file_loader.py` and `tests/unit/test_prompt_builder.py`:

```python
# test_file_loader.py
"""Tests for file loader."""

import pytest
import tempfile
import os
from utils.file_loader import FileLoader


def test_load_text():
    """Test loading text file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content")
        temp_path = f.name
    
    try:
        loader = FileLoader()
        content = loader.load_text(temp_path)
        assert content == "Test content"
    finally:
        os.remove(temp_path)


def test_load_text_error():
    """Test loading non-existent file."""
    loader = FileLoader()
    content = loader.load_text("nonexistent.txt")
    assert "Error" in content


# test_prompt_builder.py
"""Tests for prompt builder."""

from utils.prompt_builder import PromptBuilder


def test_build_system_prompt():
    """Test building system prompt."""
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(
        name="Test Agent",
        linkedin_profile="LinkedIn info",
        summary="Summary info"
    )
    
    assert "Test Agent" in prompt
    assert "LinkedIn info" in prompt
    assert "Summary info" in prompt


def test_build_system_prompt_with_rag():
    """Test prompt with RAG enabled."""
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(
        name="Test",
        linkedin_profile="",
        summary="",
        enable_rag=True
    )
    
    assert "search_knowledge_base" in prompt


def test_build_system_prompt_without_rag():
    """Test prompt with RAG disabled."""
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(
        name="Test",
        linkedin_profile="",
        summary="",
        enable_rag=False
    )
    
    assert "search_knowledge_base" not in prompt
```

### Test & Commit

```bash
pytest tests/unit/test_file_loader.py tests/unit/test_prompt_builder.py -v

git add utils/ tests/unit/test_file_loader.py tests/unit/test_prompt_builder.py
git commit -m "feat(utils): implement file loader and prompt builder utilities

- Add FileLoader for PDF and text files
- Add PromptBuilder for system prompts
- Add comprehensive unit tests

Relates to Sprint 2, Task 2.4"
```

---

## Sprint 2 Completion

### Verification

```bash
# Run all Sprint 2 tests
uv run pytest tests/unit/test_database_service.py -v
uv run pytest tests/unit/test_llm_service.py -v
uv run pytest tests/unit/test_notification_service.py -v
uv run pytest tests/unit/test_file_loader.py -v
uv run pytest tests/unit/test_prompt_builder.py -v

# Run all tests with coverage
uv run pytest tests/unit/ --cov=services --cov=utils --cov-report=html
```

### Deliverables

- [x] Database Service with CRUD operations
- [x] LLM Service wrapper for DeepSeek
- [x] Notification Service for Pushover
- [x] File Loader utility
- [x] Prompt Builder utility
- [x] Unit tests for all services
- [x] Test coverage > 80%

### Next Steps

**Ready for Sprint 3!** 🎉

Open `plan/sprint_3_rag_tools.md` to continue with:
- RAG Service with Chroma
- Tool base class and registry
- Three concrete tools

---

**Sprint 2 Complete!** ✅  
**Next:** `plan/sprint_3_rag_tools.md`

