# Sprint 1: Foundation & Data Models

**Sprint Goal:** Set up project structure, configuration, and data models  
**Deliverable:** Working configuration system + database models  
**Duration:** ~1-2 weeks

---

## Overview

This sprint establishes the foundation for the entire MVP:
- Project directory structure
- Configuration management with Pydantic Settings
- Data validation with Pydantic schemas
- Database models with SQLAlchemy
- Unit tests for all components

**Why this matters:** A solid foundation makes all future sprints easier and faster.

---

## Prerequisites

### Environment Setup
```bash
# Navigate to parent project
cd path/to/agents/

# Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux

# Verify Python version
python --version  # Should be 3.12

# Install/update dependencies
uv pip install -r requirements.txt
```

### Add Required Dependencies
Add these to parent `requirements.txt` if not present:

```bash
cd path/to/agents/

# Add to requirements.txt
cat >> requirements.txt << 'EOF'

# Sprint 1 dependencies
pydantic>=2.0.0
pydantic-settings>=2.0.0
sqlalchemy>=2.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
EOF

# Install
uv pip install -r requirements.txt
```

---

## Task 1.1: Project Structure Setup

**Estimated Time:** 30 minutes  
**Priority:** P0 (Must do first)

### Goal
Create clean directory structure following ADR 001.

### Steps

```bash
# Navigate to project directory
cd path/to/agents/1_foundations/exercises/career_chatbot_deepseek

# Create directory structure
mkdir -p config
mkdir -p models
mkdir -p services
mkdir -p tools
mkdir -p memory
mkdir -p core
mkdir -p utils
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p knowledge_base
mkdir -p data

# Create __init__.py files
touch config/__init__.py
touch models/__init__.py
touch services/__init__.py
touch tools/__init__.py
touch memory/__init__.py
touch core/__init__.py
touch utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# Create placeholder files
touch config/settings.py
touch models/schemas.py
touch models/database.py
touch tools/base.py
touch tools/registry.py
touch utils/file_loader.py
touch utils/prompt_builder.py

# Verify structure
tree -L 2 -I '__pycache__|*.pyc|.venv'
```

### Expected Output
```
career_chatbot_deepseek/
├── config/
│   ├── __init__.py
│   └── settings.py
├── models/
│   ├── __init__.py
│   ├── schemas.py
│   └── database.py
├── services/
│   └── __init__.py
├── tools/
│   ├── __init__.py
│   ├── base.py
│   └── registry.py
├── memory/
│   └── __init__.py
├── core/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── file_loader.py
│   └── prompt_builder.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   └── integration/
├── knowledge_base/
├── data/
└── app_deepseek.py (existing)
```

### Acceptance Criteria
- [ ] All directories created
- [ ] All `__init__.py` files present
- [ ] Placeholder files created
- [ ] Can import modules: `python -c "import config; import models; import services"`

### Commit
```bash
git add .
git commit -m "feat: create modular project structure

- Add config/, models/, services/, tools/, memory/, core/, utils/ directories
- Add __init__.py files for all modules
- Create placeholder files for core components

Relates to Sprint 1, Task 1.1"
```

---

## Task 1.2: Configuration Management

**Estimated Time:** 2-3 hours  
**Priority:** P0

### Goal
Implement type-safe configuration management using Pydantic Settings.

### Step 1: Implement Settings Class

Create `config/settings.py`:

```python
"""Configuration management using Pydantic Settings.

This module provides centralized, type-safe configuration management
for the Career Chatbot application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings with type safety and validation.
    
    All settings can be configured via:
    1. .env file
    2. Environment variables
    3. Default values
    
    Example:
        >>> from config.settings import settings
        >>> print(settings.agent_name)
        'Kin Lu'
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys (Required)
    deepseek_api_key: str
    
    # Optional API Keys
    google_api_key: Optional[str] = None
    pushover_user: Optional[str] = None
    pushover_token: Optional[str] = None
    
    # Agent Configuration
    agent_name: str = "Kin Lu"
    primary_model: str = "deepseek-chat"
    evaluator_model: str = "gemini-2.0-flash"
    
    # Feature Flags
    enable_rag: bool = True
    enable_evaluation: bool = True
    enable_memory: bool = False  # MVP: Working memory only
    
    # RAG Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 3
    
    # Memory Configuration
    working_memory_max_turns: int = 10
    working_memory_max_tokens: int = 4000
    
    # Database Paths
    db_path: str = "data/career_qa.db"
    chroma_path: str = "data/chroma_db"
    kb_path: str = "knowledge_base"
    
    # Performance
    max_retries: int = 1
    timeout_seconds: int = 30


# Global settings instance
settings = Settings()
```

### Step 2: Create .env.example

Create `.env.example`:

```bash
# API Keys (Required)
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# Optional API Keys
GOOGLE_API_KEY=your-google-api-key-here
PUSHOVER_USER=your-pushover-user-key
PUSHOVER_TOKEN=your-pushover-app-token

# Agent Configuration
AGENT_NAME="Your Name"
PRIMARY_MODEL=deepseek-chat
EVALUATOR_MODEL=gemini-2.0-flash

# Feature Flags
ENABLE_RAG=true
ENABLE_EVALUATION=true
ENABLE_MEMORY=false

# Paths
DB_PATH=data/career_qa.db
CHROMA_PATH=data/chroma_db
KB_PATH=knowledge_base
```

### Step 3: Create .gitignore

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Environment
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data
data/
*.db
*.sqlite
chroma_db/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temp
tmp/
temp/
```

### Step 4: Write Unit Tests

Create `tests/unit/test_settings.py`:

```python
"""Tests for configuration management."""

import pytest
from pydantic import ValidationError
from config.settings import Settings


def test_settings_with_required_fields():
    """Test settings initialization with required fields."""
    settings = Settings(deepseek_api_key="test-key")
    assert settings.deepseek_api_key == "test-key"
    assert settings.agent_name == "Kin Lu"  # default


def test_settings_validation_missing_required():
    """Test validation fails when required fields missing."""
    with pytest.raises(ValidationError):
        Settings()


def test_settings_feature_flags():
    """Test feature flags work correctly."""
    settings = Settings(
        deepseek_api_key="test",
        enable_rag=False,
        enable_evaluation=False
    )
    assert settings.enable_rag is False
    assert settings.enable_evaluation is False


def test_settings_defaults():
    """Test default values are set correctly."""
    settings = Settings(deepseek_api_key="test")
    assert settings.primary_model == "deepseek-chat"
    assert settings.chunk_size == 1000
    assert settings.top_k_results == 3


def test_settings_custom_values():
    """Test custom values override defaults."""
    settings = Settings(
        deepseek_api_key="test",
        agent_name="Test Agent",
        chunk_size=500
    )
    assert settings.agent_name == "Test Agent"
    assert settings.chunk_size == 500
```

### Step 5: Test

```bash
# Run tests
pytest tests/unit/test_settings.py -v

# Test with actual .env file
cp .env.example .env
# Edit .env with your keys
python -c "from config.settings import settings; print(f'Agent: {settings.agent_name}')"
```

### Acceptance Criteria
- [ ] Settings class implemented with all fields
- [ ] Type hints for all settings
- [ ] `.env.example` created
- [ ] `.gitignore` created
- [ ] Unit tests passing
- [ ] Can load settings from .env

### Commit
```bash
git add config/settings.py .env.example .gitignore tests/unit/test_settings.py
git commit -m "feat(config): implement Pydantic Settings for configuration

- Add Settings class with type safety and validation
- Support .env file loading
- Add feature flags for RAG, evaluation, memory
- Create .env.example template
- Add .gitignore
- Add comprehensive unit tests

Relates to Sprint 1, Task 1.2"
```

---

## Task 1.3: Pydantic Schemas

**Estimated Time:** 2 hours  
**Priority:** P0

### Goal
Define Pydantic models for data validation.

### Step 1: Implement Schemas

Create `models/schemas.py`:

```python
"""Pydantic models for data validation.

These models ensure data integrity throughout the application
by providing runtime validation and type safety.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class Evaluation(BaseModel):
    """Evaluator response schema.
    
    Attributes:
        acceptable: Whether the response meets quality standards
        feedback: Specific feedback for improvement
        score: Optional quality score from 1-10
    """
    acceptable: bool = Field(..., description="Whether response is acceptable")
    feedback: str = Field(..., description="Specific feedback for improvement")
    score: Optional[int] = Field(None, ge=1, le=10, description="Quality score 1-10")


class ContactRecord(BaseModel):
    """User contact information.
    
    Attributes:
        email: Valid email address
        name: Contact's name
        notes: Additional notes about the contact
        timestamp: When the contact was recorded
    """
    email: EmailStr
    name: str = "Not provided"
    notes: str = "Not provided"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QARecord(BaseModel):
    """Question-Answer pair with metadata.
    
    Attributes:
        question: User's question
        answer: Agent's answer
        context_used: RAG context chunks used (if any)
        evaluation_score: Quality score from evaluator
        timestamp: When the Q&A occurred
    """
    question: str
    answer: str
    context_used: Optional[List[str]] = None
    evaluation_score: Optional[int] = Field(None, ge=1, le=10)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolCall(BaseModel):
    """Tool execution result.
    
    Attributes:
        tool_name: Name of the tool that was called
        arguments: Arguments passed to the tool
        result: Result returned by the tool
        success: Whether the tool execution succeeded
        timestamp: When the tool was called
    """
    tool_name: str
    arguments: dict
    result: dict
    success: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Simplified models for MVP
class ConversationCreate(BaseModel):
    """Create conversation record.
    
    Used for creating new conversation entries in the database.
    """
    user_message: str
    agent_response: str
    evaluation_score: Optional[int] = None


class ContactCreate(BaseModel):
    """Create contact record.
    
    Used for creating new contact entries in the database.
    """
    email: EmailStr
    name: Optional[str] = None
    notes: Optional[str] = None
```

### Step 2: Write Unit Tests

Create `tests/unit/test_schemas.py`:

```python
"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError
from models.schemas import Evaluation, ContactRecord, QARecord, ConversationCreate


def test_evaluation_valid():
    """Test valid evaluation."""
    eval = Evaluation(
        acceptable=True,
        feedback="Great response",
        score=9
    )
    assert eval.acceptable is True
    assert eval.score == 9


def test_evaluation_score_validation():
    """Test score must be 1-10."""
    with pytest.raises(ValidationError):
        Evaluation(acceptable=True, feedback="test", score=11)
    
    with pytest.raises(ValidationError):
        Evaluation(acceptable=True, feedback="test", score=0)


def test_contact_record_email_validation():
    """Test email validation."""
    with pytest.raises(ValidationError):
        ContactRecord(email="invalid-email")
    
    # Valid email
    contact = ContactRecord(email="test@example.com")
    assert contact.email == "test@example.com"


def test_contact_record_defaults():
    """Test default values."""
    contact = ContactRecord(email="test@example.com")
    assert contact.name == "Not provided"
    assert contact.notes == "Not provided"
    assert contact.timestamp is not None


def test_qa_record_defaults():
    """Test QA record default values."""
    qa = QARecord(question="test?", answer="test answer")
    assert qa.context_used is None
    assert qa.evaluation_score is None
    assert qa.timestamp is not None


def test_qa_record_with_context():
    """Test QA record with context."""
    qa = QARecord(
        question="What is your experience?",
        answer="I have 10 years experience",
        context_used=["Context chunk 1", "Context chunk 2"],
        evaluation_score=8
    )
    assert len(qa.context_used) == 2
    assert qa.evaluation_score == 8


def test_conversation_create():
    """Test conversation creation model."""
    conv = ConversationCreate(
        user_message="Hello",
        agent_response="Hi there!"
    )
    assert conv.user_message == "Hello"
    assert conv.evaluation_score is None
```

### Step 3: Test

```bash
# Run tests
pytest tests/unit/test_schemas.py -v

# Test imports
python -c "from models.schemas import Evaluation, ContactRecord; print('Schemas OK')"
```

### Acceptance Criteria
- [ ] All Pydantic models defined
- [ ] Email validation working
- [ ] Score validation (1-10) working
- [ ] Default values working
- [ ] Unit tests passing

### Commit
```bash
git add models/schemas.py tests/unit/test_schemas.py
git commit -m "feat(models): add Pydantic schemas for data validation

- Add Evaluation, ContactRecord, QARecord, ToolCall models
- Add email validation for contacts
- Add score validation (1-10) for evaluations
- Add comprehensive unit tests

Relates to Sprint 1, Task 1.3"
```

---

## Task 1.4: SQLAlchemy Database Models

**Estimated Time:** 2-3 hours  
**Priority:** P0

### Goal
Define SQLAlchemy ORM models for database persistence.

### Step 1: Implement Database Models

Create `models/database.py`:

```python
"""SQLAlchemy ORM models for database persistence.

This module defines the database schema using SQLAlchemy ORM,
providing a clean interface for database operations.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class Conversation(Base):
    """Conversation history table.
    
    Stores all user-agent conversations with optional evaluation scores.
    
    Attributes:
        id: Primary key
        user_message: User's message
        agent_response: Agent's response
        context_used: JSON string of RAG context chunks
        evaluation_score: Quality score from evaluator (1-10)
        was_regenerated: Whether response was regenerated
        timestamp: When conversation occurred
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_message = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    context_used = Column(Text)  # JSON string
    evaluation_score = Column(Float)
    was_regenerated = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, timestamp={self.timestamp})>"


class Contact(Base):
    """Contact information table.
    
    Stores contact information from users who want to connect.
    
    Attributes:
        id: Primary key
        email: Contact's email address
        name: Contact's name
        notes: Additional notes
        timestamp: When contact was recorded
    """
    __tablename__ = "contacts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255))
    notes = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<Contact(id={self.id}, email={self.email})>"


class UnknownQuestion(Base):
    """Unknown questions tracking table.
    
    Tracks questions the agent couldn't answer, helping identify
    knowledge gaps and improvement opportunities.
    
    Attributes:
        id: Primary key
        question: The unknown question
        count: Number of times asked
        first_asked: When first asked
        last_asked: When last asked
    """
    __tablename__ = "unknown_questions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    count = Column(Integer, default=1)
    first_asked = Column(DateTime, default=datetime.utcnow)
    last_asked = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<UnknownQuestion(id={self.id}, count={self.count})>"


# Database initialization helpers
def init_db(db_url: str = "sqlite:///data/career_qa.db"):
    """Initialize database and create tables.
    
    Args:
        db_url: SQLAlchemy database URL
        
    Returns:
        SQLAlchemy engine instance
        
    Example:
        >>> engine = init_db()
        >>> # Tables are now created
    """
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session_maker(engine):
    """Get session maker for database operations.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        Session maker class
        
    Example:
        >>> engine = init_db()
        >>> SessionLocal = get_session_maker(engine)
        >>> session = SessionLocal()
    """
    return sessionmaker(bind=engine)
```

### Step 2: Write Unit Tests

Create `tests/unit/test_database.py`:

```python
"""Tests for database models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database import Base, Conversation, Contact, UnknownQuestion


@pytest.fixture
def db_session():
    """Create in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_conversation_create(db_session):
    """Test creating conversation record."""
    conv = Conversation(
        user_message="Hello",
        agent_response="Hi there!",
        evaluation_score=8.5
    )
    db_session.add(conv)
    db_session.commit()
    
    assert conv.id is not None
    assert conv.timestamp is not None
    assert conv.was_regenerated is False


def test_conversation_query(db_session):
    """Test querying conversations."""
    conv1 = Conversation(user_message="Q1", agent_response="A1")
    conv2 = Conversation(user_message="Q2", agent_response="A2")
    db_session.add_all([conv1, conv2])
    db_session.commit()
    
    count = db_session.query(Conversation).count()
    assert count == 2


def test_contact_create(db_session):
    """Test creating contact record."""
    contact = Contact(
        email="test@example.com",
        name="Test User",
        notes="Interested in collaboration"
    )
    db_session.add(contact)
    db_session.commit()
    
    assert contact.id is not None
    assert contact.email == "test@example.com"


def test_contact_query_by_email(db_session):
    """Test querying contact by email."""
    contact = Contact(email="test@example.com", name="Test")
    db_session.add(contact)
    db_session.commit()
    
    found = db_session.query(Contact).filter_by(email="test@example.com").first()
    assert found is not None
    assert found.name == "Test"


def test_unknown_question_create(db_session):
    """Test creating unknown question record."""
    q = UnknownQuestion(question="What is AI?")
    db_session.add(q)
    db_session.commit()
    
    assert q.id is not None
    assert q.count == 1


def test_unknown_question_increment(db_session):
    """Test incrementing unknown question count."""
    q = UnknownQuestion(question="What is AI?")
    db_session.add(q)
    db_session.commit()
    
    # Simulate increment
    q.count += 1
    db_session.commit()
    
    assert q.count == 2


def test_unknown_question_query_by_count(db_session):
    """Test querying unknown questions by count."""
    q1 = UnknownQuestion(question="Q1", count=5)
    q2 = UnknownQuestion(question="Q2", count=10)
    q3 = UnknownQuestion(question="Q3", count=3)
    db_session.add_all([q1, q2, q3])
    db_session.commit()
    
    # Get top question
    top = db_session.query(UnknownQuestion).order_by(
        UnknownQuestion.count.desc()
    ).first()
    
    assert top.question == "Q2"
    assert top.count == 10
```

### Step 3: Test Database Initialization

Create `tests/unit/test_database_init.py`:

```python
"""Tests for database initialization."""

import os
import tempfile
from models.database import init_db, get_session_maker, Conversation


def test_init_db_creates_tables():
    """Test that init_db creates all tables."""
    # Use temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        engine = init_db(f"sqlite:///{db_path}")
        
        # Check tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert "conversations" in tables
        assert "contacts" in tables
        assert "unknown_questions" in tables
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_get_session_maker():
    """Test session maker creation."""
    engine = init_db("sqlite:///:memory:")
    SessionLocal = get_session_maker(engine)
    
    session = SessionLocal()
    assert session is not None
    
    # Test can use session
    conv = Conversation(user_message="test", agent_response="test")
    session.add(conv)
    session.commit()
    
    assert conv.id is not None
    session.close()
```

### Step 4: Test

```bash
# Run tests
pytest tests/unit/test_database.py -v
pytest tests/unit/test_database_init.py -v

# Test database creation
python -c "
from models.database import init_db
import os
os.makedirs('data', exist_ok=True)
engine = init_db('sqlite:///data/test.db')
print('Database created successfully')
"

# Verify database file
ls -lh data/test.db

# Clean up
rm data/test.db
```

### Acceptance Criteria
- [ ] All SQLAlchemy models defined
- [ ] Database initialization working
- [ ] Can create and query records
- [ ] Indexes on timestamp columns
- [ ] Unit tests passing
- [ ] In-memory database tests working

### Commit
```bash
git add models/database.py tests/unit/test_database.py tests/unit/test_database_init.py
git commit -m "feat(models): add SQLAlchemy ORM models for persistence

- Add Conversation, Contact, UnknownQuestion tables
- Add database initialization helper
- Add indexes for performance
- Add comprehensive unit tests with in-memory DB

Relates to Sprint 1, Task 1.4"
```

---

## Sprint 1 Completion

### Verification Checklist

Run all tests:
```bash
# Run all Sprint 1 tests
pytest tests/unit/test_settings.py -v
pytest tests/unit/test_schemas.py -v
pytest tests/unit/test_database.py -v
pytest tests/unit/test_database_init.py -v

# Run all tests with coverage
pytest tests/unit/ --cov=config --cov=models --cov-report=html

# Check coverage (should be > 80%)
open htmlcov/index.html
```

### Sprint 1 Deliverables

- [x] Project structure created
- [x] Configuration management with Pydantic Settings
- [x] Pydantic schemas for validation
- [x] SQLAlchemy models for persistence
- [x] Unit tests for all components
- [x] Test coverage > 80%

### What We Built

1. **Configuration System** (`config/settings.py`)
   - Type-safe settings
   - .env file support
   - Feature flags
   - Validation

2. **Data Validation** (`models/schemas.py`)
   - Pydantic models
   - Email validation
   - Score validation
   - Default values

3. **Database Models** (`models/database.py`)
   - SQLAlchemy ORM
   - Three tables (Conversation, Contact, UnknownQuestion)
   - Indexes for performance
   - Helper functions

4. **Testing Infrastructure**
   - Unit tests for all components
   - In-memory database testing
   - Coverage reporting

### Next Steps

**Ready for Sprint 2!** 🎉

Open `plan/sprint_2_services.md` to continue with:
- Database Service (CRUD operations)
- LLM Service (DeepSeek wrapper)
- Notification Service (Pushover)
- Utils (File loader, Prompt builder)

### Retrospective Notes

**What went well:**
- 

**What could be improved:**
- 

**Lessons learned:**
- 

---

**Sprint 1 Complete!** ✅  
**Next:** `plan/sprint_2_services.md`

