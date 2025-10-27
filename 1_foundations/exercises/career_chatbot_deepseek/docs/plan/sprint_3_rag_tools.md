# Sprint 3: RAG & Tools

**Sprint Goal:** Implement RAG service and tool system  
**Deliverable:** Working knowledge base retrieval + tool calling  
**Duration:** ~1-2 weeks

---

## Overview

This sprint implements:
- **RAG Service:** Vector database (Chroma) for knowledge retrieval
- **Tool System:** Base class, registry, and concrete tools
- **Knowledge Base:** Ingestion and search

**Why this matters:** RAG provides context-aware responses, tools enable actions.

---

## Prerequisites

```bash
# Add dependencies to parent requirements.txt
cd path/to/agents/
cat >> requirements.txt << 'EOF'

# Sprint 3 dependencies
chromadb>=0.4.22
sentence-transformers>=2.2.0
EOF

uv pip install -r requirements.txt
```

---

## Task 3.1: RAG Service

**Estimated Time:** 3-4 hours  
**Priority:** P0

### Implementation

Create `services/rag_service.py`:

```python
"""RAG service using Chroma vector database."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Optional
from config.settings import settings


class RAGService:
    """RAG service for knowledge base retrieval.
    
    Example:
        >>> rag = RAGService()
        >>> rag.ingest_documents(["doc1.txt", "doc2.txt"])
        >>> results = rag.search("What is your experience?")
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize RAG service.
        
        Args:
            persist_directory: Optional Chroma persistence directory
        """
        self.persist_directory = persist_directory or settings.chroma_path
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name="career_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    
    def ingest_text(self, text: str, document_id: str, metadata: Optional[dict] = None):
        """Ingest text into vector database.
        
        Args:
            text: Text to ingest
            document_id: Unique document ID
            metadata: Optional metadata
        """
        # Simple chunking by paragraphs
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        
        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                ids=[f"{document_id}_chunk_{i}"],
                metadatas=[metadata or {}]
            )
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results and results['documents']:
            return results['documents'][0]
        return []
    
    def clear(self):
        """Clear all documents from collection."""
        self.client.delete_collection("career_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="career_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
```

### Tests

Create `tests/unit/test_rag_service.py`:

```python
"""Tests for RAG service."""

import pytest
import tempfile
import shutil
from services.rag_service import RAGService


@pytest.fixture
def rag_service():
    """Create RAG service with temporary directory."""
    temp_dir = tempfile.mkdtemp()
    service = RAGService(persist_directory=temp_dir)
    yield service
    shutil.rmtree(temp_dir)


def test_ingest_and_search(rag_service):
    """Test ingesting and searching documents."""
    rag_service.ingest_text(
        "I have 10 years of experience in software engineering.",
        "doc1"
    )
    
    results = rag_service.search("experience")
    assert len(results) > 0
    assert "experience" in results[0].lower()


def test_search_no_results(rag_service):
    """Test search with no results."""
    results = rag_service.search("nonexistent query")
    assert results == []


def test_clear(rag_service):
    """Test clearing collection."""
    rag_service.ingest_text("Test document", "doc1")
    rag_service.clear()
    
    results = rag_service.search("Test")
    assert results == []
```

### Test & Commit

```bash
pytest tests/unit/test_rag_service.py -v

git add services/rag_service.py tests/unit/test_rag_service.py
git commit -m "feat(services): implement RAG service with Chroma

- Add RAGService for vector search
- Implement document ingestion and search
- Add unit tests

Relates to Sprint 3, Task 3.1"
```

---

## Task 3.2: Tool Base & Registry

**Estimated Time:** 2 hours  
**Priority:** P0

### Implementation

Create `tools/base.py`:

```python
"""Base tool class for function calling."""

from abc import ABC, abstractmethod
from typing import Dict, Any


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
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters JSON schema."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with given parameters.
        
        Returns:
            Result dictionary with 'success' and 'message' keys
        """
        pass
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
```

Create `tools/registry.py`:

```python
"""Tool registry for managing tools."""

from typing import Dict, List, Optional
from tools.base import BaseTool


class ToolRegistry:
    """Registry for managing tools.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(ContactTool(db_service))
        >>> tool = registry.get("record_user_details")
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None
        """
        return self._tools.get(name)
    
    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def to_openai_format(self) -> List[Dict]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]
```

### Tests

Create `tests/unit/test_tools.py`:

```python
"""Tests for tool system."""

import pytest
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
```

### Test & Commit

```bash
pytest tests/unit/test_tools.py -v

git add tools/base.py tools/registry.py tests/unit/test_tools.py
git commit -m "feat(tools): implement tool base class and registry

- Add BaseTool abstract class
- Add ToolRegistry for managing tools
- Add unit tests

Relates to Sprint 3, Task 3.2"
```

---

## Task 3.3: Concrete Tools

**Estimated Time:** 2-3 hours  
**Priority:** P0

### Implementation

Create `tools/contact_tool.py`, `tools/question_tool.py`, `tools/search_tool.py`:

```python
# tools/contact_tool.py
"""Contact recording tool."""

from tools.base import BaseTool
from services.database_service import DatabaseService
from services.notification_service import NotificationService


class ContactTool(BaseTool):
    """Tool for recording user contact information."""
    
    def __init__(self, db_service: DatabaseService, notif_service: NotificationService):
        self.db_service = db_service
        self.notif_service = notif_service
    
    @property
    def name(self) -> str:
        return "record_user_details"
    
    @property
    def description(self) -> str:
        return "Record user contact details when they want to get in touch"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "User's email"},
                "name": {"type": "string", "description": "User's name"},
                "notes": {"type": "string", "description": "Additional notes"}
            },
            "required": ["email"]
        }
    
    def execute(self, email: str, name: str = "", notes: str = "") -> dict:
        try:
            contact_id = self.db_service.save_contact(email, name, notes)
            self.notif_service.send(f"New contact: {email} ({name})")
            return {
                "success": True,
                "message": f"Contact saved (ID: {contact_id})"
            }
        except Exception as e:
            return {"success": False, "message": str(e)}


# tools/question_tool.py
"""Unknown question recording tool."""

from tools.base import BaseTool
from services.database_service import DatabaseService


class QuestionTool(BaseTool):
    """Tool for recording unknown questions."""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
    
    @property
    def name(self) -> str:
        return "record_unknown_question"
    
    @property
    def description(self) -> str:
        return "Record questions that cannot be answered"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The unknown question"}
            },
            "required": ["question"]
        }
    
    def execute(self, question: str) -> dict:
        try:
            self.db_service.record_unknown_question(question)
            return {"success": True, "message": "Question recorded"}
        except Exception as e:
            return {"success": False, "message": str(e)}


# tools/search_tool.py
"""Knowledge base search tool."""

from tools.base import BaseTool
from services.rag_service import RAGService


class SearchTool(BaseTool):
    """Tool for searching knowledge base."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    @property
    def name(self) -> str:
        return "search_knowledge_base"
    
    @property
    def description(self) -> str:
        return "Search the knowledge base for relevant information"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    
    def execute(self, query: str) -> dict:
        try:
            results = self.rag_service.search(query)
            return {
                "success": True,
                "message": f"Found {len(results)} results",
                "results": results
            }
        except Exception as e:
            return {"success": False, "message": str(e), "results": []}
```

### Tests

Create `tests/unit/test_concrete_tools.py`:

```python
"""Tests for concrete tools."""

import pytest
from tools.contact_tool import ContactTool
from tools.question_tool import QuestionTool
from tools.search_tool import SearchTool
from services.database_service import DatabaseService
from services.notification_service import NotificationService
from services.rag_service import RAGService
import tempfile
import shutil


@pytest.fixture
def db_service():
    return DatabaseService(db_url="sqlite:///:memory:")


@pytest.fixture
def notif_service():
    return NotificationService()


@pytest.fixture
def rag_service():
    temp_dir = tempfile.mkdtemp()
    service = RAGService(persist_directory=temp_dir)
    yield service
    shutil.rmtree(temp_dir)


def test_contact_tool(db_service, notif_service):
    """Test contact tool execution."""
    tool = ContactTool(db_service, notif_service)
    
    result = tool.execute(email="test@example.com", name="Test User")
    assert result["success"] is True


def test_question_tool(db_service):
    """Test question tool execution."""
    tool = QuestionTool(db_service)
    
    result = tool.execute(question="What is AI?")
    assert result["success"] is True


def test_search_tool(rag_service):
    """Test search tool execution."""
    rag_service.ingest_text("Test document about AI", "doc1")
    tool = SearchTool(rag_service)
    
    result = tool.execute(query="AI")
    assert result["success"] is True
```

### Test & Commit

```bash
pytest tests/unit/test_concrete_tools.py -v

git add tools/contact_tool.py tools/question_tool.py tools/search_tool.py tests/unit/test_concrete_tools.py
git commit -m "feat(tools): implement concrete tools for contact, question, search

- Add ContactTool for recording user details
- Add QuestionTool for unknown questions
- Add SearchTool for RAG search
- Add unit tests

Relates to Sprint 3, Task 3.3"
```

---

## Sprint 3 Completion

### Verification

```bash
pytest tests/unit/test_rag_service.py tests/unit/test_tools.py tests/unit/test_concrete_tools.py -v

pytest tests/unit/ --cov=services --cov=tools --cov-report=html
```

### Deliverables

- [x] RAG Service with Chroma
- [x] Tool base class and registry
- [x] Three concrete tools (Contact, Question, Search)
- [x] Unit tests for all components
- [x] Test coverage > 80%

### Next Steps

**Ready for Sprint 4!** 🎉

Open `plan/sprint_4_agents.md` to continue with:
- Career Agent
- Evaluator Agent
- Working Memory

---

**Sprint 3 Complete!** ✅  
**Next:** `plan/sprint_4_agents.md`

