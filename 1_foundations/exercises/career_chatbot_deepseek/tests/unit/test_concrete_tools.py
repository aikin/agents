import shutil
import tempfile

import pytest

from services.database_service import DatabaseService
from services.notification_service import NotificationService
from services.rag_service import RAGService
from tools.contact_tool import ContactTool
from tools.question_tool import QuestionTool
from tools.search_tool import SearchTool


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
