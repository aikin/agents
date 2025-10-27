
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