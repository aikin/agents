from datetime import datetime

import pytest
from pydantic import ValidationError

from models.schemas import (
    ContactCreate,
    ContactRecord,
    ConversationCreate,
    Evaluation,
    QARecord,
    ToolCall,
)


def test_evaluation_valid():

    eval_result = Evaluation(acceptable=True, feedback="Great response", score=9)
    assert eval_result.acceptable is True
    assert eval_result.score == 9


def test_evaluation_score_validation():

    with pytest.raises(ValidationError):
        Evaluation(acceptable=True, feedback="test", score=11)

    with pytest.raises(ValidationError):
        Evaluation(acceptable=True, feedback="test", score=0)


def test_contact_record_email_validation():

    with pytest.raises(ValidationError):
        ContactRecord(email="invalid-email")

    # Valid email
    contact = ContactRecord(email="test@example.com")
    assert contact.email == "test@example.com"


def test_contact_record_defaults():

    contact = ContactRecord(email="test@example.com")
    assert contact.name == "Not provided"
    assert contact.notes == "Not provided"
    assert isinstance(contact.timestamp, datetime)


def test_qa_record_defaults():

    qa_record = QARecord(question="test?", answer="test answer")
    assert qa_record.context_used is None
    assert qa_record.evaluation_score is None
    assert isinstance(qa_record.timestamp, datetime)


def test_qa_record_with_context():

    qa_record = QARecord(
        question="What is your experience?",
        answer="I have 10 years experience",
        context_used=["Context chunk 1", "Context chunk 2"],
        evaluation_score=8,
    )
    assert len(qa_record.context_used) == 2
    assert qa_record.evaluation_score == 8


def test_conversation_create():

    conv = ConversationCreate(user_message="Hello", agent_response="Hi there!")
    assert conv.user_message == "Hello"
    assert conv.evaluation_score is None


def test_conversation_create_score_validation():

    with pytest.raises(ValidationError):
        ConversationCreate(
            user_message="Hello",
            agent_response="Hi",
            evaluation_score=0,
        )


def test_contact_create():

    contact = ContactCreate(email="test@example.com", name="Test", notes="Hi")
    assert contact.email == "test@example.com"
    assert contact.name == "Test"
    assert contact.notes == "Hi"


def test_tool_call_defaults():

    result = ToolCall(
        tool_name="search",
        arguments={"query": "test"},
        result={"items": []},
        success=True,
    )
    assert result.tool_name == "search"
    assert result.arguments["query"] == "test"
    assert isinstance(result.timestamp, datetime)


def test_tool_call_validation():

    with pytest.raises(ValidationError):
        ToolCall(tool_name="search", arguments={}, result={}, success="yes")


