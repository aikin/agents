import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import Base, Contact, Conversation, UnknownQuestion


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_conversation_create(db_session):
    conv = Conversation(user_message="Hello", agent_response="Hi there!", evaluation_score=8.5)
    db_session.add(conv)
    db_session.commit()

    assert conv.id is not None
    assert conv.timestamp is not None
    assert conv.was_regenerated is False


def test_conversation_query(db_session):
    conv1 = Conversation(user_message="Q1", agent_response="A1")
    conv2 = Conversation(user_message="Q2", agent_response="A2")
    db_session.add_all([conv1, conv2])
    db_session.commit()

    count = db_session.query(Conversation).count()
    assert count == 2


def test_contact_create(db_session):
    contact = Contact(
        email="test@example.com", name="Test User", notes="Interested in collaboration"
    )
    db_session.add(contact)
    db_session.commit()

    assert contact.id is not None
    assert contact.email == "test@example.com"


def test_contact_query_by_email(db_session):
    contact = Contact(email="test@example.com", name="Test")
    db_session.add(contact)
    db_session.commit()

    found = db_session.query(Contact).filter_by(email="test@example.com").first()
    assert found is not None
    assert found.name == "Test"


def test_unknown_question_create(db_session):
    q = UnknownQuestion(question="What is AI?")
    db_session.add(q)
    db_session.commit()

    assert q.id is not None
    assert q.count == 1


def test_unknown_question_increment(db_session):
    q = UnknownQuestion(question="What is AI?")
    db_session.add(q)
    db_session.commit()

    # Simulate increment
    q.count += 1
    db_session.commit()

    assert q.count == 2


def test_unknown_question_query_by_count(db_session):
    q1 = UnknownQuestion(question="Q1", count=5)
    q2 = UnknownQuestion(question="Q2", count=10)
    q3 = UnknownQuestion(question="Q3", count=3)
    db_session.add_all([q1, q2, q3])
    db_session.commit()

    top = db_session.query(UnknownQuestion).order_by(UnknownQuestion.count.desc()).first()

    assert top.question == "Q2"
    assert top.count == 10
