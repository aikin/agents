"""Database service for SQL operations.

Provides a clean interface for all database operations,
following the Repository pattern.
"""

import json
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from models.database import Base, Contact, Conversation, UnknownQuestion


class DatabaseService:
  def __init__(self, db_url: str | None = None):
    self.db_url = db_url or f"sqlite:///{settings.db_path}"
    self.engine = create_engine(self.db_url, echo=False)
    Base.metadata.create_all(self.engine)
    self.SessionLocal = sessionmaker(bind=self.engine)

  @contextmanager
  def get_session(self) -> Session:
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
    evaluation_score: float | None = None,
    context_used: list[str] | None = None) -> int:

    conversation = Conversation(
      user_message=user_message,
      agent_response=agent_response,
      evaluation_score=evaluation_score,
      context_used=json.dumps(context_used) if context_used else None
    )

    with self.get_session() as session:
      session.add(conversation)
      session.flush()
      return conversation.id


  def save_contact(
    self,
    email: str,
    name: str | None = None,
    notes: str | None = None
  ) -> int:
    contact = Contact(
      email=email,
      name=name or "Not provided",
      notes=notes or "Not provided"
    )

    with self.get_session() as session:
      session.add(contact)
      session.flush()
      return contact.id

  def record_unknown_question(self, question: str) -> None:
    with self.get_session() as session:
      existing = session.query(UnknownQuestion).filter_by(question=question).first()
      if existing:
        existing.count += 1
        existing.last_asked = func.now()
      else:
        new_q = UnknownQuestion(question=question)
        session.add(new_q)

  def get_analytics(self) -> dict[str, Any]:
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

  def get_recent_conversations(self, limit: int = 10) -> list[dict]:
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
