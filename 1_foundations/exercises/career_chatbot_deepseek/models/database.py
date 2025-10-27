"""SQLAlchemy ORM models for database persistence.

This module defines the database schema using SQLAlchemy ORM,
providing a clean interface for database operations.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()

class Conversation(Base):

    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_message = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    context_used = Column(Text)  # JSON string
    evaluation_score = Column(Float)
    was_regenerated = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, timestamp={self.timestamp})>"


class Contact(Base):
  
  __tablename__ = "contacts"

  id = Column(Integer, primary_key=True, autoincrement=True)
  email = Column(String(255), nullable=False, index=True)
  name = Column(String(255))
  notes = Column(Text)
  timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

  def __repr__(self):
    return f"<Contact(id={self.id}, email={self.email})>"

class UnknownQuestion(Base):
  
  __tablename__ = "unknown_questions"

  id = Column(Integer, primary_key=True, autoincrement=True)
  question = Column(Text, nullable=False)
  count = Column(Integer, default=1)
  timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

  def __repr__(self):
    return f"<UnknownQuestion(id={self.id}, question={self.question})>"


def init_db(db_url: str = "sqlite:///data/career_qa.db"):
  engine = create_engine(db_url, echo=False)
  Base.metadata.create_all(engine)
  return engine

def get_session_maker(engine):
  return sessionmaker(bind=engine)