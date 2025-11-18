"""Pydantic models for data validation.

These models ensure data integrity throughout the application
by providing runtime validation and type safety.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, EmailStr, Field, StrictBool


def current_utc_time() -> datetime:
    return datetime.now(UTC)


class Evaluation(BaseModel):
    acceptable: bool = Field(..., description="Whether response is acceptable")
    feedback: str = Field(..., description="Specific feedback for improvement")
    score: int | None = Field(
        None,
        ge=1,
        le=10,
        description="Quality score from 1-10",
    )


class ContactRecord(BaseModel):
    email: EmailStr
    name: str = "Not provided"
    notes: str = "Not provided"
    timestamp: datetime = Field(default_factory=current_utc_time)


class QARecord(BaseModel):
    question: str
    answer: str
    context_used: list[str] | None = None
    evaluation_score: int | None = Field(None, ge=1, le=10)
    timestamp: datetime = Field(default_factory=current_utc_time)


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    result: dict[str, object]
    success: StrictBool
    timestamp: datetime = Field(default_factory=current_utc_time)


class ConversationCreate(BaseModel):
    user_message: str
    agent_response: str
    evaluation_score: int | None = Field(None, ge=1, le=10)


class ContactCreate(BaseModel):
    email: EmailStr
    name: str | None = None
    notes: str | None = None


