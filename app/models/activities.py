from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import ARRAY, String, Column, DateTime, ForeignKey, UniqueConstraint, func
from sqlmodel import Field, Relationship, SQLModel

from app.models.base import BaseModel
from app.models.academic_hierarchy import Chapter
from app.models.user import User


class ChapterActivityBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    type: str = Field(max_length=20)
    question_text: str
    options: Optional[list[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    correct_option_index: Optional[int] = None
    answer_text: Optional[str] = None
    answer_image_url: Optional[str] = None
    is_published: bool = Field(default=True)
    sort_order: Optional[int] = Field(default=None)


class ChapterActivity(ChapterActivityBase, BaseModel, table=True):
    __tablename__ = "chapter_activities"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: Chapter = Relationship(back_populates="activities")


class ChapterActivityCreate(ChapterActivityBase):
    pass


class ChapterActivityRead(ChapterActivityBase):
    id: int


class ActivityPlaySession(BaseModel, table=True):
    __tablename__ = "activity_play_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(sa_column=Column(ForeignKey("user.id")))
    chapter_id: int = Field(foreign_key="chapters.id")
    status: str = Field(default="in_progress", max_length=20)
    total_questions: int = Field(default=0)
    correct_count: int = Field(default=0)
    score: int = Field(default=0)
    started_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
    )

    user: User = Relationship()
    chapter: Chapter = Relationship()


class ActivityAnswer(BaseModel, table=True):
    __tablename__ = "activity_answers"
    __table_args__ = (UniqueConstraint("session_id", "activity_id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="activity_play_sessions.id")
    activity_id: int = Field(foreign_key="chapter_activities.id")
    selected_option_index: Optional[int] = None
    submitted_answer_text: Optional[str] = None
    is_correct: Optional[bool] = None
    score: int = Field(default=0)
    answered_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    session: ActivityPlaySession = Relationship()
    activity: ChapterActivity = Relationship()
