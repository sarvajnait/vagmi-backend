from datetime import datetime
from typing import Optional
from sqlalchemy import Column, DateTime, Integer, ForeignKey, UniqueConstraint, func
from sqlmodel import Field, SQLModel
from app.models.base import BaseModel


class WrongAnswerEntry(BaseModel, table=True):
    __tablename__ = "wrong_answer_entries"
    __table_args__ = (UniqueConstraint("user_id", "activity_id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    activity_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_chapter_activities.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True),
    )
    activity_group_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_activity_groups.id", ondelete="CASCADE"), nullable=True, index=True),
    )
    times_attempted: int = Field(default=1)
    last_wrong_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    )
    is_mastered: bool = Field(default=False)
