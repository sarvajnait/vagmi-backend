from datetime import datetime
from typing import Optional, List
from sqlalchemy import ARRAY, DateTime, String, Column, Integer, ForeignKey, UniqueConstraint, func
from sqlmodel import Field, Relationship, SQLModel
from app.models.base import BaseModel


class CompTopicBase(SQLModel):
    title: str = Field(max_length=255)
    summary: Optional[str] = Field(default=None)
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sort_order: Optional[int] = Field(default=None)


class CompTopic(CompTopicBase, BaseModel, table=True):
    __tablename__ = "comp_topics"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompTopicCreate(CompTopicBase):
    pass


class CompTopicRead(CompTopicBase):
    id: int


class CompActivityGroupBase(SQLModel):
    name: str = Field(max_length=255)
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    timer_seconds: Optional[int] = Field(default=None)
    sort_order: Optional[int] = Field(default=None)


class CompActivityGroup(CompActivityGroupBase, BaseModel, table=True):
    __tablename__ = "comp_activity_groups"

    id: Optional[int] = Field(default=None, primary_key=True)
    activities: List["CompChapterActivity"] = Relationship(back_populates="activity_group", cascade_delete=True)


class CompActivityGroupCreate(CompActivityGroupBase):
    pass


class CompActivityGroupRead(CompActivityGroupBase):
    id: int


class CompChapterActivityBase(SQLModel):
    activity_group_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_activity_groups.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    type: str = Field(max_length=20)
    question_text: str
    options: Optional[list[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    correct_option_index: Optional[int] = None
    answer_text: Optional[str] = None
    answer_description: Optional[str] = None
    answer_image_url: Optional[str] = None
    is_published: bool = Field(default=True)
    sort_order: Optional[int] = Field(default=None)


class CompChapterActivity(CompChapterActivityBase, BaseModel, table=True):
    __tablename__ = "comp_chapter_activities"

    id: Optional[int] = Field(default=None, primary_key=True)
    activity_group: CompActivityGroup = Relationship(back_populates="activities")


class CompChapterActivityCreate(CompChapterActivityBase):
    pass


class CompChapterActivityRead(CompChapterActivityBase):
    id: int


class CompActivityPlaySession(BaseModel, table=True):
    __tablename__ = "comp_activity_play_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    )
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True),
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True),
    )
    status: str = Field(default="in_progress", max_length=20)
    total_questions: int = Field(default=0)
    correct_count: int = Field(default=0)
    score: int = Field(default=0)
    started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
    )


class CompActivityAnswer(BaseModel, table=True):
    __tablename__ = "comp_activity_answers"
    __table_args__ = (UniqueConstraint("session_id", "activity_id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_activity_play_sessions.id", ondelete="CASCADE"), nullable=False)
    )
    activity_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_chapter_activities.id", ondelete="CASCADE"), nullable=False)
    )
    selected_option_index: Optional[int] = None
    is_correct: Optional[bool] = None
    score: int = Field(default=0)
    answered_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
