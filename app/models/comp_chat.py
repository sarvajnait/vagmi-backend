import uuid
from typing import Optional
from sqlmodel import Field
import sqlalchemy as sa
from sqlalchemy import Column, Integer, ForeignKey, String, Text
from app.models.base import BaseModel


class CompChatSession(BaseModel, table=True):
    __tablename__ = "comp_chat_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    comp_subject_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_subjects.id", ondelete="CASCADE"), nullable=False)
    )
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="SET NULL"), nullable=True),
    )
    title: str = Field(
        default="",
        sa_column=Column(Text, nullable=False, server_default=""),
    )
    thread_id: str = Field(
        sa_column=Column(String(128), nullable=False, unique=True, index=True)
    )
