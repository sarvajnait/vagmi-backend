from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, ForeignKey
from sqlmodel import Field

from app.models.base import BaseModel


class Notification(BaseModel, table=True):
    __tablename__ = "notifications"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    body: str
    # Academic filters — null means "send to all"
    class_level_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("class_levels.id", ondelete="SET NULL"), nullable=True),
    )
    board_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("boards.id", ondelete="SET NULL"), nullable=True),
    )
    medium_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("mediums.id", ondelete="SET NULL"), nullable=True),
    )
    # Competitive filters
    exam_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("exams.id", ondelete="SET NULL"), nullable=True),
    )
    comp_medium_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_exam_mediums.id", ondelete="SET NULL"), nullable=True),
    )
    level_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_levels.id", ondelete="SET NULL"), nullable=True),
    )
    sent_count: int = Field(default=0)
    sent_at: Optional[datetime] = Field(default=None)
