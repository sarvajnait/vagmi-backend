from datetime import date
from typing import Optional

from sqlalchemy import Column, Integer, ForeignKey
from sqlmodel import Field

from app.models.base import BaseModel


class UserCompProfile(BaseModel, table=True):
    __tablename__ = "user_comp_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    )
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
    exam_date: Optional[date] = Field(default=None)
    daily_commitment_hours: Optional[int] = Field(default=None)
