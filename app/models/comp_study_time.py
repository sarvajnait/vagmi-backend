from datetime import date
from typing import Optional
from sqlalchemy import Column, Date, Integer, ForeignKey, UniqueConstraint
from sqlmodel import Field, SQLModel
from app.models.base import BaseModel


class StudyTimeLog(BaseModel, table=True):
    __tablename__ = "study_time_logs"
    __table_args__ = (UniqueConstraint("user_id", "logged_date"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    logged_date: date = Field(sa_column=Column(Date, nullable=False))
    duration_seconds: int = Field(default=0)
