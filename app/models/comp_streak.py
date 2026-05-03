from datetime import date, datetime
from typing import Optional
from sqlalchemy import Column, Date, DateTime, Integer, ForeignKey, UniqueConstraint
from sqlmodel import Field, SQLModel
from app.models.base import BaseModel


class UserStreak(BaseModel, table=True):
    __tablename__ = "user_streaks"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    )
    current_streak: int = Field(default=0)
    longest_streak: int = Field(default=0)
    last_activity_date: Optional[date] = Field(
        default=None,
        sa_column=Column(Date, nullable=True),
    )


class UserStreakDay(SQLModel, table=True):
    __tablename__ = "user_streak_days"
    __table_args__ = (UniqueConstraint("user_id", "activity_date"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    activity_date: date = Field(sa_column=Column(Date, nullable=False))


class UserMilestone(SQLModel, table=True):
    __tablename__ = "user_milestones"
    __table_args__ = (UniqueConstraint("user_id", "milestone_days"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    milestone_days: int = Field(nullable=False)
    achieved_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False)
    )
