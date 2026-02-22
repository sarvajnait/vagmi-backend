from datetime import datetime
from typing import Optional

from sqlmodel import Field

from app.models.base import BaseModel


class Notification(BaseModel, table=True):
    __tablename__ = "notifications"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    body: str
    # Optional filters — null means "send to all"
    class_level_id: Optional[int] = Field(default=None, foreign_key="class_levels.id")
    board_id: Optional[int] = Field(default=None, foreign_key="boards.id")
    medium_id: Optional[int] = Field(default=None, foreign_key="mediums.id")
    sent_count: int = Field(default=0)
    sent_at: Optional[datetime] = Field(default=None)
