from typing import Optional
from sqlalchemy import Column, Integer, ForeignKey
from sqlmodel import Field, SQLModel
from app.models.base import BaseModel


class UserNotification(BaseModel, table=True):
    __tablename__ = "user_notifications"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    title: str
    body: str
    notif_type: str = Field(max_length=50)
    icon_emoji: Optional[str] = Field(default=None, max_length=10)
    is_read: bool = Field(default=False)
