from typing import Optional
from sqlmodel import SQLModel


class ChatRequest(SQLModel):
    message: str
    class_level: Optional[str] = None
    board: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
