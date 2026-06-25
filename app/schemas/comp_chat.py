from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    comp_subject_id: int
    comp_chapter_id: Optional[int] = None


class SessionResponse(BaseModel):
    id: int
    comp_subject_id: int
    comp_chapter_id: Optional[int]
    title: str
    thread_id: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class MessageOut(BaseModel):
    role: str       # "human" | "ai"
    content: str


class SessionHistoryResponse(BaseModel):
    session: SessionResponse
    messages: list[MessageOut]


class CompChatRequest(BaseModel):
    session_id: int
    message: str
