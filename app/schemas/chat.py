from pydantic import BaseModel
from typing import Optional


class HierarchyFilter(BaseModel):
    """Filter hierarchy by IDs instead of names"""

    class_level_id: Optional[int] = None
    board_id: Optional[int] = None
    medium_id: Optional[int] = None
    subject_id: Optional[int] = None


class DeleteRequest(BaseModel):
    """Delete request using IDs"""

    filename: str
    class_level_id: int
    board_id: int
    medium_id: int
    subject_id: int
    chapter_id: int


class ChatRequest(BaseModel):
    """Chat request using IDs"""

    message: str
    class_level_id: int
    board_id: int
    medium_id: int
    subject_id: int
    chapter_id: Optional[int] = None
