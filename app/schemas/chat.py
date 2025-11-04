from pydantic import BaseModel
from typing import Optional


class HierarchyFilter(BaseModel):
    """Filter schema for educational hierarchy using database IDs.
    
    Attributes:
        class_level_id: ID of the class level (e.g., Class 10, Class 12)
        board_id: ID of the educational board (e.g., CBSE, ICSE, State Board)
        medium_id: ID of the medium of instruction (e.g., English, Hindi)
        subject_id: ID of the subject (e.g., Mathematics, Physics, Chemistry)
    """

    class_level_id: Optional[int] = None
    board_id: Optional[int] = None
    medium_id: Optional[int] = None
    subject_id: Optional[int] = None


class DeleteRequest(BaseModel):
    """Request schema for deleting educational content using hierarchical IDs.
    
    Attributes:
        filename: Name of the file to be deleted
        class_level_id: ID of the class level containing the content
        board_id: ID of the educational board
        medium_id: ID of the medium of instruction
        subject_id: ID of the subject
        chapter_id: ID of the specific chapter
    """

    filename: str
    class_level_id: int
    board_id: int
    medium_id: int
    subject_id: int
    chapter_id: int


class ChatRequest(BaseModel):
    """Request schema for AI chat interactions with educational context.
    
    This schema is used for chat requests to the AI tutor, providing the necessary
    hierarchical context to ensure responses are appropriate for the student's
    educational level and curriculum. The chapter_id is optional to allow for
    general subject-level queries.
    
    Attributes:
        message: The student's question or message
        class_level_id: ID of the student's class level
        board_id: ID of the educational board (CBSE, ICSE, etc.)
        medium_id: ID of the medium of instruction
        subject_id: ID of the subject being discussed
        chapter_id: Optional ID of the specific chapter (for chapter-specific queries)
    """

    message: str
    class_level_id: int
    board_id: int
    medium_id: int
    subject_id: int
    chapter_id: Optional[int] = None
