from typing import Optional
from pydantic import BaseModel


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
