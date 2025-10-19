from typing import Optional
from pydantic import BaseModel


class HierarchyFilter(BaseModel):
    """Filter hierarchy by IDs instead of names"""

    class_level_id: Optional[int] = None
    board_id: Optional[int] = None
    medium_id: Optional[int] = None
    subject_id: Optional[int] = None
