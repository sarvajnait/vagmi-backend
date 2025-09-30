from typing import Optional
from sqlmodel import SQLModel

class HierarchyFilter(SQLModel):
    class_level: Optional[str] = None
    board: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None
