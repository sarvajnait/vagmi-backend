from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.academic_hierarchy import Chapter


# --------------------
# Student Textbook
# --------------------
class StudentTextbookBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    title: str
    description: Optional[str] = None
    file_url: str


class StudentTextbook(StudentTextbookBase, BaseModel, table=True):
    __tablename__ = "student_textbooks"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="student_textbooks")


class StudentTextbookCreate(StudentTextbookBase):
    pass


class StudentTextbookRead(StudentTextbookBase):
    id: int


# --------------------
# Student Notes (with file support like textbook)
# --------------------
class StudentNotesBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    title: str
    description: Optional[str] = None
    file_url: str


class StudentNotes(StudentNotesBase, BaseModel, table=True):
    __tablename__ = "student_notes"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="student_notes")


class StudentNotesCreate(StudentNotesBase):
    pass


class StudentNotesRead(StudentNotesBase):
    id: int
