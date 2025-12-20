from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.academic_hierarchy import Chapter, Subject


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


class StudentVideoBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    title: str
    description: Optional[str] = None
    file_url: str


class StudentVideo(StudentVideoBase, BaseModel, table=True):
    __tablename__ = "student_videos"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="student_videos")


class StudentVideoCreate(StudentVideoBase):
    pass


class StudentVideoRead(StudentVideoBase):
    id: int


# --------------------
# Previous Year Question Papers (subject-level)
# --------------------
class PreviousYearQuestionPaperBase(SQLModel):
    subject_id: int = Field(foreign_key="subjects.id")
    title: str
    num_pages: int
    file_url: str
    is_premium: bool = Field(default=False)
    enabled: bool = Field(default=True)


class PreviousYearQuestionPaper(
    PreviousYearQuestionPaperBase, BaseModel, table=True
):
    __tablename__ = "previous_year_question_papers"

    id: Optional[int] = Field(default=None, primary_key=True)
    subject: "Subject" = Relationship(
        back_populates="previous_year_question_papers"
    )


class PreviousYearQuestionPaperCreate(PreviousYearQuestionPaperBase):
    pass


class PreviousYearQuestionPaperRead(PreviousYearQuestionPaperBase):
    id: int


class PreviousYearQuestionPaperUpdate(SQLModel):
    subject_id: Optional[int] = None
    title: Optional[str] = None
    num_pages: Optional[int] = None
    file_url: Optional[str] = None
    is_premium: Optional[bool] = None
    enabled: Optional[bool] = None
