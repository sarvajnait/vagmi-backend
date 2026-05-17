from typing import Optional, TYPE_CHECKING
from sqlalchemy import Column, Integer, ForeignKey, Text
from sqlmodel import SQLModel, Field
from app.models.base import BaseModel

if TYPE_CHECKING:
    pass


# --------------------
# Comp Student Textbook
# --------------------
class CompStudentTextbookBase(SQLModel):
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    title: str
    description: Optional[str] = None
    file_url: str
    sort_order: Optional[int] = Field(default=None)
    original_filename: Optional[str] = None
    audio_url: Optional[str] = None
    audio_status: Optional[str] = None  # "processing" | "completed" | "failed"


class CompStudentTextbook(CompStudentTextbookBase, BaseModel, table=True):
    __tablename__ = "comp_student_textbooks"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompStudentTextbookCreate(CompStudentTextbookBase):
    pass


class CompStudentTextbookRead(CompStudentTextbookBase):
    id: int


# --------------------
# Comp Student Note
# --------------------
class CompStudentNoteBase(SQLModel):
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    title: str
    description: Optional[str] = None
    file_url: Optional[str] = None
    sort_order: Optional[int] = Field(default=None)
    original_filename: Optional[str] = None
    # Extended Markdown content
    content: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    content_status: Optional[str] = None  # "processing" | "completed" | "failed"
    is_published: bool = Field(default=False)
    version: int = Field(default=1)
    word_count: Optional[int] = None
    read_time_min: Optional[int] = None
    source: Optional[str] = None  # "docx_upload" | "excel_upload"
    language: str = Field(default="en")
    # Audio
    audio_url: Optional[str] = None
    audio_status: Optional[str] = None  # "processing" | "completed" | "failed"
    audio_sync_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))


class CompStudentNote(CompStudentNoteBase, BaseModel, table=True):
    __tablename__ = "comp_student_notes"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompStudentNoteCreate(CompStudentNoteBase):
    pass


class CompStudentNoteRead(CompStudentNoteBase):
    id: int


# --------------------
# Comp Student Video
# --------------------
class CompStudentVideoBase(SQLModel):
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    title: str
    description: Optional[str] = None
    file_url: str
    sort_order: Optional[int] = Field(default=None)
    original_filename: Optional[str] = None


class CompStudentVideo(CompStudentVideoBase, BaseModel, table=True):
    __tablename__ = "comp_student_videos"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompStudentVideoCreate(CompStudentVideoBase):
    pass


class CompStudentVideoRead(CompStudentVideoBase):
    id: int


# --------------------
# Comp Previous Year Paper (level-scoped)
# --------------------
class CompPreviousYearPaperBase(SQLModel):
    level_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_levels.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    title: str
    year: Optional[int] = Field(default=None)
    num_questions: Optional[int] = Field(default=None)
    num_pages: Optional[int] = Field(default=None)
    file_url: str
    is_premium: bool = Field(default=False)
    enabled: bool = Field(default=True)
    sort_order: Optional[int] = Field(default=None)
    original_filename: Optional[str] = None


class CompPreviousYearPaper(CompPreviousYearPaperBase, BaseModel, table=True):
    __tablename__ = "comp_previous_year_papers"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompPreviousYearPaperCreate(CompPreviousYearPaperBase):
    pass


class CompPreviousYearPaperRead(CompPreviousYearPaperBase):
    id: int


class CompPreviousYearPaperUpdate(SQLModel):
    level_id: Optional[int] = None
    title: Optional[str] = None
    year: Optional[int] = None
    num_questions: Optional[int] = None
    num_pages: Optional[int] = None
    file_url: Optional[str] = None
    is_premium: Optional[bool] = None
    enabled: Optional[bool] = None
