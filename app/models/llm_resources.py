from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import ARRAY, String
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.academic_hierarchy import Chapter


# --------------------
# LLM Textbook
# --------------------
class LLMTextbookBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    title: str
    description: Optional[str] = None
    file_url: str


class LLMTextbook(LLMTextbookBase, BaseModel, table=True):
    __tablename__ = "llm_textbooks"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="llm_textbooks")


class LLMTextbookCreate(LLMTextbookBase):
    pass


class LLMTextbookRead(LLMTextbookBase):
    id: int


# --------------------
# Additional Notes
# --------------------
class AdditionalNotesBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    note: str


class AdditionalNotes(AdditionalNotesBase, BaseModel, table=True):
    __tablename__ = "additional_notes"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="additional_notes")


class AdditionalNotesCreate(AdditionalNotesBase):
    pass


class AdditionalNotesRead(AdditionalNotesBase):
    id: int


# --------------------
# LLM Image
# --------------------
class LLMImageBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    title: str
    description: Optional[str] = None
    file_url: str
    tags: Optional[list[str]] = Field(default=None, sa_column=Column(ARRAY(String)))


class LLMImage(LLMImageBase, BaseModel, table=True):
    __tablename__ = "llm_images"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="llm_images")


class LLMImageCreate(LLMImageBase):
    pass


class LLMImageRead(LLMImageBase):
    id: int


# --------------------
# LLM Notes (PDF-based, for RAG)
# --------------------
class LLMNoteBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    title: str
    description: Optional[str] = None
    file_url: str


class LLMNote(LLMNoteBase, BaseModel, table=True):
    __tablename__ = "llm_notes"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="llm_notes")


class LLMNoteCreate(LLMNoteBase):
    pass


class LLMNoteRead(LLMNoteBase):
    id: int


# --------------------
# Q&A Patterns (PDF-based, for RAG)
# --------------------
class QAPatternBase(SQLModel):
    chapter_id: int = Field(foreign_key="chapters.id")
    title: str
    description: Optional[str] = None
    file_url: str


class QAPattern(QAPatternBase, BaseModel, table=True):
    __tablename__ = "qa_patterns"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: "Chapter" = Relationship(back_populates="qa_patterns")


class QAPatternCreate(QAPatternBase):
    pass


class QAPatternRead(QAPatternBase):
    id: int
