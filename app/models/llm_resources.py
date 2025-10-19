from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship
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
