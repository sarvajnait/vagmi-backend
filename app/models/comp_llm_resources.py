from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import ARRAY, String, Integer, ForeignKey
from app.models.base import BaseModel

if TYPE_CHECKING:
    pass


# --------------------
# Comp LLM Textbook
# --------------------
class CompLLMTextbookBase(SQLModel):
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


class CompLLMTextbook(CompLLMTextbookBase, BaseModel, table=True):
    __tablename__ = "comp_llm_textbooks"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompLLMTextbookCreate(CompLLMTextbookBase):
    pass


class CompLLMTextbookRead(CompLLMTextbookBase):
    id: int


# --------------------
# Comp LLM Note
# --------------------
class CompLLMNoteBase(SQLModel):
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


class CompLLMNote(CompLLMNoteBase, BaseModel, table=True):
    __tablename__ = "comp_llm_notes"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompLLMNoteCreate(CompLLMNoteBase):
    pass


class CompLLMNoteRead(CompLLMNoteBase):
    id: int


# --------------------
# Comp Additional Note
# --------------------
class CompAdditionalNoteBase(SQLModel):
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    note: str
    sort_order: Optional[int] = Field(default=None)


class CompAdditionalNote(CompAdditionalNoteBase, BaseModel, table=True):
    __tablename__ = "comp_additional_notes"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompAdditionalNoteCreate(CompAdditionalNoteBase):
    pass


class CompAdditionalNoteRead(CompAdditionalNoteBase):
    id: int


# --------------------
# Comp QA Pattern
# --------------------
class CompQAPatternBase(SQLModel):
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


class CompQAPattern(CompQAPatternBase, BaseModel, table=True):
    __tablename__ = "comp_qa_patterns"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompQAPatternCreate(CompQAPatternBase):
    pass


class CompQAPatternRead(CompQAPatternBase):
    id: int


# --------------------
# Comp LLM Image
# --------------------
class CompLLMImageBase(SQLModel):
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
    tags: Optional[list[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    sort_order: Optional[int] = Field(default=None)
    original_filename: Optional[str] = None


class CompLLMImage(CompLLMImageBase, BaseModel, table=True):
    __tablename__ = "comp_llm_images"

    id: Optional[int] = Field(default=None, primary_key=True)


class CompLLMImageCreate(CompLLMImageBase):
    pass


class CompLLMImageRead(CompLLMImageBase):
    id: int
