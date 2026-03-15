from typing import Optional
from sqlalchemy import Column, Integer, ForeignKey
from sqlmodel import Field, SQLModel
from app.models.base import BaseModel


class CompChapterArtifactBase(SQLModel):
    comp_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    sub_chapter_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("comp_sub_chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    artifact_type: str = Field(max_length=50, index=True)
    # "pending" | "processing" | "completed" | "failed"
    status: str = Field(default="pending", max_length=20)
    content: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


class CompChapterArtifact(CompChapterArtifactBase, BaseModel, table=True):
    __tablename__ = "comp_chapter_artifacts"

    id: Optional[int] = Field(default=None, primary_key=True)
