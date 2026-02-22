from typing import Optional

from sqlalchemy import Column, Integer, ForeignKey
from sqlmodel import Field, SQLModel

from app.models.base import BaseModel


class ChapterArtifactBase(SQLModel):
    chapter_id: int = Field(
        sa_column=Column(Integer, ForeignKey("chapters.id", ondelete="CASCADE"), index=True, nullable=False)
    )
    artifact_type: str = Field(max_length=50, index=True)
    # "pending" | "processing" | "completed" | "failed"
    status: str = Field(default="pending", max_length=20)
    content: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


class ChapterArtifact(ChapterArtifactBase, BaseModel, table=True):
    __tablename__ = "chapter_artifacts"

    id: Optional[int] = Field(default=None, primary_key=True)
