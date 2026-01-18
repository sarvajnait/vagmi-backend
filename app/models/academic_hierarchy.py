from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from app.models.base import BaseModel
from app.models.llm_resources import (
    LLMTextbook,
    AdditionalNotes,
    LLMImage,
    LLMNote,
    QAPattern,
)
from app.models.student_content import (
    StudentTextbook,
    StudentNotes,
    StudentVideo,
    PreviousYearQuestionPaper,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.activities import ChapterActivity, ActivityGroup


# --------------------
# Class Level
# --------------------
class ClassLevelBase(SQLModel):
    name: str = Field(unique=True, index=True)


class ClassLevel(ClassLevelBase, BaseModel, table=True):
    __tablename__ = "class_levels"

    id: Optional[int] = Field(default=None, primary_key=True)
    boards: List["Board"] = Relationship(
        back_populates="class_level", cascade_delete=True
    )


class ClassLevelCreate(ClassLevelBase):
    pass


class ClassLevelRead(ClassLevelBase):
    id: int


# --------------------
# Board
# --------------------
class BoardBase(SQLModel):
    name: str
    class_level_id: int = Field(foreign_key="class_levels.id")


class Board(BoardBase, BaseModel, table=True):
    __tablename__ = "boards"

    id: Optional[int] = Field(default=None, primary_key=True)
    class_level: ClassLevel = Relationship(back_populates="boards")
    mediums: List["Medium"] = Relationship(back_populates="board", cascade_delete=True)

    class Config:
        from_attributes = True


class BoardCreate(BoardBase):
    pass


class BoardRead(BoardBase):
    id: int
    class_level_name: Optional[str] = None


# --------------------
# Medium
# --------------------
class MediumBase(SQLModel):
    name: str
    board_id: int = Field(foreign_key="boards.id")


class Medium(MediumBase, BaseModel, table=True):
    __tablename__ = "mediums"

    id: Optional[int] = Field(default=None, primary_key=True)
    board: Board = Relationship(back_populates="mediums")
    subjects: List["Subject"] = Relationship(
        back_populates="medium", cascade_delete=True
    )


class MediumCreate(MediumBase):
    pass


class MediumRead(MediumBase):
    id: int
    board_name: Optional[str] = None


# --------------------
# Subject
# --------------------
class SubjectBase(SQLModel):
    name: str
    medium_id: int = Field(foreign_key="mediums.id")


class Subject(SubjectBase, BaseModel, table=True):
    __tablename__ = "subjects"

    id: Optional[int] = Field(default=None, primary_key=True)
    medium: Medium = Relationship(back_populates="subjects")
    chapters: List["Chapter"] = Relationship(
        back_populates="subject", cascade_delete=True
    )
    previous_year_question_papers: List["PreviousYearQuestionPaper"] = Relationship(
        back_populates="subject", cascade_delete=True
    )


class SubjectCreate(SubjectBase):
    pass


class SubjectRead(SubjectBase):
    id: int
    medium_name: Optional[str] = None


# --------------------
# Chapter
# --------------------
class ChapterBase(SQLModel):
    name: str
    subject_id: int = Field(foreign_key="subjects.id")
    icon_url: Optional[str] = None
    is_premium: bool = Field(default=False)
    enabled: bool = Field(default=True)
    chapter_number: int = Field(default=0, description="Used for ordering chapters")


class Chapter(ChapterBase, BaseModel, table=True):
    __tablename__ = "chapters"

    id: Optional[int] = Field(default=None, primary_key=True)
    subject: Subject = Relationship(back_populates="chapters")

    llm_textbooks: List["LLMTextbook"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    llm_images: List["LLMImage"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    llm_notes: List["LLMNote"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    qa_patterns: List["QAPattern"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    student_textbooks: List["StudentTextbook"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    additional_notes: List["AdditionalNotes"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    student_notes: List["StudentNotes"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    student_videos: List["StudentVideo"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    activities: List["ChapterActivity"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )
    activity_groups: List["ActivityGroup"] = Relationship(
        back_populates="chapter", cascade_delete=True
    )


class ChapterCreate(ChapterBase):
    pass


class ChapterUpdate(BaseModel):
    name: Optional[str] = None
    subject_id: Optional[int] = None
    icon_url: Optional[str] = None
    is_premium: Optional[bool] = None
    enabled: Optional[bool] = None
    chapter_number: Optional[int] = None


class ChapterRead(ChapterBase):
    id: int
    subject_name: Optional[str] = None
