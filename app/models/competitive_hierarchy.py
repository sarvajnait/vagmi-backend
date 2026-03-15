from typing import Optional, List
from sqlalchemy import Column, Integer, ForeignKey
from sqlmodel import SQLModel, Field, Relationship
from app.models.base import BaseModel


# --------------------
# Exam Category
# --------------------
class ExamCategoryBase(SQLModel):
    name: str = Field(unique=True, index=True)
    sort_order: Optional[int] = Field(default=None)


class ExamCategory(ExamCategoryBase, BaseModel, table=True):
    __tablename__ = "exam_categories"

    id: Optional[int] = Field(default=None, primary_key=True)
    exams: List["Exam"] = Relationship(back_populates="exam_category", cascade_delete=True)


class ExamCategoryCreate(ExamCategoryBase):
    pass


class ExamCategoryRead(ExamCategoryBase):
    id: int


# --------------------
# Exam
# --------------------
class ExamBase(SQLModel):
    name: str
    exam_category_id: int = Field(
        sa_column=Column(Integer, ForeignKey("exam_categories.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    sort_order: Optional[int] = Field(default=None)


class Exam(ExamBase, BaseModel, table=True):
    __tablename__ = "exams"

    id: Optional[int] = Field(default=None, primary_key=True)
    exam_category: ExamCategory = Relationship(back_populates="exams")
    mediums: List["CompExamMedium"] = Relationship(back_populates="exam", cascade_delete=True)


class ExamCreate(ExamBase):
    pass


class ExamRead(ExamBase):
    id: int
    exam_category_name: Optional[str] = None


# --------------------
# Comp Exam Medium
# --------------------
class CompExamMediumBase(SQLModel):
    name: str
    exam_id: int = Field(
        sa_column=Column(Integer, ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    sort_order: Optional[int] = Field(default=None)


class CompExamMedium(CompExamMediumBase, BaseModel, table=True):
    __tablename__ = "comp_exam_mediums"

    id: Optional[int] = Field(default=None, primary_key=True)
    exam: Exam = Relationship(back_populates="mediums")
    levels: List["Level"] = Relationship(back_populates="medium", cascade_delete=True)


class CompExamMediumCreate(CompExamMediumBase):
    pass


class CompExamMediumRead(CompExamMediumBase):
    id: int
    exam_name: Optional[str] = None


# --------------------
# Level
# --------------------
class LevelBase(SQLModel):
    name: str
    medium_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_exam_mediums.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    sort_order: Optional[int] = Field(default=None)


class Level(LevelBase, BaseModel, table=True):
    __tablename__ = "comp_levels"

    id: Optional[int] = Field(default=None, primary_key=True)
    medium: CompExamMedium = Relationship(back_populates="levels")
    subjects: List["CompSubject"] = Relationship(back_populates="level", cascade_delete=True)


class LevelCreate(LevelBase):
    pass


class LevelRead(LevelBase):
    id: int
    medium_name: Optional[str] = None


# --------------------
# Comp Subject
# --------------------
class CompSubjectBase(SQLModel):
    name: str
    level_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_levels.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    sort_order: Optional[int] = Field(default=None)


class CompSubject(CompSubjectBase, BaseModel, table=True):
    __tablename__ = "comp_subjects"

    id: Optional[int] = Field(default=None, primary_key=True)
    level: Level = Relationship(back_populates="subjects")
    chapters: List["CompChapter"] = Relationship(back_populates="subject", cascade_delete=True)


class CompSubjectCreate(CompSubjectBase):
    pass


class CompSubjectRead(CompSubjectBase):
    id: int
    level_name: Optional[str] = None


# --------------------
# Comp Chapter
# --------------------
class CompChapterBase(SQLModel):
    name: str
    subject_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_subjects.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    icon_url: Optional[str] = None
    is_premium: bool = Field(default=False)
    enabled: bool = Field(default=True)
    chapter_number: int = Field(default=0)
    include_subchapters: bool = Field(default=False)
    sort_order: Optional[int] = Field(default=None)


class CompChapter(CompChapterBase, BaseModel, table=True):
    __tablename__ = "comp_chapters"

    id: Optional[int] = Field(default=None, primary_key=True)
    subject: CompSubject = Relationship(back_populates="chapters")
    sub_chapters: List["SubChapter"] = Relationship(back_populates="chapter", cascade_delete=True)


class CompChapterCreate(CompChapterBase):
    pass


class CompChapterUpdate(SQLModel):
    name: Optional[str] = None
    subject_id: Optional[int] = None
    icon_url: Optional[str] = None
    is_premium: Optional[bool] = None
    enabled: Optional[bool] = None
    chapter_number: Optional[int] = None
    include_subchapters: Optional[bool] = None
    sort_order: Optional[int] = None


class CompChapterRead(CompChapterBase):
    id: int
    subject_name: Optional[str] = None


# --------------------
# Sub Chapter
# --------------------
class SubChapterBase(SQLModel):
    name: str
    chapter_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_chapters.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    chapter_number: int = Field(default=0)
    sort_order: Optional[int] = Field(default=None)


class SubChapter(SubChapterBase, BaseModel, table=True):
    __tablename__ = "comp_sub_chapters"

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter: CompChapter = Relationship(back_populates="sub_chapters")


class SubChapterCreate(SubChapterBase):
    pass


class SubChapterUpdate(SQLModel):
    name: Optional[str] = None
    chapter_id: Optional[int] = None
    chapter_number: Optional[int] = None
    sort_order: Optional[int] = None


class SubChapterRead(SubChapterBase):
    id: int
    chapter_name: Optional[str] = None
