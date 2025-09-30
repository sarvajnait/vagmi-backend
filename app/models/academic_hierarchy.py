from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from app.models.base import BaseModel


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


class Chapter(ChapterBase, BaseModel, table=True):
    __tablename__ = "chapters"

    id: Optional[int] = Field(default=None, primary_key=True)
    subject: Subject = Relationship(back_populates="chapters")


class ChapterCreate(ChapterBase):
    pass


class ChapterRead(ChapterBase):
    id: int
    subject_name: Optional[str] = None
