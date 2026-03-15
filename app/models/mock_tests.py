from typing import Optional, List
from sqlalchemy import ARRAY, String, Column, Integer, ForeignKey
from sqlmodel import Field, Relationship, SQLModel
from app.models.base import BaseModel


class MockTestBase(SQLModel):
    level_id: int = Field(
        sa_column=Column(Integer, ForeignKey("comp_levels.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    title: str
    duration_minutes: Optional[int] = Field(default=None)
    total_marks: Optional[int] = Field(default=None)
    enabled: bool = Field(default=True)
    sort_order: Optional[int] = Field(default=None)


class MockTest(MockTestBase, BaseModel, table=True):
    __tablename__ = "mock_tests"

    id: Optional[int] = Field(default=None, primary_key=True)
    questions: List["MockTestQuestion"] = Relationship(back_populates="mock_test", cascade_delete=True)


class MockTestCreate(MockTestBase):
    pass


class MockTestUpdate(SQLModel):
    title: Optional[str] = None
    duration_minutes: Optional[int] = None
    total_marks: Optional[int] = None
    enabled: Optional[bool] = None
    sort_order: Optional[int] = None


class MockTestRead(MockTestBase):
    id: int


class MockTestQuestionBase(SQLModel):
    mock_test_id: int = Field(
        sa_column=Column(Integer, ForeignKey("mock_tests.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    question_text: str
    options: Optional[list[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    correct_option_index: Optional[int] = None
    explanation: Optional[str] = None
    marks: int = Field(default=1)
    sort_order: Optional[int] = Field(default=None)


class MockTestQuestion(MockTestQuestionBase, BaseModel, table=True):
    __tablename__ = "mock_test_questions"

    id: Optional[int] = Field(default=None, primary_key=True)
    mock_test: MockTest = Relationship(back_populates="questions")


class MockTestQuestionCreate(MockTestQuestionBase):
    pass


class MockTestQuestionUpdate(SQLModel):
    question_text: Optional[str] = None
    options: Optional[list[str]] = None
    correct_option_index: Optional[int] = None
    explanation: Optional[str] = None
    marks: Optional[int] = None
    sort_order: Optional[int] = None


class MockTestQuestionRead(MockTestQuestionBase):
    id: int
