from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from app.models.base import BaseModel


class ActivityGenerationJobBase(SQLModel):
    job_type: str = Field(max_length=50)
    status: str = Field(default="pending", max_length=20)
    payload: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    result: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    error: Optional[str] = None


class ActivityGenerationJob(ActivityGenerationJobBase, BaseModel, table=True):
    __tablename__ = "activity_generation_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
