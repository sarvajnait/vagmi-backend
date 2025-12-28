from typing import Optional
from sqlmodel import SQLModel, Field, Column, Relationship
from sqlalchemy import JSON, Integer, ForeignKey, String
from app.models.base import BaseModel
from app.models.user import User


class LLMUsageBase(SQLModel):
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id")))
    model_name: Optional[str] = Field(default=None, sa_column=Column(String))
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: Optional[dict] = Field(
        default=None, sa_column=Column(JSON)
    )
    output_token_details: Optional[dict] = Field(
        default=None, sa_column=Column(JSON)
    )


class LLMUsage(LLMUsageBase, BaseModel, table=True):
    __tablename__ = "llmusage"

    id: Optional[int] = Field(default=None, primary_key=True)
    user: User = Relationship(back_populates="llm_usages")


class LLMUsageCreate(LLMUsageBase):
    pass


class LLMUsageRead(LLMUsageBase):
    id: int
