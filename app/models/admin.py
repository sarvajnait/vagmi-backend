from typing import Optional
from sqlmodel import Field
from app.models.base import BaseModel


class Admin(BaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    phone: str = Field(unique=True, index=True)
