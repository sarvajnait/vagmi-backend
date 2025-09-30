"""Base models and common imports for all models."""

from datetime import datetime
from sqlmodel import Field, SQLModel
import sqlalchemy as sa


class BaseModel(SQLModel):
    """Base model with common fields.

    Attributes:
        created_at: When the record was created
        updated_at: Wen the record was last updated
    """

    created_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"onupdate": sa.func.now(), "server_default": sa.func.now()},
    )
