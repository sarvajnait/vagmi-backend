from typing import TYPE_CHECKING, Optional
from sqlmodel import Field, Relationship, Column, Integer, ForeignKey
from app.models.base import BaseModel
from datetime import date
import bcrypt

if TYPE_CHECKING:
    from app.models.academic_hierarchy import ClassLevel
    from app.models.academic_hierarchy import Board
    from app.models.academic_hierarchy import Medium


class User(BaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(default=None, index=True)
    phone: str = Field(unique=True, index=True)
    hashed_password: str
    dob: Optional[date] = Field(default=None)
    gender: Optional[str] = Field(default=None, max_length=20)

    # Option 1: Using sa_column with ForeignKey
    class_level_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("class_levels.id", ondelete="SET NULL")),
    )
    board_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("boards.id", ondelete="SET NULL")),
    )
    medium_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("mediums.id", ondelete="SET NULL")),
    )

    class_level: Optional["ClassLevel"] = Relationship()
    board: Optional["Board"] = Relationship()
    medium: Optional["Medium"] = Relationship()

    def verify_password(self, password: str) -> bool:
        """Verify if the provided password matches the hash."""
        return bcrypt.checkpw(
            password.encode("utf-8"), self.hashed_password.encode("utf-8")
        )

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
