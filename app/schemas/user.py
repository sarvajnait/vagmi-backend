from pydantic import BaseModel, EmailStr
from typing import Optional


class UserUpdate(BaseModel):
    """Request schema for updating user profile information.

    Attributes:
        name: Optional updated name of the user
        email: Optional updated email address (must be valid email format)
        avatar_url: Optional updated URL for the user's avatar/profile picture
    """

    name: Optional[str] = None
    email: Optional[EmailStr] = None
    avatar_url: Optional[str] = None
