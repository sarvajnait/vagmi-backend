from pydantic import BaseModel, EmailStr
from typing import Optional


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    avatar_url: Optional[str] = None
