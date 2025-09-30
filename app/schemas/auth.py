"""This file contains the authentication schema for the application."""

import re
from datetime import datetime

from pydantic import BaseModel, SecretStr, Field
from pydantic_core import PydanticCustomError


class Token(BaseModel):
    """Token model for authentication.

    Attributes:
        access_token: The JWT access token.
        token_type: The type of token (always "bearer").
        expires_at: The token expiration timestamp.
    """

    access_token: str = Field(..., description="The JWT access token")
    token_type: str = Field(default="bearer", description="The type of token")
    expires_at: datetime = Field(..., description="The token expiration timestamp")


class TokenPair(BaseModel):
    """Pair of access and refresh tokens.

    Attributes:
        access_token: The JWT access token
        refresh_token: The JWT refresh token
        token_type: The type of token (always "bearer")
        expires_at: Expiry of the access token
        refresh_expires_at: Expiry of the refresh token
    """

    access_token: str = Field(..., description="The JWT access token")
    refresh_token: str = Field(..., description="The JWT refresh token")
    token_type: str = Field(default="bearer", description="The type of token")
    expires_at: datetime = Field(..., description="When the access token expires")
    refresh_expires_at: datetime = Field(
        ..., description="When the refresh token expires"
    )


class UserCreate(BaseModel):
    """Request model for user registration using phone."""

    phone: str = Field(..., description="User's phone number")
    password: str = Field(..., description="User's password")
    board: int = Field(..., description="Selected board ID")
    medium: int = Field(..., description="Selected medium ID")
    grade: int = Field(..., description="Selected grade/class level ID")


class UserResponse(BaseModel):
    id: int
    phone: str = Field(..., description="User's phone number")
    board_id: int
    medium_id: int
    class_level_id: int


class AuthResponse(BaseModel):
    user: UserResponse
    tokens: TokenPair
