from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from pydantic import BaseModel

from app.models.user import User
from app.schemas.auth import UserResponse
from app.services.database import get_session
from app.utils.sanitization import sanitize_string
from .auth import get_current_user  # reuse your auth dependency

router = APIRouter()


class UserProfileUpdate(BaseModel):
    # Only allow updating fields that exist on the User model and are safe
    name: str | None = None
    board_id: int | None = None
    medium_id: int | None = None
    class_level_id: int | None = None


@router.put("/me", response_model=UserResponse)
async def update_profile(
    user_update: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user = session.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    data = user_update.model_dump(exclude_unset=True)

    if "name" in data and data["name"] is not None:
        user.name = sanitize_string(data["name"])  # sanitize free-text

    if "board_id" in data:
        user.board_id = data["board_id"]

    if "medium_id" in data:
        user.medium_id = data["medium_id"]

    if "class_level_id" in data:
        user.class_level_id = data["class_level_id"]

    session.add(user)
    session.commit()
    session.refresh(user)

    return UserResponse(
        id=user.id,
        phone=user.phone,
        name=user.name,
        board_id=user.board_id,
        medium_id=user.medium_id,
        class_level_id=user.class_level_id,
    )
