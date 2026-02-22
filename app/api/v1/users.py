from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from pydantic import BaseModel
from datetime import date

from app.models.user import User
from app.schemas.auth import UserResponse
from app.services.database import get_session
from app.services.subscriptions import get_active_subscription_summary
from app.utils.sanitization import sanitize_string
from .auth import get_current_user  # reuse your auth dependency

router = APIRouter()


class UserProfileUpdate(BaseModel):
    # Only allow updating fields that exist on the User model and are safe
    name: str | None = None
    board_id: int | None = None
    medium_id: int | None = None
    class_level_id: int | None = None
    dob: date | None = None
    gender: str | None = None


@router.get("/me", response_model=UserResponse)
async def get_profile(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    user = await session.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    subscription = await get_active_subscription_summary(session, user.id)
    return UserResponse(
        id=user.id,
        phone=user.phone,
        name=user.name,
        board_id=user.board_id,
        medium_id=user.medium_id,
        class_level_id=user.class_level_id,
        dob=user.dob,
        gender=user.gender,
        is_premium=bool(subscription),
        subscription=subscription,
    )


@router.put("/me", response_model=UserResponse)
async def update_profile(
    user_update: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    user = await session.get(User, current_user.id)
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

    if "dob" in data:
        user.dob = data["dob"]

    if "gender" in data and data["gender"] is not None:
        user.gender = sanitize_string(data["gender"])

    session.add(user)
    await session.commit()
    await session.refresh(user)

    subscription = await get_active_subscription_summary(session, user.id)
    return UserResponse(
        id=user.id,
        phone=user.phone,
        name=user.name,
        board_id=user.board_id,
        medium_id=user.medium_id,
        class_level_id=user.class_level_id,
        dob=user.dob,
        gender=user.gender,
        is_premium=bool(subscription),
        subscription=subscription,
    )


class FcmTokenRequest(BaseModel):
    token: str


@router.post("/fcm-token", status_code=204)
async def register_fcm_token(
    body: FcmTokenRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Register or refresh the FCM device token for push notifications."""
    user = await session.get(User, current_user.id)
    user.fcm_token = body.token
    session.add(user)
    await session.commit()


@router.delete("/delete-charan")
async def delete_charan(
    session: AsyncSession = Depends(get_session),
):
    target_phone = "7406832289"
    _result = await session.exec(select(User).where(User.phone == target_phone))
    user = _result.first()

    if not user:
        return {"deleted": False, "message": "User not found"}

    await session.delete(user)
    await session.commit()
    return {"deleted": True}
