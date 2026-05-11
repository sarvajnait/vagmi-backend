"""Admin push notification endpoint."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.admin.auth import get_current_user
from app.models import User
from app.models.notifications import Notification
from app.models.user_notifications import UserNotification
from app.models.user_comp_profile import UserCompProfile
from app.services.database import get_session
from app.services.fcm_service import send_multicast

router = APIRouter()


class SendNotificationRequest(BaseModel):
    title: str
    body: str
    # Academic filters
    class_level_id: Optional[int] = None
    board_id: Optional[int] = None
    medium_id: Optional[int] = None
    # Competitive filters
    exam_id: Optional[int] = None
    comp_medium_id: Optional[int] = None
    level_id: Optional[int] = None


@router.post("/send")
async def send_notification(
    body: SendNotificationRequest,
    _admin=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Send a push notification to users with FCM tokens.

    Supply academic filters (class_level_id, board_id, medium_id) to target
    academic app users, or competitive filters (exam_id, comp_medium_id,
    level_id) to target comp app users. Omit all filters to send to everyone.
    """
    is_comp = any([body.exam_id, body.comp_medium_id, body.level_id])

    if is_comp:
        # Target comp users via user_comp_profiles join
        query = select(User.id).join(UserCompProfile, User.id == UserCompProfile.user_id)
        if body.exam_id is not None:
            query = query.where(UserCompProfile.exam_id == body.exam_id)
        if body.comp_medium_id is not None:
            query = query.where(UserCompProfile.comp_medium_id == body.comp_medium_id)
        if body.level_id is not None:
            query = query.where(UserCompProfile.level_id == body.level_id)
    else:
        # Target academic users directly on User table
        filters = []
        if body.class_level_id is not None:
            filters.append(User.class_level_id == body.class_level_id)
        if body.board_id is not None:
            filters.append(User.board_id == body.board_id)
        if body.medium_id is not None:
            filters.append(User.medium_id == body.medium_id)
        query = select(User.id).where(*filters) if filters else select(User.id)

    user_ids_result = await session.exec(query)
    all_user_ids = user_ids_result.all()

    if not all_user_ids:
        raise HTTPException(status_code=404, detail="No users found for the given filters")

    # FCM push — best-effort, inbox always written
    if is_comp:
        token_query = select(User.fcm_token).join(
            UserCompProfile, User.id == UserCompProfile.user_id
        ).where(User.fcm_token.is_not(None))
        if body.exam_id is not None:
            token_query = token_query.where(UserCompProfile.exam_id == body.exam_id)
        if body.comp_medium_id is not None:
            token_query = token_query.where(UserCompProfile.comp_medium_id == body.comp_medium_id)
        if body.level_id is not None:
            token_query = token_query.where(UserCompProfile.level_id == body.level_id)
    else:
        filters = []
        if body.class_level_id is not None:
            filters.append(User.class_level_id == body.class_level_id)
        if body.board_id is not None:
            filters.append(User.board_id == body.board_id)
        if body.medium_id is not None:
            filters.append(User.medium_id == body.medium_id)
        token_query = select(User.fcm_token).where(User.fcm_token.is_not(None), *filters)

    token_result = await session.exec(token_query)
    tokens = [t for t in token_result.all() if t]
    sent_count = send_multicast(tokens, title=body.title, body=body.body) if tokens else 0

    # Persist broadcast log
    notification = Notification(
        title=body.title,
        body=body.body,
        class_level_id=body.class_level_id,
        board_id=body.board_id,
        medium_id=body.medium_id,
        exam_id=body.exam_id,
        comp_medium_id=body.comp_medium_id,
        level_id=body.level_id,
        sent_count=sent_count,
        sent_at=datetime.now(tz=timezone.utc),
    )
    session.add(notification)

    session.add_all([
        UserNotification(
            user_id=uid,
            title=body.title,
            body=body.body,
            notif_type="admin_broadcast",
            icon_emoji="📢",
        )
        for uid in all_user_ids
    ])

    await session.commit()

    return {
        "message": f"Notification sent to {sent_count}/{len(tokens)} devices" if tokens else "Saved to inbox (no FCM tokens found)",
        "sent_count": sent_count,
        "total_tokens": len(tokens),
        "inbox_recipients": len(all_user_ids),
    }


@router.get("/history")
async def notification_history(
    _admin=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """List past sent notifications."""
    result = await session.exec(
        select(Notification).order_by(Notification.sent_at.desc()).limit(50)
    )
    notifications = result.all()
    return {
        "data": [
            {
                "id": n.id,
                "title": n.title,
                "body": n.body,
                "class_level_id": n.class_level_id,
                "board_id": n.board_id,
                "medium_id": n.medium_id,
                "exam_id": n.exam_id,
                "comp_medium_id": n.comp_medium_id,
                "level_id": n.level_id,
                "sent_count": n.sent_count,
                "sent_at": n.sent_at,
            }
            for n in notifications
        ]
    }
