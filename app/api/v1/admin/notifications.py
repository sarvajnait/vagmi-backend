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
from app.services.database import get_session
from app.services.fcm_service import send_multicast

router = APIRouter()


class SendNotificationRequest(BaseModel):
    title: str
    body: str
    # Optional filters — omit to send to ALL users
    class_level_id: Optional[int] = None
    board_id: Optional[int] = None
    medium_id: Optional[int] = None


@router.post("/send")
async def send_notification(
    body: SendNotificationRequest,
    _admin=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Send a push notification to users with FCM tokens.

    Optionally filter by class_level_id, board_id, medium_id.
    Omitting all filters sends to every user who has a token.
    """
    # Build filter list
    filters = [User.fcm_token.is_not(None)]
    if body.class_level_id is not None:
        filters.append(User.class_level_id == body.class_level_id)
    if body.board_id is not None:
        filters.append(User.board_id == body.board_id)
    if body.medium_id is not None:
        filters.append(User.medium_id == body.medium_id)

    result = await session.exec(select(User.fcm_token).where(*filters))
    tokens = [t for t in result.all() if t]

    if not tokens:
        raise HTTPException(status_code=404, detail="No users with FCM tokens found for the given filters")

    sent_count = send_multicast(tokens, title=body.title, body=body.body)

    # Persist notification record
    notification = Notification(
        title=body.title,
        body=body.body,
        class_level_id=body.class_level_id,
        board_id=body.board_id,
        medium_id=body.medium_id,
        sent_count=sent_count,
        sent_at=datetime.now(tz=timezone.utc),
    )
    session.add(notification)
    await session.commit()

    return {
        "message": f"Notification sent to {sent_count}/{len(tokens)} devices",
        "sent_count": sent_count,
        "total_tokens": len(tokens),
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
                "sent_count": n.sent_count,
                "sent_at": n.sent_at,
            }
            for n in notifications
        ]
    }
