from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.user import User
from app.models.user_notifications import UserNotification
from app.services.fcm_service import send_multicast


async def create_notification(
    user_id: int,
    title: str,
    body: str,
    notif_type: str,
    icon_emoji: str,
    db: AsyncSession,
    send_push: bool = True,
) -> UserNotification:
    notif = UserNotification(
        user_id=user_id,
        title=title,
        body=body,
        notif_type=notif_type,
        icon_emoji=icon_emoji,
    )
    db.add(notif)
    await db.flush()

    if send_push:
        try:
            user_result = await db.exec(
                select(User.fcm_token).where(User.id == user_id)
            )
            token = user_result.first()
            if token:
                send_multicast([token], title=title, body=body)
        except Exception:
            pass

    return notif


async def notify_milestone(
    user_id: int,
    milestone_days: int,
    milestone_name: str,
    db: AsyncSession,
) -> None:
    await create_notification(
        user_id=user_id,
        title=f"🏆 {milestone_name} unlocked!",
        body=f"You've hit a {milestone_days}-day streak. Keep it up!",
        notif_type="milestone",
        icon_emoji="🏆",
        db=db,
    )


async def notify_wrong_answer_reminder(
    user_id: int,
    count: int,
    db: AsyncSession,
) -> None:
    await create_notification(
        user_id=user_id,
        title=f"📕 {count} wrong answers due for revision",
        body="Questions from your notebook are ready to retry.",
        notif_type="wrong_answer_reminder",
        icon_emoji="📕",
        db=db,
    )


async def notify_chapter_complete(
    user_id: int,
    chapter_name: str,
    accuracy_pct: int,
    db: AsyncSession,
) -> None:
    await create_notification(
        user_id=user_id,
        title=f"✅ {chapter_name} completed!",
        body=f"You finished with {accuracy_pct}% accuracy.",
        notif_type="chapter_complete",
        icon_emoji="✅",
        db=db,
    )
