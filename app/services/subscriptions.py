from datetime import date
from typing import Optional

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import SubscriptionPlan, UserSubscription


async def get_active_subscription_summary(
    session: AsyncSession, user_id: int
) -> Optional[dict]:
    today = date.today()
    result = await session.exec(
        select(UserSubscription, SubscriptionPlan)
        .join(SubscriptionPlan)
        .where(
            UserSubscription.user_id == user_id,
            UserSubscription.status == "active",
            UserSubscription.starts_at <= today,
            UserSubscription.ends_at >= today,
        )
        .order_by(UserSubscription.ends_at.desc())
    )
    row = result.first()
    if not row:
        return None
    sub, plan = row
    return {
        "id": sub.id,
        "plan_id": sub.plan_id,
        "plan_name": plan.name,
        "starts_at": sub.starts_at,
        "ends_at": sub.ends_at,
        "status": sub.status,
        "class_level_id": plan.class_level_id,
        "board_id": plan.board_id,
        "medium_id": plan.medium_id,
        "is_active": True,
    }
