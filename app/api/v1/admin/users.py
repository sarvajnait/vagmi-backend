from datetime import date
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select

from app.api.v1.admin.auth import get_current_user
from app.models import SubscriptionPlan, User, UserSubscription
from app.services.database import get_session

router = APIRouter()


@router.get("/")
async def list_users(
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    class_level_id: Optional[int] = Query(None),
    board_id: Optional[int] = Query(None),
    medium_id: Optional[int] = Query(None),
    _admin=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    search = (q or "").strip()
    filters = []
    if search:
        like = f"%{search}%"
        filters.append(or_(User.name.ilike(like), User.phone.ilike(like)))
    if class_level_id is not None:
        filters.append(User.class_level_id == class_level_id)
    if board_id is not None:
        filters.append(User.board_id == board_id)
    if medium_id is not None:
        filters.append(User.medium_id == medium_id)

    count_query = select(func.count()).select_from(User)
    if filters:
        count_query = count_query.where(*filters)
    _result = await session.exec(count_query)
    total_result = _result.first()
    if isinstance(total_result, tuple):
        total = total_result[0]
    else:
        total = total_result or 0

    query = select(User)
    if filters:
        query = query.where(*filters)
    offset = (page - 1) * page_size
    _result = await session.exec(
        query.order_by(User.created_at.desc()).offset(offset).limit(page_size)
    )
    users = _result.all()
    user_ids = [u.id for u in users if u.id is not None]
    active_subs_by_user: Dict[int, Dict] = {}

    if user_ids:
        today = date.today()
        _result = await session.exec(
            select(UserSubscription, SubscriptionPlan)
            .join(SubscriptionPlan)
            .where(
                UserSubscription.user_id.in_(user_ids),
                UserSubscription.status == "active",
                UserSubscription.starts_at <= today,
                UserSubscription.ends_at >= today,
            )
            .order_by(UserSubscription.ends_at.desc()))
        subs = _result.all()
        for sub, plan in subs:
            if sub.user_id in active_subs_by_user:
                continue
            active_subs_by_user[sub.user_id] = {
                "id": sub.id,
                "plan_id": sub.plan_id,
                "plan_name": plan.name,
                "starts_at": sub.starts_at,
                "ends_at": sub.ends_at,
                "status": sub.status,
            }

    data = [
        {
            "id": user.id,
            "name": user.name,
            "phone": user.phone,
            "class_level_id": user.class_level_id,
            "board_id": user.board_id,
            "medium_id": user.medium_id,
            "active_subscription": active_subs_by_user.get(user.id),
        }
        for user in users
    ]

    return {
        "data": data,
        "meta": {
            "page": page,
            "page_size": page_size,
            "total": total,
        },
    }
