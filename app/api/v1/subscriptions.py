from datetime import date
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select

from app.models import (
    Board,
    ClassLevel,
    Medium,
    User,
    SubscriptionPlan,
    SubscriptionPlanCreate,
    SubscriptionPlanRead,
    SubscriptionPlanUpdate,
    UserSubscription,
    UserSubscriptionCreate,
    UserSubscriptionRead,
    UserSubscriptionUpdate,
)
from app.services.database import get_session

router = APIRouter()


class BulkSubscriptionCreate(BaseModel):
    user_ids: List[int] = Field(min_length=1)
    plan_id: int
    starts_at: date = date(2026, 1, 1)
    ends_at: date = date(2026, 2, 1)
    status: str = "active"
    notes: Optional[str] = None


# --------------------
# Subscription Plans
# --------------------
@router.get("/plans", response_model=Dict[str, List[SubscriptionPlanRead]])
async def list_plans(
    class_level_id: Optional[int] = Query(None),
    board_id: Optional[int] = Query(None),
    medium_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    query = (
        select(SubscriptionPlan, ClassLevel, Board, Medium)
        .select_from(SubscriptionPlan)
        .join(ClassLevel, SubscriptionPlan.class_level_id == ClassLevel.id)
        .join(Board, SubscriptionPlan.board_id == Board.id)
        .join(Medium, SubscriptionPlan.medium_id == Medium.id)
    )

    if class_level_id is not None:
        query = query.where(SubscriptionPlan.class_level_id == class_level_id)
    if board_id is not None:
        query = query.where(SubscriptionPlan.board_id == board_id)
    if medium_id is not None:
        query = query.where(SubscriptionPlan.medium_id == medium_id)

    _result = await session.exec(query.order_by(ClassLevel.name, Board.name, Medium.name))
    results = _result.all()

    plans = [
        SubscriptionPlanRead(
            id=plan.id,
            name=plan.name,
            class_level_id=plan.class_level_id,
            board_id=plan.board_id,
            medium_id=plan.medium_id,
            amount_inr=plan.amount_inr,
            is_active=plan.is_active,
            description=plan.description,
            class_level_name=class_level.name,
            board_name=board.name,
            medium_name=medium.name,
        )
        for plan, class_level, board, medium in results
    ]
    return {"data": plans}


@router.get("/plans/{plan_id}", response_model=Dict[str, SubscriptionPlanRead])
async def get_plan(plan_id: int, session: AsyncSession = Depends(get_session)):
    plan = await session.get(SubscriptionPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")

    class_level = await session.get(ClassLevel, plan.class_level_id)
    board = await session.get(Board, plan.board_id)
    medium = await session.get(Medium, plan.medium_id)

    return {
        "data": SubscriptionPlanRead(
            id=plan.id,
            name=plan.name,
            class_level_id=plan.class_level_id,
            board_id=plan.board_id,
            medium_id=plan.medium_id,
            amount_inr=plan.amount_inr,
            is_active=plan.is_active,
            description=plan.description,
            class_level_name=class_level.name if class_level else None,
            board_name=board.name if board else None,
            medium_name=medium.name if medium else None,
        )
    }


@router.post("/plans", response_model=Dict[str, SubscriptionPlanRead])
async def create_plan(
    plan: SubscriptionPlanCreate, session: AsyncSession = Depends(get_session)
):
    class_level = await session.get(ClassLevel, plan.class_level_id)
    board = await session.get(Board, plan.board_id)
    medium = await session.get(Medium, plan.medium_id)

    if not class_level or not board or not medium:
        raise HTTPException(status_code=400, detail="Invalid class/board/medium")

    _result = await session.exec(
        select(SubscriptionPlan).where(
            SubscriptionPlan.class_level_id == plan.class_level_id,
            SubscriptionPlan.board_id == plan.board_id,
            SubscriptionPlan.medium_id == plan.medium_id,
        )
    )
    existing = _result.first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Plan already exists for this class/board/medium",
        )

    db_plan = SubscriptionPlan.model_validate(plan)
    session.add(db_plan)
    await session.commit()
    await session.refresh(db_plan)

    return {
        "data": SubscriptionPlanRead(
            id=db_plan.id,
            name=db_plan.name,
            class_level_id=db_plan.class_level_id,
            board_id=db_plan.board_id,
            medium_id=db_plan.medium_id,
            amount_inr=db_plan.amount_inr,
            is_active=db_plan.is_active,
            description=db_plan.description,
            class_level_name=class_level.name,
            board_name=board.name,
            medium_name=medium.name,
        )
    }


@router.put("/plans/{plan_id}", response_model=Dict[str, SubscriptionPlanRead])
async def update_plan(
    plan_id: int,
    plan_data: SubscriptionPlanUpdate,
    session: AsyncSession = Depends(get_session),
):
    db_plan = await session.get(SubscriptionPlan, plan_id)
    if not db_plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")

    update_data = plan_data.model_dump(exclude_unset=True)

    if {
        "class_level_id",
        "board_id",
        "medium_id",
    } & update_data.keys():
        class_level_id = update_data.get("class_level_id", db_plan.class_level_id)
        board_id = update_data.get("board_id", db_plan.board_id)
        medium_id = update_data.get("medium_id", db_plan.medium_id)

        class_level = await session.get(ClassLevel, class_level_id)
        board = await session.get(Board, board_id)
        medium = await session.get(Medium, medium_id)
        if not class_level or not board or not medium:
            raise HTTPException(status_code=400, detail="Invalid class/board/medium")

        _result = await session.exec(
            select(SubscriptionPlan).where(
                SubscriptionPlan.class_level_id == class_level_id,
                SubscriptionPlan.board_id == board_id,
                SubscriptionPlan.medium_id == medium_id,
                SubscriptionPlan.id != plan_id,
            ))

        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Another plan already exists for this class/board/medium",
            )

        db_plan.class_level_id = class_level_id
        db_plan.board_id = board_id
        db_plan.medium_id = medium_id

    for field, value in update_data.items():
        if field in {"class_level_id", "board_id", "medium_id"}:
            continue
        setattr(db_plan, field, value)

    session.add(db_plan)
    await session.commit()
    await session.refresh(db_plan)

    class_level = await session.get(ClassLevel, db_plan.class_level_id)
    board = await session.get(Board, db_plan.board_id)
    medium = await session.get(Medium, db_plan.medium_id)

    return {
        "data": SubscriptionPlanRead(
            id=db_plan.id,
            name=db_plan.name,
            class_level_id=db_plan.class_level_id,
            board_id=db_plan.board_id,
            medium_id=db_plan.medium_id,
            amount_inr=db_plan.amount_inr,
            is_active=db_plan.is_active,
            description=db_plan.description,
            class_level_name=class_level.name if class_level else None,
            board_name=board.name if board else None,
            medium_name=medium.name if medium else None,
        )
    }


@router.delete("/plans/{plan_id}")
async def delete_plan(plan_id: int, session: AsyncSession = Depends(get_session)):
    plan = await session.get(SubscriptionPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")
    await session.delete(plan)
    await session.commit()
    return {"message": "Subscription plan deleted successfully"}


# --------------------
# User Subscriptions
# --------------------
@router.get("/subscriptions", response_model=Dict[str, List[UserSubscriptionRead]])
async def list_subscriptions(
    user_id: Optional[int] = Query(None),
    plan_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    active_only: bool = Query(False),
    session: AsyncSession = Depends(get_session),
):
    query = select(UserSubscription, User, SubscriptionPlan).join(User).join(
        SubscriptionPlan
    )

    if user_id is not None:
        query = query.where(UserSubscription.user_id == user_id)
    if plan_id is not None:
        query = query.where(UserSubscription.plan_id == plan_id)
    if status is not None:
        query = query.where(UserSubscription.status == status)
    if active_only:
        today = date.today()
        query = query.where(
            UserSubscription.starts_at <= today, UserSubscription.ends_at >= today
        )

    _result = await session.exec(query.order_by(UserSubscription.starts_at.desc()))
    results = _result.all()

    subs = [
        UserSubscriptionRead(
            id=sub.id,
            user_id=sub.user_id,
            plan_id=sub.plan_id,
            starts_at=sub.starts_at,
            ends_at=sub.ends_at,
            status=sub.status,
            notes=sub.notes,
            user_name=user.name,
            plan_name=plan.name,
        )
        for sub, user, plan in results
    ]
    return {"data": subs}


@router.get("/subscriptions/{subscription_id}", response_model=Dict[str, UserSubscriptionRead])
async def get_subscription(
    subscription_id: int, session: AsyncSession = Depends(get_session)
):
    sub = await session.get(UserSubscription, subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")

    user = await session.get(User, sub.user_id)
    plan = await session.get(SubscriptionPlan, sub.plan_id)

    return {
        "data": UserSubscriptionRead(
            id=sub.id,
            user_id=sub.user_id,
            plan_id=sub.plan_id,
            starts_at=sub.starts_at,
            ends_at=sub.ends_at,
            status=sub.status,
            notes=sub.notes,
            user_name=user.name if user else None,
            plan_name=plan.name if plan else None,
        )
    }


@router.post("/subscriptions", response_model=Dict[str, UserSubscriptionRead])
async def create_subscription(
    subscription: UserSubscriptionCreate, session: AsyncSession = Depends(get_session)
):
    user = await session.get(User, subscription.user_id)
    plan = await session.get(SubscriptionPlan, subscription.plan_id)
    if not user or not plan:
        raise HTTPException(status_code=400, detail="Invalid user or plan")

    _result = await session.exec(
        select(UserSubscription).where(
            UserSubscription.user_id == subscription.user_id,
            UserSubscription.plan_id == subscription.plan_id,
            UserSubscription.status == "active",
            UserSubscription.ends_at >= subscription.starts_at,
        )
    )
    existing = _result.first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Active subscription already exists for this user and plan",
        )

    db_sub = UserSubscription.model_validate(subscription)
    session.add(db_sub)
    await session.commit()
    await session.refresh(db_sub)

    return {
        "data": UserSubscriptionRead(
            id=db_sub.id,
            user_id=db_sub.user_id,
            plan_id=db_sub.plan_id,
            starts_at=db_sub.starts_at,
            ends_at=db_sub.ends_at,
            status=db_sub.status,
            notes=db_sub.notes,
            user_name=user.name,
            plan_name=plan.name,
        )
    }


@router.post("/subscriptions/bulk")
async def bulk_create_subscriptions(
    payload: BulkSubscriptionCreate, session: AsyncSession = Depends(get_session)
):
    plan = await session.get(SubscriptionPlan, payload.plan_id)
    if not plan:
        raise HTTPException(status_code=400, detail="Invalid plan")

    _result = await session.exec(select(User).where(User.id.in_(payload.user_ids)))
    users = _result.all()
    user_map = {u.id: u for u in users}

    created = []
    skipped = []
    errors = []

    for user_id in payload.user_ids:
        user = user_map.get(user_id)
        if not user:
            errors.append({"user_id": user_id, "reason": "User not found"})
            continue

        _result = await session.exec(
            select(UserSubscription).where(
                UserSubscription.user_id == user_id,
                UserSubscription.plan_id == payload.plan_id,
                UserSubscription.status == "active",
                UserSubscription.ends_at >= payload.starts_at,
            )
        )
        existing = _result.first()
        if existing:
            skipped.append(
                {"user_id": user_id, "reason": "Active subscription exists"}
            )
            continue

        db_sub = UserSubscription(
            user_id=user_id,
            plan_id=payload.plan_id,
            starts_at=payload.starts_at,
            ends_at=payload.ends_at,
            status=payload.status,
            notes=payload.notes,
        )
        session.add(db_sub)
        created.append(user_id)

    await session.commit()

    return {
        "created_count": len(created),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "created_user_ids": created,
        "skipped": skipped,
        "errors": errors,
    }


@router.put("/subscriptions/{subscription_id}", response_model=Dict[str, UserSubscriptionRead])
async def update_subscription(
    subscription_id: int,
    subscription_data: UserSubscriptionUpdate,
    session: AsyncSession = Depends(get_session),
):
    db_sub = await session.get(UserSubscription, subscription_id)
    if not db_sub:
        raise HTTPException(status_code=404, detail="Subscription not found")

    update_data = subscription_data.model_dump(exclude_unset=True)

    if "plan_id" in update_data:
        plan = await session.get(SubscriptionPlan, update_data["plan_id"])
        if not plan:
            raise HTTPException(status_code=400, detail="Invalid plan")
        db_sub.plan_id = update_data["plan_id"]

    for field, value in update_data.items():
        if field == "plan_id":
            continue
        setattr(db_sub, field, value)

    session.add(db_sub)
    await session.commit()
    await session.refresh(db_sub)

    user = await session.get(User, db_sub.user_id)
    plan = await session.get(SubscriptionPlan, db_sub.plan_id)

    return {
        "data": UserSubscriptionRead(
            id=db_sub.id,
            user_id=db_sub.user_id,
            plan_id=db_sub.plan_id,
            starts_at=db_sub.starts_at,
            ends_at=db_sub.ends_at,
            status=db_sub.status,
            notes=db_sub.notes,
            user_name=user.name if user else None,
            plan_name=plan.name if plan else None,
        )
    }


@router.delete("/subscriptions/{subscription_id}")
async def delete_subscription(
    subscription_id: int, session: AsyncSession = Depends(get_session)
):
    sub = await session.get(UserSubscription, subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    await session.delete(sub)
    await session.commit()
    return {"message": "Subscription deleted successfully"}
