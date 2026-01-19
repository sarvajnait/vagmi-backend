from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.admin.auth import get_current_user as get_current_admin
from app.models import (
    Chapter,
    ActivityGroup,
    ActivityGroupCreate,
    ChapterActivity,
)
from app.models.admin import Admin
from app.services.database import get_session

router = APIRouter()


class OrderUpdate(BaseModel):
    ids: list[int]


class ActivityGroupUpdate(BaseModel):
    name: Optional[str] = None
    timer_seconds: Optional[int] = None
    sort_order: Optional[int] = None


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


@router.post("/")
async def create_activity_group(
    payload: ActivityGroupCreate,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        chapter = await session.get(Chapter, payload.chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        if not payload.name or not payload.name.strip():
            raise HTTPException(status_code=400, detail="Group name is required")

        # Auto-assign sort_order if not provided
        if payload.sort_order is None:
            _result = await session.exec(
                select(func.max(ActivityGroup.sort_order)).where(
                    ActivityGroup.chapter_id == payload.chapter_id
                )
            )
            max_order = _result.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            sort_order = (max_order or 0) + 1
        else:
            sort_order = payload.sort_order

        activity_group = ActivityGroup(
            name=payload.name.strip(),
            chapter_id=payload.chapter_id,
            sort_order=sort_order,
        )
        session.add(activity_group)
        await session.commit()
        await session.refresh(activity_group)

        return {"message": "Activity group created", "data": activity_group.dict()}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_activity_groups(
    chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    try:
        query = select(ActivityGroup)
        if chapter_id is not None:
            query = query.where(ActivityGroup.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(ActivityGroup))
        _result = await session.exec(query)
        groups = _result.all()

        # Get activity counts for each group
        groups_data = []
        for group in groups:
            _count_result = await session.exec(
                select(func.count()).where(
                    ChapterActivity.activity_group_id == group.id
                )
            )
            activity_count = _count_result.first()
            if isinstance(activity_count, tuple):
                activity_count = activity_count[0]

            groups_data.append({
                **group.dict(),
                "activity_count": activity_count or 0
            })

        return {"data": groups_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{group_id}")
async def get_activity_group(
    group_id: int,
    session: AsyncSession = Depends(get_session)
):
    group = await session.get(ActivityGroup, group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Activity group not found")

    # Get activity count
    _count_result = await session.exec(
        select(func.count()).where(
            ChapterActivity.activity_group_id == group_id
        )
    )
    activity_count = _count_result.first()
    if isinstance(activity_count, tuple):
        activity_count = activity_count[0]

    return {
        "data": {
            **group.dict(),
            "activity_count": activity_count or 0
        }
    }


@router.put("/{group_id}")
async def update_activity_group(
    group_id: int,
    payload: ActivityGroupUpdate,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        group = await session.get(ActivityGroup, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Activity group not found")

        if payload.name is not None:
            if not payload.name.strip():
                raise HTTPException(status_code=400, detail="Group name cannot be empty")
            group.name = payload.name.strip()

        if payload.timer_seconds is not None:
            group.timer_seconds = payload.timer_seconds

        if payload.sort_order is not None:
            group.sort_order = payload.sort_order

        session.add(group)
        await session.commit()
        await session.refresh(group)
        return {"message": "Activity group updated", "data": group.dict()}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{group_id}")
async def delete_activity_group(
    group_id: int,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        group = await session.get(ActivityGroup, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Activity group not found")

        # Cascade delete will automatically remove all associated activities
        await session.delete(group)
        await session.commit()
        return {"message": "Activity group deleted"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/order")
async def reorder_activity_groups(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(ActivityGroup).where(
            ActivityGroup.chapter_id == chapter_id,
            ActivityGroup.id.in_(payload.ids),
        )
    )
    groups = _result.all()
    if len(groups) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid activity group ids for chapter")

    group_map = {group.id: group for group in groups}
    for index, group_id in enumerate(payload.ids, start=1):
        group_map[group_id].sort_order = index
        session.add(group_map[group_id])

    await session.commit()
    return {"message": "Activity group order updated"}
