from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import ActivityGenerationJob
from app.models.comp_activities import (
    CompActivityGroup, CompActivityGroupCreate, CompActivityGroupRead,
    CompChapterActivity, CompChapterActivityCreate, CompChapterActivityRead,
    CompTopic, CompTopicCreate, CompTopicRead,
)
from app.services.activity_jobs import enqueue_activity_job
from app.services.database import get_session
from app.utils.files import upload_to_do, delete_from_do

router = APIRouter()


class OrderUpdate(BaseModel):
    ids: list[int]


class CompActivityGroupUpdate(BaseModel):
    name: Optional[str] = None
    timer_seconds: Optional[int] = None
    sort_order: Optional[int] = None


class GenerateCompActivitiesRequest(BaseModel):
    comp_chapter_id: Optional[int] = None
    sub_chapter_id: Optional[int] = None
    topic_titles: list[str] = []
    mcq_count: int = 5
    descriptive_count: int = 5
    activity_group_id: Optional[int] = None


class CompTopicsRequest(BaseModel):
    comp_chapter_id: Optional[int] = None
    sub_chapter_id: Optional[int] = None


class PublishRequest(BaseModel):
    ids: list[int]
    is_published: bool = True


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


def _resolve_fk(comp_chapter_id: Optional[int], sub_chapter_id: Optional[int]):
    if comp_chapter_id is None and sub_chapter_id is None:
        raise HTTPException(status_code=400, detail="Either comp_chapter_id or sub_chapter_id must be provided")
    return comp_chapter_id, sub_chapter_id


# ============================================================
# Activity Groups
# ============================================================

@router.post("/activity-groups")
async def create_comp_activity_group(
    payload: CompActivityGroupCreate,
    session: AsyncSession = Depends(get_session),
):
    try:
        comp_chapter_id, sub_chapter_id = _resolve_fk(payload.comp_chapter_id, payload.sub_chapter_id)
        if not payload.name or not payload.name.strip():
            raise HTTPException(status_code=400, detail="Group name is required")
        filter_col = CompActivityGroup.comp_chapter_id if comp_chapter_id else CompActivityGroup.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        if payload.sort_order is None:
            _result = await session.exec(select(func.max(CompActivityGroup.sort_order)).where(filter_col == filter_val))
            max_order = _result.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            sort_order = (max_order or 0) + 1
        else:
            sort_order = payload.sort_order
        group = CompActivityGroup(
            name=payload.name.strip(),
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            timer_seconds=payload.timer_seconds,
            sort_order=sort_order,
        )
        session.add(group)
        await session.commit()
        await session.refresh(group)
        return {"message": "Activity group created", "data": group.dict()}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity-groups")
async def get_comp_activity_groups(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    try:
        query = select(CompActivityGroup)
        if comp_chapter_id is not None:
            query = query.where(CompActivityGroup.comp_chapter_id == comp_chapter_id)
        elif sub_chapter_id is not None:
            query = query.where(CompActivityGroup.sub_chapter_id == sub_chapter_id)
        query = query.order_by(*sort_ordering(CompActivityGroup))
        result = await session.exec(query)
        groups = result.all()
        groups_data = []
        for group in groups:
            _count = await session.exec(select(func.count()).where(CompChapterActivity.activity_group_id == group.id))
            activity_count = _count.first()
            if isinstance(activity_count, tuple):
                activity_count = activity_count[0]
            groups_data.append({**group.dict(), "activity_count": activity_count or 0})
        return {"data": groups_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity-groups/{group_id}")
async def get_comp_activity_group(group_id: int, session: AsyncSession = Depends(get_session)):
    group = await session.get(CompActivityGroup, group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Activity group not found")
    _count = await session.exec(select(func.count()).where(CompChapterActivity.activity_group_id == group_id))
    activity_count = _count.first()
    if isinstance(activity_count, tuple):
        activity_count = activity_count[0]
    return {"data": {**group.dict(), "activity_count": activity_count or 0}}


@router.put("/activity-groups/{group_id}")
async def update_comp_activity_group(
    group_id: int,
    payload: CompActivityGroupUpdate,
    session: AsyncSession = Depends(get_session),
):
    try:
        group = await session.get(CompActivityGroup, group_id)
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


@router.delete("/activity-groups/{group_id}")
async def delete_comp_activity_group(group_id: int, session: AsyncSession = Depends(get_session)):
    try:
        group = await session.get(CompActivityGroup, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Activity group not found")
        await session.delete(group)
        await session.commit()
        return {"message": "Activity group deleted"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/activity-groups/order")
async def reorder_comp_activity_groups(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    filter_col = CompActivityGroup.comp_chapter_id if comp_chapter_id else CompActivityGroup.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(CompActivityGroup).where(filter_col == filter_val, CompActivityGroup.id.in_(payload.ids)))
    groups = _result.all()
    if len(groups) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid activity group ids")
    group_map = {g.id: g for g in groups}
    for index, gid in enumerate(payload.ids, start=1):
        group_map[gid].sort_order = index
        session.add(group_map[gid])
    await session.commit()
    return {"message": "Activity group order updated"}


# ============================================================
# Activities
# ============================================================

@router.post("/activities")
async def create_comp_activity(
    activity_group_id: int = Form(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    type: str = Form(...),
    question_text: str = Form(...),
    options: Optional[list[str]] = Form(None),
    correct_option_index: Optional[int] = Form(None),
    answer_text: Optional[str] = Form(None),
    answer_description: Optional[str] = Form(None),
    is_published: bool = Form(True),
    sort_order: Optional[int] = Form(None),
    answer_image: UploadFile | None = File(None),
    session: AsyncSession = Depends(get_session),
):
    try:
        group = await session.get(CompActivityGroup, activity_group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Activity group not found")

        cleaned_options = [opt.strip() for opt in options] if options else None

        if sort_order is None:
            _result = await session.exec(
                select(func.max(CompChapterActivity.sort_order)).where(
                    CompChapterActivity.activity_group_id == activity_group_id
                )
            )
            max_order = _result.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            sort_order = (max_order or 0) + 1

        answer_image_url = None
        if answer_image:
            folder_id = comp_chapter_id or sub_chapter_id
            do_path = f"comp/chapters/{folder_id}/activities/answers"
            answer_image_url = upload_to_do(answer_image, do_path)

        activity = CompChapterActivity(
            activity_group_id=activity_group_id,
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            type=type,
            question_text=question_text.strip(),
            options=cleaned_options if type == "mcq" else None,
            correct_option_index=correct_option_index if type == "mcq" else None,
            answer_text=answer_text.strip() if answer_text and type == "descriptive" else None,
            answer_description=answer_description.strip() if answer_description else None,
            answer_image_url=answer_image_url,
            is_published=is_published,
            sort_order=sort_order,
        )
        session.add(activity)
        await session.commit()
        await session.refresh(activity)
        return {"message": "Activity created", "data": activity.dict()}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activities")
async def get_comp_activities(
    activity_group_id: Optional[int] = None,
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    status: str = Query("all"),
    session: AsyncSession = Depends(get_session),
):
    try:
        query = select(CompChapterActivity)
        if activity_group_id is not None:
            query = query.where(CompChapterActivity.activity_group_id == activity_group_id)
        if comp_chapter_id is not None:
            query = query.where(CompChapterActivity.comp_chapter_id == comp_chapter_id)
        elif sub_chapter_id is not None:
            query = query.where(CompChapterActivity.sub_chapter_id == sub_chapter_id)
        if status == "published":
            query = query.where(CompChapterActivity.is_published == True)
        elif status == "draft":
            query = query.where(CompChapterActivity.is_published == False)
        query = query.order_by(*sort_ordering(CompChapterActivity))
        result = await session.exec(query)
        return {"data": [a.dict() for a in result.all()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/activities/{activity_id}")
async def update_comp_activity(
    activity_id: int,
    activity_group_id: Optional[int] = Form(None),
    type: Optional[str] = Form(None),
    question_text: Optional[str] = Form(None),
    options: Optional[list[str]] = Form(None),
    correct_option_index: Optional[int] = Form(None),
    answer_text: Optional[str] = Form(None),
    answer_description: Optional[str] = Form(None),
    is_published: Optional[bool] = Form(None),
    sort_order: Optional[int] = Form(None),
    answer_image: UploadFile | None = File(None),
    session: AsyncSession = Depends(get_session),
):
    try:
        activity = await session.get(CompChapterActivity, activity_id)
        if not activity:
            raise HTTPException(status_code=404, detail="Activity not found")
        if activity_group_id is not None:
            activity.activity_group_id = activity_group_id
        if type is not None:
            activity.type = type
        if question_text is not None:
            activity.question_text = question_text.strip()
        if options is not None:
            activity.options = [opt.strip() for opt in options]
        if correct_option_index is not None:
            activity.correct_option_index = correct_option_index
        if answer_text is not None:
            activity.answer_text = answer_text.strip()
        if answer_description is not None:
            activity.answer_description = answer_description.strip() if answer_description else None
        if is_published is not None:
            activity.is_published = is_published
        if sort_order is not None:
            activity.sort_order = sort_order
        if answer_image:
            folder_id = activity.comp_chapter_id or activity.sub_chapter_id
            new_url = upload_to_do(answer_image, f"comp/chapters/{folder_id}/activities/answers")
            if activity.answer_image_url:
                delete_from_do(activity.answer_image_url)
            activity.answer_image_url = new_url
        session.add(activity)
        await session.commit()
        await session.refresh(activity)
        return {"message": "Activity updated", "data": activity.dict()}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/activities/{activity_id}")
async def delete_comp_activity(activity_id: int, session: AsyncSession = Depends(get_session)):
    try:
        activity = await session.get(CompChapterActivity, activity_id)
        if not activity:
            raise HTTPException(status_code=404, detail="Activity not found")
        if activity.answer_image_url:
            delete_from_do(activity.answer_image_url)
        await session.delete(activity)
        await session.commit()
        return {"message": "Activity deleted"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/activities/order")
async def reorder_comp_activities(
    payload: OrderUpdate,
    activity_group_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    _result = await session.exec(
        select(CompChapterActivity).where(
            CompChapterActivity.activity_group_id == activity_group_id,
            CompChapterActivity.id.in_(payload.ids),
        )
    )
    activities = _result.all()
    if len(activities) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid activity ids for group")
    a_map = {a.id: a for a in activities}
    for index, aid in enumerate(payload.ids, start=1):
        a_map[aid].sort_order = index
        session.add(a_map[aid])
    await session.commit()
    return {"message": "Activity order updated"}


# ============================================================
# Topics
# ============================================================

@router.get("/topics")
async def get_comp_topics(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(CompTopic)
    if comp_chapter_id is not None:
        query = query.where(CompTopic.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompTopic.sub_chapter_id == sub_chapter_id)
    query = query.order_by(*sort_ordering(CompTopic))
    result = await session.exec(query)
    return {"data": [t.dict() for t in result.all()]}


@router.post("/topics/ai/generate")
async def ai_generate_comp_topics(
    payload: CompTopicsRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    comp_chapter_id, sub_chapter_id = _resolve_fk(payload.comp_chapter_id, payload.sub_chapter_id)
    job = ActivityGenerationJob(
        job_type="comp_topics",
        status="pending",
        payload={"comp_chapter_id": comp_chapter_id, "sub_chapter_id": sub_chapter_id},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    background_tasks.add_task(enqueue_activity_job, job.id)
    return {"data": {"job_id": job.id, "status": job.status}}


@router.post("/topics/ai/save")
async def ai_save_comp_topics(
    payload: CompTopicsRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    comp_chapter_id, sub_chapter_id = _resolve_fk(payload.comp_chapter_id, payload.sub_chapter_id)
    job = ActivityGenerationJob(
        job_type="comp_topics_save",
        status="pending",
        payload={"comp_chapter_id": comp_chapter_id, "sub_chapter_id": sub_chapter_id},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    background_tasks.add_task(enqueue_activity_job, job.id)
    return {"data": {"job_id": job.id, "status": job.status}}


@router.post("/activities/ai/generate")
async def ai_generate_comp_activities(
    payload: GenerateCompActivitiesRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    comp_chapter_id, sub_chapter_id = _resolve_fk(payload.comp_chapter_id, payload.sub_chapter_id)
    mcq_count = min(max(payload.mcq_count, 0), 20)
    descriptive_count = min(max(payload.descriptive_count, 0), 20)
    if mcq_count + descriptive_count == 0:
        raise HTTPException(status_code=400, detail="Counts cannot both be zero")
    job = ActivityGenerationJob(
        job_type="comp_activities",
        status="pending",
        payload={
            "comp_chapter_id": comp_chapter_id,
            "sub_chapter_id": sub_chapter_id,
            "topic_titles": payload.topic_titles,
            "mcq_count": mcq_count,
            "descriptive_count": descriptive_count,
            "activity_group_id": payload.activity_group_id,
        },
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    background_tasks.add_task(enqueue_activity_job, job.id)
    return {"data": {"job_id": job.id, "status": job.status}}


@router.get("/ai/jobs/{job_id}")
async def get_comp_ai_job(job_id: int, session: AsyncSession = Depends(get_session)):
    job = await session.get(ActivityGenerationJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"data": job.dict()}
