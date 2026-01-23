from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.admin.auth import get_current_user as get_current_admin
from app.models import Chapter, ActivityGenerationJob, Topic
from app.models.admin import Admin
from app.services.database import get_session
from app.services.activity_jobs import enqueue_activity_job

router = APIRouter()


class TopicCreateRequest(BaseModel):
    title: str
    summary: Optional[str] = None
    chapter_id: int


class TopicUpdateRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None


@router.get("/")
async def get_topics(
    chapter_id: int = Query(...),
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    query = select(Topic).where(Topic.chapter_id == chapter_id).order_by(
        Topic.sort_order.asc(), Topic.created_at.asc()
    )
    result = await session.exec(query)
    topics = result.all()
    return {"data": [t.dict() for t in topics]}


@router.post("/")
async def create_topic(
    payload: TopicCreateRequest,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    chapter = await session.get(Chapter, payload.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    # Get next sort order
    result = await session.exec(
        select(func.max(Topic.sort_order)).where(Topic.chapter_id == payload.chapter_id)
    )
    max_order = result.first()
    if isinstance(max_order, tuple):
        max_order = max_order[0]
    next_order = (max_order or 0) + 1

    topic = Topic(
        title=payload.title,
        summary=payload.summary,
        chapter_id=payload.chapter_id,
        sort_order=next_order,
    )
    session.add(topic)
    await session.commit()
    await session.refresh(topic)
    return {"data": topic.dict()}


@router.delete("/{topic_id}")
async def delete_topic(
    topic_id: int,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    topic = await session.get(Topic, topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    await session.delete(topic)
    await session.commit()
    return {"message": "Topic deleted"}


@router.post("/ai/generate")
async def ai_generate_topics(
    chapter_id: int = Query(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    chapter = await session.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    job = ActivityGenerationJob(
        job_type="topics_save",
        status="pending",
        payload={"chapter_id": chapter_id},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    background_tasks.add_task(enqueue_activity_job, job.id)
    return {"data": {"job_id": job.id, "status": job.status}}
