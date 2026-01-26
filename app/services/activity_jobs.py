import asyncio
from typing import Any, Dict, List

from sqlalchemy import func
from sqlmodel import select

from app.models import ActivityGenerationJob, Chapter, ChapterActivity, Topic
from app.services.activity_ai import generate_activities, generate_topics, normalize_activity
from app.services.database import async_session_maker


def _clean_topics(topics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned = []
    for topic in topics:
        title = str(topic.get("title", "")).strip()
        summary = str(topic.get("summary", "")).strip()
        if title:
            cleaned.append({"title": title, "summary": summary})
    return cleaned[:6]


async def _run_topics_job(job: ActivityGenerationJob, session):
    from app.models import Subject

    chapter_id = job.payload.get("chapter_id")
    chapter = await session.get(Chapter, chapter_id)
    if not chapter:
        raise ValueError("Chapter not found")

    subject = await session.get(Subject, chapter.subject_id)
    subject_name = subject.name if subject else ""

    topics = generate_topics(chapter_id, subject_name)
    if not topics:
        raise ValueError("No chapter content found")

    job.result = {"topics": _clean_topics(topics)}


async def _run_topics_save_job(job: ActivityGenerationJob, session):
    from app.models import Subject

    chapter_id = job.payload.get("chapter_id")
    chapter = await session.get(Chapter, chapter_id)
    if not chapter:
        raise ValueError("Chapter not found")

    subject = await session.get(Subject, chapter.subject_id)
    subject_name = subject.name if subject else ""

    topics = generate_topics(chapter_id, subject_name)
    if not topics:
        raise ValueError("No chapter content found")

    cleaned = _clean_topics(topics)

    # Get max sort order for existing topics
    result = await session.exec(
        select(func.max(Topic.sort_order)).where(Topic.chapter_id == chapter_id)
    )
    max_order = result.first()
    if isinstance(max_order, tuple):
        max_order = max_order[0]
    next_order = (max_order or 0) + 1

    created_ids = []
    for t in cleaned:
        topic = Topic(
            title=t["title"],
            summary=t.get("summary"),
            chapter_id=chapter_id,
            sort_order=next_order,
        )
        next_order += 1
        session.add(topic)
        await session.flush()
        created_ids.append(topic.id)

    job.result = {"topics": cleaned, "created_ids": created_ids, "count": len(created_ids)}


async def _run_activities_job(job: ActivityGenerationJob, session):
    from app.models import ActivityGroup, Subject

    chapter_id = job.payload.get("chapter_id")
    topic_titles = job.payload.get("topic_titles", [])
    # Backward compat: support old single topic_title field
    if not topic_titles:
        single = job.payload.get("topic_title")
        if single:
            topic_titles = [single]
    mcq_count = int(job.payload.get("mcq_count", 0))
    descriptive_count = int(job.payload.get("descriptive_count", 0))
    activity_group_id = job.payload.get("activity_group_id")

    chapter = await session.get(Chapter, chapter_id)
    if not chapter:
        raise ValueError("Chapter not found")

    subject = await session.get(Subject, chapter.subject_id)
    subject_name = subject.name if subject else ""

    # If no activity_group_id provided, create a new group for these activities
    if not activity_group_id:
        # Find the max sort_order for activity groups in this chapter
        _group_result = await session.exec(
            select(func.max(ActivityGroup.sort_order)).where(
                ActivityGroup.chapter_id == chapter_id
            )
        )
        max_group_order = _group_result.first()
        if isinstance(max_group_order, tuple):
            max_group_order = max_group_order[0]
        next_group_order = (max_group_order or 0) + 1

        # Create new activity group
        group_name = ", ".join(topic_titles) if topic_titles else "Generated Activities"
        activity_group = ActivityGroup(
            name=group_name[:255],
            chapter_id=chapter_id,
            sort_order=next_group_order,
        )
        session.add(activity_group)
        await session.flush()
        activity_group_id = activity_group.id

    raw = generate_activities(chapter_id, topic_titles, mcq_count, descriptive_count, subject_name)
    normalized = []
    for item in raw:
        normalized_item = normalize_activity(item)
        if normalized_item:
            normalized.append(normalized_item)

    if not normalized:
        raise ValueError("No valid activities generated")

    expected_total = mcq_count + descriptive_count
    normalized = normalized[:expected_total]

    _result = await session.exec(
        select(func.max(ChapterActivity.sort_order)).where(
            ChapterActivity.activity_group_id == activity_group_id
        )
    )
    max_order = _result.first()
    if isinstance(max_order, tuple):
        max_order = max_order[0]
    next_order = (max_order or 0) + 1

    created_ids = []
    for item in normalized:
        activity = ChapterActivity(
            activity_group_id=activity_group_id,
            chapter_id=chapter_id,
            type=item["type"],
            question_text=item["question_text"],
            options=item.get("options"),
            correct_option_index=item.get("correct_option_index"),
            answer_text=item.get("answer_text"),
            answer_image_url=None,
            is_published=False,
            sort_order=next_order,
        )
        next_order += 1
        session.add(activity)
        await session.flush()
        created_ids.append(activity.id)

    job.result = {"created_ids": created_ids, "count": len(created_ids), "activity_group_id": activity_group_id}


async def run_activity_job(job_id: int):
    async with async_session_maker() as session:
        job = await session.get(ActivityGenerationJob, job_id)
        if not job:
            return

        job.status = "running"
        session.add(job)
        await session.commit()
        await session.refresh(job)

        try:
            if job.job_type == "topics":
                await _run_topics_job(job, session)
            elif job.job_type == "topics_save":
                await _run_topics_save_job(job, session)
            elif job.job_type == "activities":
                await _run_activities_job(job, session)
            else:
                raise ValueError("Unsupported job type")

            job.status = "completed"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)

        session.add(job)
        await session.commit()


def enqueue_activity_job(job_id: int):
    asyncio.run(run_activity_job(job_id))
