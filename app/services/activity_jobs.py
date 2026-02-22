import asyncio
from typing import Any, Dict, List

from sqlalchemy import func
from sqlmodel import select

from app.models import ActivityGenerationJob, Chapter, ChapterActivity, Topic, Medium, ChapterArtifact, StudentTextbook, StudentNotes
from app.models.activities import ActivityGroup
from app.services.activity_ai import generate_activities, generate_topics, normalize_activity, generate_chapter_summary
from app.services.database import async_session_maker


def _clean_topics(topics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned = []
    for topic in topics:
        title = str(topic.get("title", "")).strip()
        summary = str(topic.get("summary", "")).strip()
        if title:
            cleaned.append({"title": title, "summary": summary})
    return cleaned


async def _run_topics_job(job: ActivityGenerationJob, session):
    from app.models import Subject

    chapter_id = job.payload.get("chapter_id")
    chapter = await session.get(Chapter, chapter_id)
    if not chapter:
        raise ValueError("Chapter not found")

    subject = await session.get(Subject, chapter.subject_id)
    medium = await session.get(Medium, subject.medium_id) if subject else None
    medium_name = medium.name if medium else ""

    topics = generate_topics(chapter_id, medium_name)
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
    medium = await session.get(Medium, subject.medium_id) if subject else None
    medium_name = medium.name if medium else ""

    topics = generate_topics(chapter_id, medium_name)
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
    medium = await session.get(Medium, subject.medium_id) if subject else None
    medium_name = medium.name if medium else ""

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

    raw = generate_activities(chapter_id, topic_titles, mcq_count, descriptive_count, medium_name)
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
            answer_description=item.get("answer_description"),
            answer_image_url=None,
            is_published=False,
            sort_order=next_order,
        )
        next_order += 1
        session.add(activity)
        await session.flush()
        created_ids.append(activity.id)

    job.result = {"created_ids": created_ids, "count": len(created_ids), "activity_group_id": activity_group_id}


async def _run_textbook_process_job(job: ActivityGenerationJob, session):
    """
    Runs embedding + chapter summary generation after a textbook upload.
    Writes the summary as a ChapterArtifact with artifact_type='chapter_summary'.
    If an artifact row already exists for this chapter (from a previous upload),
    it is updated in place so there is always at most one summary per chapter.
    """
    from app.api.v1.llm_resources import process_textbook_upload
    from sqlmodel import select

    chapter_id = job.payload.get("chapter_id")
    file_url = job.payload.get("file_url")
    textbook_id = job.payload.get("textbook_id")
    source_file = job.payload.get("source_file", "")

    chapter = await session.get(Chapter, chapter_id)
    if not chapter:
        raise ValueError("Chapter not found")

    from app.models import Subject
    subject = await session.get(Subject, chapter.subject_id)
    medium = await session.get(Medium, subject.medium_id) if subject else None
    medium_name = medium.name if medium else ""

    # 1. Mark artifact as processing (upsert)
    existing = await session.exec(
        select(ChapterArtifact).where(
            ChapterArtifact.chapter_id == chapter_id,
            ChapterArtifact.artifact_type == "chapter_summary",
        )
    )
    artifact = existing.first()
    if artifact is None:
        artifact = ChapterArtifact(
            chapter_id=chapter_id,
            artifact_type="chapter_summary",
            status="processing",
        )
        session.add(artifact)
    else:
        artifact.status = "processing"
        artifact.error = None
    await session.commit()
    await session.refresh(artifact)

    # 2. Embed the textbook (sync call — runs in thread via asyncio)
    import asyncio
    metadata = {
        "chapter_id": chapter_id,
        "source_file": source_file,
        "file_url": file_url,
        "textbook_id": textbook_id,
    }
    doc_count = await asyncio.get_event_loop().run_in_executor(
        None, process_textbook_upload, file_url, metadata
    )

    # 3. Generate and store summary
    summary = await asyncio.get_event_loop().run_in_executor(
        None, generate_chapter_summary, chapter_id, medium_name
    )

    artifact.content = summary
    artifact.status = "completed"
    session.add(artifact)
    await session.commit()

    # 4. Generate topics and save to DB
    from loguru import logger
    topics_raw = await asyncio.get_event_loop().run_in_executor(
        None, generate_topics, chapter_id, medium_name
    )
    cleaned_topics = _clean_topics(topics_raw)

    if not cleaned_topics:
        logger.warning(f"No topics generated for chapter_id={chapter_id}, skipping activity generation")
        job.result = {
            "documents_processed": doc_count,
            "artifact_id": artifact.id,
            "chapter_id": chapter_id,
            "topics_count": 0,
            "activity_groups_count": 0,
        }
        return

    # Get next sort_order for topics
    topic_order_result = await session.exec(
        select(func.max(Topic.sort_order)).where(Topic.chapter_id == chapter_id)
    )
    max_topic_order = topic_order_result.first()
    if isinstance(max_topic_order, tuple):
        max_topic_order = max_topic_order[0]
    next_topic_order = (max_topic_order or 0) + 1

    saved_topics = []
    for t in cleaned_topics:
        topic = Topic(
            title=t["title"],
            summary=t.get("summary"),
            chapter_id=chapter_id,
            sort_order=next_topic_order,
        )
        next_topic_order += 1
        session.add(topic)
        await session.flush()
        saved_topics.append(topic)

    logger.info(f"Saved {len(saved_topics)} topics for chapter_id={chapter_id}")

    # 5. Create all ActivityGroups first (sequential — need flushed IDs before LLM calls)
    group_order_result = await session.exec(
        select(func.max(ActivityGroup.sort_order)).where(
            ActivityGroup.chapter_id == chapter_id
        )
    )
    max_group_order = group_order_result.first()
    if isinstance(max_group_order, tuple):
        max_group_order = max_group_order[0]
    next_group_order = (max_group_order or 0) + 1

    MCQ_COUNT = 7
    DESCRIPTIVE_COUNT = 3

    topic_groups: list[tuple[Topic, ActivityGroup]] = []
    for topic in saved_topics:
        activity_group = ActivityGroup(
            name=topic.title[:255],
            chapter_id=chapter_id,
            sort_order=next_group_order,
        )
        session.add(activity_group)
        await session.flush()
        next_group_order += 1
        topic_groups.append((topic, activity_group))

    # 6. Generate all activities in one chunk-based call (tagged with topic field)
    all_topic_titles = [topic.title for topic, _ in topic_groups]
    logger.info(f"[chapter={chapter_id}] Starting chunk-based activity generation for {len(all_topic_titles)} topics")

    raw_activities = await asyncio.get_event_loop().run_in_executor(
        None,
        generate_activities,
        chapter_id,
        all_topic_titles,
        MCQ_COUNT,
        DESCRIPTIVE_COUNT,
        medium_name,
    )
    logger.info(f"[chapter={chapter_id}] Total raw tagged activities: {len(raw_activities)}")

    # Build lookup: normalized topic title → (topic, activity_group)
    title_to_group: dict[str, tuple[Topic, ActivityGroup]] = {
        topic.title.strip().lower(): (topic, group)
        for topic, group in topic_groups
    }

    # Distribute tagged activities into per-topic buckets (capped per topic)
    group_id_to_bucket: dict[int, list] = {group.id: [] for _, group in topic_groups}
    group_id_caps: dict[int, int] = {group.id: MCQ_COUNT + DESCRIPTIVE_COUNT for _, group in topic_groups}

    for item in raw_activities:
        raw_topic_tag = str(item.get("topic", "")).strip().lower()
        match = title_to_group.get(raw_topic_tag)
        if not match:
            # Fuzzy fallback: pick the topic title that is a substring match
            for key, val in title_to_group.items():
                if raw_topic_tag in key or key in raw_topic_tag:
                    match = val
                    break
        if not match:
            logger.warning(f"[chapter={chapter_id}] Activity topic tag '{item.get('topic')}' matched no known topic — skipping")
            continue
        _, group = match
        if len(group_id_to_bucket[group.id]) >= group_id_caps[group.id]:
            continue  # cap reached for this topic
        norm = normalize_activity(item)
        if norm:
            group_id_to_bucket[group.id].append(norm)
        else:
            logger.warning(f"[chapter={chapter_id}] normalize_activity rejected: {item}")

    # Save all activities (sequential DB writes)
    total_activities_created = 0
    group_id_to_topic = {group.id: topic for topic, group in topic_groups}

    for group_id, normalized in group_id_to_bucket.items():
        topic = group_id_to_topic[group_id]
        next_activity_order = 1
        for item in normalized:
            activity = ChapterActivity(
                activity_group_id=group_id,
                chapter_id=chapter_id,
                type=item["type"],
                question_text=item["question_text"],
                options=item.get("options"),
                correct_option_index=item.get("correct_option_index"),
                answer_text=item.get("answer_text"),
                answer_description=item.get("answer_description"),
                answer_image_url=None,
                is_published=True,
                sort_order=next_activity_order,
            )
            next_activity_order += 1
            session.add(activity)
        await session.flush()
        total_activities_created += len(normalized)
        logger.info(
            f"Created {len(normalized)} activities for topic '{topic.title}' "
            f"(group_id={group_id})"
        )

    job.result = {
        "documents_processed": doc_count,
        "artifact_id": artifact.id,
        "chapter_id": chapter_id,
        "topics_count": len(saved_topics),
        "activity_groups_count": len(saved_topics),
        "activities_count": total_activities_created,
    }


async def _run_audio_generation_job(job: ActivityGenerationJob, session):
    """
    Generate an audiobook MP3 from a student textbook or notes PDF.
    Updates audio_url and audio_status on the source record when done.
    """
    from app.services.audio_generation import generate_audio_from_pdf

    resource_type = job.payload.get("resource_type")  # "textbook" | "notes"
    resource_id = job.payload.get("resource_id")
    file_url = job.payload.get("file_url")
    chapter_id = job.payload.get("chapter_id")

    # Resolve the record to update
    if resource_type == "textbook":
        record = await session.get(StudentTextbook, resource_id)
    elif resource_type == "notes":
        record = await session.get(StudentNotes, resource_id)
    else:
        raise ValueError(f"Unknown resource_type for audio job: {resource_type}")

    if not record:
        raise ValueError(f"{resource_type} id={resource_id} not found")

    # Generate audio (sync, CPU+network bound — run in executor)
    audio_url = await asyncio.get_event_loop().run_in_executor(
        None,
        generate_audio_from_pdf,
        file_url,
        resource_type,
        resource_id,
        chapter_id,
    )

    record.audio_url = audio_url
    record.audio_status = "completed"
    session.add(record)

    job.result = {"audio_url": audio_url, "resource_type": resource_type, "resource_id": resource_id}


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
            elif job.job_type == "textbook_process":
                await _run_textbook_process_job(job, session)
            elif job.job_type == "audio_generation":
                await _run_audio_generation_job(job, session)
            else:
                raise ValueError("Unsupported job type")

            job.status = "completed"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)

            # For textbook_process jobs: mark the artifact as failed immediately
            # so the frontend doesn't stay stuck on "processing" in the same session.
            if job.job_type == "textbook_process":
                try:
                    chapter_id = (job.payload or {}).get("chapter_id")
                    if chapter_id:
                        art_result = await session.exec(
                            select(ChapterArtifact).where(
                                ChapterArtifact.chapter_id == chapter_id,
                                ChapterArtifact.artifact_type == "chapter_summary",
                                ChapterArtifact.status == "processing",
                            )
                        )
                        artifact = art_result.first()
                        if artifact:
                            artifact.status = "failed"
                            artifact.error = str(e)
                            session.add(artifact)
                except Exception:
                    pass

            # For audio jobs: mark the source record as failed so the frontend
            # can show the correct status instead of staying stuck on "processing".
            if job.job_type == "audio_generation":
                try:
                    resource_type = job.payload.get("resource_type")
                    resource_id = job.payload.get("resource_id")
                    if resource_type == "textbook":
                        record = await session.get(StudentTextbook, resource_id)
                    elif resource_type == "notes":
                        record = await session.get(StudentNotes, resource_id)
                    else:
                        record = None
                    if record:
                        record.audio_status = "failed"
                        session.add(record)
                except Exception:
                    pass

        session.add(job)
        await session.commit()


def enqueue_activity_job(job_id: int):
    """
    Runs the async job in a dedicated thread with its own event loop,
    avoiding the 'cannot run nested event loops' error when called from
    FastAPI's BackgroundTasks (which already runs in an event loop).
    """
    import threading

    def _run():
        asyncio.run(run_activity_job(job_id))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
