import asyncio
from typing import Any, Dict, List

from sqlalchemy import func
from sqlmodel import delete, select

from app.models import ActivityGenerationJob, Chapter, ChapterActivity, Topic, Medium, ChapterArtifact, StudentTextbook, StudentNotes
from app.models.activities import ActivityGroup
from app.models.comp_activities import CompTopic, CompActivityGroup, CompChapterActivity
from app.models.comp_artifacts import CompChapterArtifact
from app.models.comp_student_content import CompStudentTextbook, CompStudentNote
from app.services.activity_ai import (
    generate_activities, generate_topics, normalize_activity,
    generate_chapter_summary, generate_one_mark_questions, generate_important_questions,
    convert_docx_to_notes_markdown, convert_excel_to_notes_markdown,
    BOARD_TEXTBOOK_COLLECTION, BOARD_QA_COLLECTION,
    COMP_TEXTBOOK_COLLECTION, COMP_QA_COLLECTION,
)
from app.services.database import async_session_maker


def _clean_topics(topics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned = []
    for topic in topics:
        title = str(topic.get("title", "")).strip()
        summary = str(topic.get("summary", "")).strip()
        if title:
            cleaned.append({"title": title, "summary": summary})
    return cleaned


def _clean_topics_or_raise(topics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned = _clean_topics(topics)
    if not cleaned:
        raise ValueError("No valid topics generated")
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

    topics = generate_topics(chapter_id, medium_name, BOARD_TEXTBOOK_COLLECTION)
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

    topics = generate_topics(chapter_id, medium_name, BOARD_TEXTBOOK_COLLECTION)
    if not topics:
        raise ValueError("No chapter content found")

    cleaned = _clean_topics_or_raise(topics)

    # Replace only after we have a validated topic list to save.
    await session.exec(delete(Topic).where(Topic.chapter_id == chapter_id))
    await session.flush()

    created_ids = []
    for idx, t in enumerate(cleaned):
        topic = Topic(
            title=t["title"],
            summary=t.get("summary"),
            chapter_id=chapter_id,
            sort_order=idx + 1,
        )
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

    raw = generate_activities(chapter_id, topic_titles, mcq_count, descriptive_count, medium_name, BOARD_TEXTBOOK_COLLECTION, BOARD_QA_COLLECTION)
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

    # 1. Mark all 3 artifacts as processing (upsert)
    import asyncio
    from loguru import logger

    artifact_types = ["chapter_summary", "one_mark_questions", "important_questions"]
    artifacts: dict[str, ChapterArtifact] = {}
    for atype in artifact_types:
        existing = await session.exec(
            select(ChapterArtifact).where(
                ChapterArtifact.chapter_id == chapter_id,
                ChapterArtifact.artifact_type == atype,
            )
        )
        artifact = existing.first()
        if artifact is None:
            artifact = ChapterArtifact(
                chapter_id=chapter_id,
                artifact_type=atype,
                status="processing",
            )
            session.add(artifact)
        else:
            artifact.status = "processing"
            artifact.error = None
        artifacts[atype] = artifact
    await session.commit()
    for artifact in artifacts.values():
        await session.refresh(artifact)

    # 2. Embed the textbook (sync call — runs in thread via asyncio)
    metadata = {
        "chapter_id": chapter_id,
        "source_file": source_file,
        "file_url": file_url,
        "textbook_id": textbook_id,
    }
    doc_count = await asyncio.get_event_loop().run_in_executor(
        None, process_textbook_upload, file_url, metadata
    )

    # 3. Generate all 3 artifacts in parallel
    summary, one_mark, important_qs = await asyncio.gather(
        asyncio.get_event_loop().run_in_executor(None, generate_chapter_summary, chapter_id, medium_name, BOARD_TEXTBOOK_COLLECTION),
        asyncio.get_event_loop().run_in_executor(None, generate_one_mark_questions, chapter_id, medium_name, BOARD_TEXTBOOK_COLLECTION),
        asyncio.get_event_loop().run_in_executor(None, generate_important_questions, chapter_id, medium_name, BOARD_TEXTBOOK_COLLECTION),
    )

    artifacts["chapter_summary"].content = summary
    artifacts["chapter_summary"].status = "completed"
    artifacts["one_mark_questions"].content = one_mark
    artifacts["one_mark_questions"].status = "completed"
    artifacts["important_questions"].content = important_qs
    artifacts["important_questions"].status = "completed"
    for artifact in artifacts.values():
        session.add(artifact)
    await session.commit()

    job.result = {
        "documents_processed": doc_count,
        "artifact_ids": {atype: a.id for atype, a in artifacts.items()},
        "chapter_id": chapter_id,
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


async def _run_comp_textbook_process_job(job: ActivityGenerationJob, session):
    """
    Comp version of textbook processing: embed + generate artifacts using comp collections.
    Writes artifacts to comp_chapter_artifacts table.
    """

    comp_chapter_id = job.payload.get("comp_chapter_id")
    sub_chapter_id = job.payload.get("sub_chapter_id")
    file_url = job.payload.get("file_url")
    textbook_id = job.payload.get("textbook_id")
    source_file = job.payload.get("source_file", "")

    # Determine medium_name: walk up the hierarchy if chapter found
    medium_name = ""
    if comp_chapter_id:
        from app.models.competitive_hierarchy import CompChapter, CompSubject, Level, CompExamMedium
        chapter = await session.get(CompChapter, comp_chapter_id)
        if chapter:
            subject = await session.get(CompSubject, chapter.subject_id)
            if subject:
                level = await session.get(Level, subject.level_id)
                if level:
                    medium = await session.get(CompExamMedium, level.medium_id)
                    if medium:
                        medium_name = medium.name

    # Use chapter_id placeholder for embedding metadata
    metadata_chapter_id = comp_chapter_id or sub_chapter_id

    # 1. Mark artifacts as processing (upsert)
    artifact_types = ["chapter_summary", "one_mark_questions", "important_questions"]
    artifacts: dict[str, CompChapterArtifact] = {}
    for atype in artifact_types:
        filter_col = CompChapterArtifact.comp_chapter_id if comp_chapter_id else CompChapterArtifact.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        existing = await session.exec(
            select(CompChapterArtifact).where(
                filter_col == filter_val,
                CompChapterArtifact.artifact_type == atype,
            )
        )
        artifact = existing.first()
        if artifact is None:
            artifact = CompChapterArtifact(
                comp_chapter_id=comp_chapter_id,
                sub_chapter_id=sub_chapter_id,
                artifact_type=atype,
                status="processing",
            )
            session.add(artifact)
        else:
            artifact.status = "processing"
            artifact.error = None
        artifacts[atype] = artifact
    await session.commit()
    for artifact in artifacts.values():
        await session.refresh(artifact)

    # 2. Embed the textbook using comp collection
    doc_count = await asyncio.get_event_loop().run_in_executor(
        None, _async_embed_comp, file_url, source_file, metadata_chapter_id, textbook_id
    )

    # 3. Generate all 3 artifacts in parallel (using comp collection)
    summary, one_mark, important_qs = await asyncio.gather(
        asyncio.get_event_loop().run_in_executor(None, generate_chapter_summary, metadata_chapter_id, medium_name, COMP_TEXTBOOK_COLLECTION),
        asyncio.get_event_loop().run_in_executor(None, generate_one_mark_questions, metadata_chapter_id, medium_name, COMP_TEXTBOOK_COLLECTION),
        asyncio.get_event_loop().run_in_executor(None, generate_important_questions, metadata_chapter_id, medium_name, COMP_TEXTBOOK_COLLECTION),
    )

    artifacts["chapter_summary"].content = summary
    artifacts["chapter_summary"].status = "completed"
    artifacts["one_mark_questions"].content = one_mark
    artifacts["one_mark_questions"].status = "completed"
    artifacts["important_questions"].content = important_qs
    artifacts["important_questions"].status = "completed"
    for artifact in artifacts.values():
        session.add(artifact)
    await session.commit()

    job.result = {
        "documents_processed": doc_count,
        "artifact_ids": {atype: a.id for atype, a in artifacts.items()},
    }


def _async_embed_comp(file_url: str, source_file: str, chapter_id: int, textbook_id: int) -> int:
    """Sync wrapper for comp textbook embedding into comp_llm_textbooks collection."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_postgres import PGVector
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from app.utils.kannada_converter import convert_kannada_text
    from app.core.config import settings
    import uuid as _uuid
    from loguru import logger

    try:
        comp_store = PGVector(
            connection=settings.POSTGRES_URL,
            collection_name=COMP_TEXTBOOK_COLLECTION,
            embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", output_dimensionality=768),
        )
        loader = PyPDFLoader(file_url)
        pages = loader.load()
        for page in pages:
            page.page_content = convert_kannada_text(page.page_content)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "], length_function=len
        )
        documents = splitter.split_documents(pages)
        if not documents:
            return 0
        for idx, doc in enumerate(documents):
            doc.metadata.update({
                "chapter_id": str(chapter_id),
                "source_file": source_file,
                "file_url": file_url,
                "textbook_id": str(textbook_id),
                "content_type": "textbook",
                "chunk_index": idx,
            })
        comp_store.add_documents(
            documents,
            ids=[str(_uuid.uuid4()) for _ in documents],
        )
        logger.info(f"Comp embed: {len(documents)} chunks → {COMP_TEXTBOOK_COLLECTION!r} (chapter_id={chapter_id})")
        return len(documents)
    except Exception as e:
        logger.error(f"Comp embed error: {e}")
        raise


async def _run_comp_topics_job(job: ActivityGenerationJob, session):
    comp_chapter_id = job.payload.get("comp_chapter_id")
    sub_chapter_id = job.payload.get("sub_chapter_id")
    chapter_id = comp_chapter_id or sub_chapter_id

    medium_name = await _get_comp_medium_name(comp_chapter_id, session)

    topics = generate_topics(chapter_id, medium_name, COMP_TEXTBOOK_COLLECTION)
    if not topics:
        raise ValueError("No chapter content found")
    job.result = {"topics": _clean_topics(topics)}


async def _run_comp_topics_save_job(job: ActivityGenerationJob, session):
    comp_chapter_id = job.payload.get("comp_chapter_id")
    sub_chapter_id = job.payload.get("sub_chapter_id")
    chapter_id = comp_chapter_id or sub_chapter_id

    medium_name = await _get_comp_medium_name(comp_chapter_id, session)

    topics = generate_topics(chapter_id, medium_name, COMP_TEXTBOOK_COLLECTION)
    if not topics:
        raise ValueError("No chapter content found")

    cleaned = _clean_topics_or_raise(topics)

    # Replace only after we have a validated topic list to save.
    filter_col = CompTopic.comp_chapter_id if comp_chapter_id else CompTopic.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    await session.exec(delete(CompTopic).where(filter_col == filter_val))
    await session.flush()

    created_ids = []
    for idx, t in enumerate(cleaned):
        topic = CompTopic(
            title=t["title"],
            summary=t.get("summary"),
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            sort_order=idx + 1,
        )
        session.add(topic)
        await session.flush()
        created_ids.append(topic.id)

    job.result = {"topics": cleaned, "created_ids": created_ids, "count": len(created_ids)}


async def _run_comp_activities_job(job: ActivityGenerationJob, session):
    comp_chapter_id = job.payload.get("comp_chapter_id")
    sub_chapter_id = job.payload.get("sub_chapter_id")
    chapter_id = comp_chapter_id or sub_chapter_id
    topic_titles = job.payload.get("topic_titles", [])
    if not topic_titles:
        single = job.payload.get("topic_title")
        if single:
            topic_titles = [single]
    mcq_count = int(job.payload.get("mcq_count", 0))
    descriptive_count = int(job.payload.get("descriptive_count", 0))
    activity_group_id = job.payload.get("activity_group_id")

    medium_name = await _get_comp_medium_name(comp_chapter_id, session)

    if not activity_group_id:
        filter_col = CompActivityGroup.comp_chapter_id if comp_chapter_id else CompActivityGroup.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _group_result = await session.exec(select(func.max(CompActivityGroup.sort_order)).where(filter_col == filter_val))
        max_group_order = _group_result.first()
        if isinstance(max_group_order, tuple):
            max_group_order = max_group_order[0]
        group_name = ", ".join(topic_titles) if topic_titles else "Generated Activities"
        activity_group = CompActivityGroup(
            name=group_name[:255],
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            sort_order=(max_group_order or 0) + 1,
        )
        session.add(activity_group)
        await session.flush()
        activity_group_id = activity_group.id

    raw = generate_activities(chapter_id, topic_titles, mcq_count, descriptive_count, medium_name, COMP_TEXTBOOK_COLLECTION, COMP_QA_COLLECTION)
    normalized = [normalize_activity(item) for item in raw if normalize_activity(item)]
    normalized = normalized[:mcq_count + descriptive_count]

    if not normalized:
        raise ValueError("No valid activities generated")

    _result = await session.exec(
        select(func.max(CompChapterActivity.sort_order)).where(CompChapterActivity.activity_group_id == activity_group_id)
    )
    max_order = _result.first()
    if isinstance(max_order, tuple):
        max_order = max_order[0]
    next_order = (max_order or 0) + 1

    created_ids = []
    for item in normalized:
        activity = CompChapterActivity(
            activity_group_id=activity_group_id,
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
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


async def _run_comp_notes_convert_job(job: ActivityGenerationJob, session):
    import aiohttp

    note_id = job.payload.get("note_id")
    source = job.payload.get("source")
    file_url = job.payload.get("file_url")

    note = await session.get(CompStudentNote, note_id)
    if not note:
        raise ValueError(f"Note id={note_id} not found")

    async with aiohttp.ClientSession() as http:
        async with http.get(file_url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            content_bytes = await resp.read()

    if source == "docx_upload":
        markdown = await asyncio.get_event_loop().run_in_executor(
            None, convert_docx_to_notes_markdown, content_bytes, note_id
        )
    elif source == "excel_upload":
        markdown = await asyncio.get_event_loop().run_in_executor(
            None, convert_excel_to_notes_markdown, content_bytes
        )
    else:
        raise ValueError(f"Unknown source: {source}")

    word_count = len(markdown.split())
    read_time_min = max(1, round(word_count / 200))

    note.content = markdown
    note.content_status = "completed"
    note.word_count = word_count
    note.read_time_min = read_time_min
    session.add(note)

    job.result = {"note_id": note_id, "word_count": word_count, "read_time_min": read_time_min}


async def _run_comp_notes_audio_job(job: ActivityGenerationJob, session):
    from app.services.elevenlabs_tts import generate_notes_audio_with_sync

    note_id = job.payload.get("note_id")
    note = await session.get(CompStudentNote, note_id)
    if not note:
        raise ValueError(f"Note id={note_id} not found")
    if note.content_status != "completed" or not note.content:
        raise ValueError(f"Note id={note_id} has no completed content to generate audio from")

    audio_url, audio_sync_json = await asyncio.get_event_loop().run_in_executor(
        None, generate_notes_audio_with_sync, note.content, note.id, note.language
    )

    note.audio_url = audio_url
    note.audio_status = "completed"
    note.audio_sync_json = audio_sync_json
    session.add(note)

    job.result = {"note_id": note_id, "audio_url": audio_url}


async def _run_comp_audio_generation_job(job: ActivityGenerationJob, session):
    from app.services.audio_generation import generate_audio_from_pdf

    resource_type = job.payload.get("resource_type")  # "textbook" | "notes"
    resource_id = job.payload.get("resource_id")
    file_url = job.payload.get("file_url")
    comp_chapter_id = job.payload.get("comp_chapter_id")
    sub_chapter_id = job.payload.get("sub_chapter_id")

    if resource_type == "textbook":
        record = await session.get(CompStudentTextbook, resource_id)
    elif resource_type == "notes":
        record = await session.get(CompStudentNote, resource_id)
    else:
        raise ValueError(f"Unknown resource_type for comp audio job: {resource_type}")

    if not record:
        raise ValueError(f"Comp {resource_type} id={resource_id} not found")

    chapter_id = comp_chapter_id or sub_chapter_id
    audio_url = await asyncio.get_event_loop().run_in_executor(
        None, generate_audio_from_pdf, file_url, resource_type, resource_id, chapter_id
    )

    record.audio_url = audio_url
    record.audio_status = "completed"
    session.add(record)

    job.result = {"audio_url": audio_url, "resource_type": resource_type, "resource_id": resource_id}


async def _get_comp_medium_name(comp_chapter_id: int | None, session) -> str:
    """Walk up the competitive hierarchy to get medium name."""
    if not comp_chapter_id:
        return ""
    from app.models.competitive_hierarchy import CompChapter, CompSubject, Level, CompExamMedium
    chapter = await session.get(CompChapter, comp_chapter_id)
    if not chapter:
        return ""
    subject = await session.get(CompSubject, chapter.subject_id)
    if not subject:
        return ""
    level = await session.get(Level, subject.level_id)
    if not level:
        return ""
    medium = await session.get(CompExamMedium, level.medium_id)
    return medium.name if medium else ""


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
            elif job.job_type == "comp_textbook_process":
                await _run_comp_textbook_process_job(job, session)
            elif job.job_type == "comp_topics":
                await _run_comp_topics_job(job, session)
            elif job.job_type == "comp_topics_save":
                await _run_comp_topics_save_job(job, session)
            elif job.job_type == "comp_activities":
                await _run_comp_activities_job(job, session)
            elif job.job_type == "comp_audio_generation":
                await _run_comp_audio_generation_job(job, session)
            elif job.job_type == "comp_notes_convert":
                await _run_comp_notes_convert_job(job, session)
            elif job.job_type == "comp_notes_audio":
                await _run_comp_notes_audio_job(job, session)
            else:
                raise ValueError("Unsupported job type")

            job.status = "completed"
        except Exception as e:
            failed_payload = dict(job.payload or {})
            await session.rollback()
            job = await session.get(ActivityGenerationJob, job_id)
            if not job:
                return
            job.status = "failed"
            job.error = str(e)
            job.payload = failed_payload

            # For textbook_process jobs: mark the artifact as failed immediately
            # so the frontend doesn't stay stuck on "processing" in the same session.
            if job.job_type == "textbook_process":
                try:
                    chapter_id = (job.payload or {}).get("chapter_id")
                    if chapter_id:
                        for atype in ["chapter_summary", "one_mark_questions", "important_questions"]:
                            art_result = await session.exec(
                                select(ChapterArtifact).where(
                                    ChapterArtifact.chapter_id == chapter_id,
                                    ChapterArtifact.artifact_type == atype,
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

            if job.job_type == "comp_textbook_process":
                try:
                    comp_chapter_id = (job.payload or {}).get("comp_chapter_id")
                    sub_chapter_id = (job.payload or {}).get("sub_chapter_id")
                    filter_col = CompChapterArtifact.comp_chapter_id if comp_chapter_id else CompChapterArtifact.sub_chapter_id
                    filter_val = comp_chapter_id or sub_chapter_id
                    if filter_val:
                        for atype in ["chapter_summary", "one_mark_questions", "important_questions"]:
                            art_result = await session.exec(
                                select(CompChapterArtifact).where(
                                    filter_col == filter_val,
                                    CompChapterArtifact.artifact_type == atype,
                                    CompChapterArtifact.status == "processing",
                                )
                            )
                            artifact = art_result.first()
                            if artifact:
                                artifact.status = "failed"
                                artifact.error = str(e)
                                session.add(artifact)
                except Exception:
                    pass

            if job.job_type == "comp_audio_generation":
                try:
                    resource_type = (job.payload or {}).get("resource_type")
                    resource_id = (job.payload or {}).get("resource_id")
                    if resource_type == "textbook":
                        record = await session.get(CompStudentTextbook, resource_id)
                    elif resource_type == "notes":
                        record = await session.get(CompStudentNote, resource_id)
                    else:
                        record = None
                    if record:
                        record.audio_status = "failed"
                        session.add(record)
                except Exception:
                    pass

            if job.job_type == "comp_notes_convert":
                try:
                    note_id = (job.payload or {}).get("note_id")
                    if note_id:
                        note = await session.get(CompStudentNote, note_id)
                        if note:
                            note.content_status = "failed"
                            session.add(note)
                except Exception:
                    pass

            if job.job_type == "comp_notes_audio":
                try:
                    note_id = (job.payload or {}).get("note_id")
                    if note_id:
                        note = await session.get(CompStudentNote, note_id)
                        if note:
                            note.audio_status = "failed"
                            session.add(note)
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
