from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.auth import get_current_user
from app.api.v1.admin.auth import get_current_user as get_current_admin
from app.models import (
    Chapter,
    ChapterActivity,
    ActivityPlaySession,
    ActivityAnswer,
    ActivityGenerationJob,
    ActivityGroup,
)
from app.models.user import User
from app.models.admin import Admin
from app.services.activity_jobs import enqueue_activity_job
from app.services.database import get_session
from app.utils.files import upload_to_do, delete_from_do

router = APIRouter()


class OrderUpdate(BaseModel):
    ids: list[int]


class SessionCreate(BaseModel):
    chapter_id: int


class AnswerSubmit(BaseModel):
    activity_id: int
    selected_option_index: Optional[int] = None
    submitted_answer_text: Optional[str] = None


class TopicsRequest(BaseModel):
    chapter_id: int


class GenerateActivitiesRequest(BaseModel):
    chapter_id: int
    topic_titles: list[str] = []
    mcq_count: int = 5
    descriptive_count: int = 5
    activity_group_id: Optional[int] = None


class PublishRequest(BaseModel):
    ids: list[int]
    is_published: bool = True


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def validate_activity_payload(
    activity_type: str,
    options: Optional[list[str]],
    correct_option_index: Optional[int],
    answer_text: Optional[str],
):
    if activity_type not in {"mcq", "descriptive"}:
        raise HTTPException(status_code=400, detail="Invalid activity type")

    if activity_type == "mcq":
        if not options or len(options) != 4:
            raise HTTPException(
                status_code=400, detail="MCQ must have exactly 4 options"
            )
        if any(not option.strip() for option in options):
            raise HTTPException(
                status_code=400, detail="MCQ options cannot be empty"
            )
        if correct_option_index not in {1, 2, 3, 4}:
            raise HTTPException(
                status_code=400,
                detail="MCQ must have a correct_option_index between 1 and 4",
            )
    else:
        if not answer_text or not answer_text.strip():
            raise HTTPException(
                status_code=400, detail="Descriptive must have an answer_text"
            )


@router.post("/")
async def create_activity(
    activity_group_id: int = Form(...),
    chapter_id: int = Form(...),
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
        chapter = await session.get(Chapter, chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        activity_group = await session.get(ActivityGroup, activity_group_id)
        if not activity_group:
            raise HTTPException(status_code=404, detail="Activity group not found")
        if activity_group.chapter_id != chapter_id:
            raise HTTPException(status_code=400, detail="Activity group does not belong to this chapter")

        cleaned_options = [opt.strip() for opt in options] if options else None
        validate_activity_payload(type, cleaned_options, correct_option_index, answer_text)

        if sort_order is None:
            _result = await session.exec(
                select(func.max(ChapterActivity.sort_order)).where(
                    ChapterActivity.activity_group_id == activity_group_id
                )
            )
            max_order = _result.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            sort_order = (max_order or 0) + 1

        answer_image_url = None
        if answer_image:
            do_path = f"chapters/{chapter_id}/activities/answers"
            answer_image_url = upload_to_do(answer_image, do_path)

        activity = ChapterActivity(
            activity_group_id=activity_group_id,
            chapter_id=chapter_id,
            type=type,
            question_text=question_text.strip(),
            options=cleaned_options if type == "mcq" else None,
            correct_option_index=correct_option_index if type == "mcq" else None,
            answer_text=answer_text.strip() if type == "descriptive" else None,
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


@router.get("/")
async def get_activities(
    activity_group_id: Optional[int] = None,
    chapter_id: Optional[int] = None,
    status: str = Query("published"),
    session: AsyncSession = Depends(get_session),
):
    try:
        query = select(ChapterActivity)
        if activity_group_id is not None:
            query = query.where(ChapterActivity.activity_group_id == activity_group_id)
        if chapter_id is not None:
            query = query.where(ChapterActivity.chapter_id == chapter_id)
        if status == "published":
            query = query.where(ChapterActivity.is_published == True)
        elif status == "draft":
            query = query.where(ChapterActivity.is_published == False)
        elif status != "all":
            raise HTTPException(status_code=400, detail="Invalid status filter")
        query = query.order_by(*sort_ordering(ChapterActivity))
        _result = await session.exec(query)
        activities = _result.all()
        return {"data": [activity.dict() for activity in activities]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{activity_id}")
async def get_activity(
    activity_id: int, session: AsyncSession = Depends(get_session)
):
    activity = await session.get(ChapterActivity, activity_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    return {"data": activity.dict()}


@router.put("/{activity_id}")
async def update_activity(
    activity_id: int,
    activity_group_id: Optional[int] = Form(None),
    chapter_id: Optional[int] = Form(None),
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
        activity = await session.get(ChapterActivity, activity_id)
        if not activity:
            raise HTTPException(status_code=404, detail="Activity not found")

        if chapter_id is not None:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            activity.chapter_id = chapter_id

        if activity_group_id is not None:
            activity_group = await session.get(ActivityGroup, activity_group_id)
            if not activity_group:
                raise HTTPException(status_code=404, detail="Activity group not found")
            # Ensure group belongs to the same chapter
            effective_chapter_id = chapter_id if chapter_id is not None else activity.chapter_id
            if activity_group.chapter_id != effective_chapter_id:
                raise HTTPException(status_code=400, detail="Activity group does not belong to this chapter")
            activity.activity_group_id = activity_group_id

        effective_type = type or activity.type
        cleaned_options = [opt.strip() for opt in options] if options else activity.options
        effective_answer_text = answer_text if answer_text is not None else activity.answer_text
        effective_correct_index = (
            correct_option_index
            if correct_option_index is not None
            else activity.correct_option_index
        )

        validate_activity_payload(
            effective_type, cleaned_options, effective_correct_index, effective_answer_text
        )

        if type is not None:
            activity.type = type

        if question_text is not None:
            activity.question_text = question_text.strip()

        if answer_description is not None:
            activity.answer_description = answer_description.strip() if answer_description else None

        if sort_order is not None:
            activity.sort_order = sort_order

        if is_published is not None:
            activity.is_published = is_published

        if effective_type == "mcq":
            if options is not None:
                activity.options = cleaned_options
            if correct_option_index is not None:
                activity.correct_option_index = correct_option_index
            activity.answer_text = None
        else:
            activity.options = None
            activity.correct_option_index = None
            if answer_text is not None:
                activity.answer_text = answer_text.strip()

        if answer_image:
            do_path = f"chapters/{activity.chapter_id}/activities/answers"
            new_image_url = upload_to_do(answer_image, do_path)
            if activity.answer_image_url:
                delete_from_do(activity.answer_image_url)
            activity.answer_image_url = new_image_url

        session.add(activity)
        await session.commit()
        await session.refresh(activity)
        return {"message": "Activity updated", "data": activity.dict()}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{activity_id}")
async def delete_activity(
    activity_id: int, session: AsyncSession = Depends(get_session)
):
    try:
        activity = await session.get(ChapterActivity, activity_id)
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


@router.put("/order")
async def reorder_activities(
    payload: OrderUpdate,
    activity_group_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(ChapterActivity).where(
            ChapterActivity.activity_group_id == activity_group_id,
            ChapterActivity.id.in_(payload.ids),
        )
    )
    activities = _result.all()
    if len(activities) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid activity ids for group")

    activity_map = {activity.id: activity for activity in activities}
    for index, activity_id in enumerate(payload.ids, start=1):
        activity_map[activity_id].sort_order = index
        session.add(activity_map[activity_id])

    await session.commit()
    return {"message": "Activity order updated"}


@router.get("/progress")
async def get_activity_progress(
    chapter_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    _result = await session.exec(
        select(ChapterActivity)
        .where(ChapterActivity.chapter_id == chapter_id)
        .where(ChapterActivity.is_published == True)
        .order_by(*sort_ordering(ChapterActivity))
    )
    activities = _result.all()
    if not activities:
        return {"data": []}

    activity_ids = [activity.id for activity in activities]
    answered_query = (
        select(ActivityAnswer.activity_id)
        .join(ActivityPlaySession)
        .where(
            ActivityPlaySession.user_id == current_user.id,
            ActivityAnswer.activity_id.in_(activity_ids),
        )
        .distinct()
    )
    answered_result = await session.exec(answered_query)
    answered_ids = {
        row[0] if isinstance(row, tuple) else row for row in answered_result.all()
    }

    return {
        "data": [
            {**activity.dict(), "completed": activity.id in answered_ids}
            for activity in activities
        ]
    }


@router.post("/sessions")
async def create_session(
    payload: SessionCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    chapter = await session.get(Chapter, payload.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    _count_result = await session.exec(
        select(func.count()).where(
            ChapterActivity.chapter_id == payload.chapter_id,
            ChapterActivity.is_published == True,
        )
    )
    total_questions = _count_result.first()
    if isinstance(total_questions, tuple):
        total_questions = total_questions[0]
    if total_questions == 0:
        raise HTTPException(status_code=404, detail="No activities found for chapter")

    play_session = ActivityPlaySession(
        user_id=current_user.id,
        chapter_id=payload.chapter_id,
        total_questions=total_questions,
    )
    session.add(play_session)
    await session.commit()
    await session.refresh(play_session)

    _result = await session.exec(
        select(ChapterActivity)
        .where(
            ChapterActivity.chapter_id == payload.chapter_id,
            ChapterActivity.is_published == True,
        )
        .order_by(*sort_ordering(ChapterActivity))
        .limit(1)
    )
    first_activity = _result.first()

    return {
        "data": {
            "session": play_session.dict(),
            "next_activity": first_activity.dict() if first_activity else None,
        }
    }


@router.post("/sessions/{session_id}/answers")
async def submit_answer(
    session_id: int,
    payload: AnswerSubmit,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    play_session = await session.get(ActivityPlaySession, session_id)
    if not play_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if play_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")
    if play_session.status == "completed":
        raise HTTPException(status_code=400, detail="Session already completed")

    activity = await session.get(ChapterActivity, payload.activity_id)
    if not activity or activity.chapter_id != play_session.chapter_id:
        raise HTTPException(status_code=404, detail="Activity not found for chapter")
    if not activity.is_published:
        raise HTTPException(status_code=400, detail="Activity not published")

    _existing_result = await session.exec(
        select(ActivityAnswer).where(
            ActivityAnswer.session_id == session_id,
            ActivityAnswer.activity_id == payload.activity_id,
        )
    )
    if _existing_result.first():
        raise HTTPException(status_code=400, detail="Answer already submitted")

    if activity.type == "mcq":
        if payload.selected_option_index not in {1, 2, 3, 4}:
            raise HTTPException(
                status_code=400, detail="selected_option_index must be 1-4"
            )
        is_correct = payload.selected_option_index == activity.correct_option_index
        score = 1 if is_correct else 0
    else:
        submitted = normalize_text(payload.submitted_answer_text)
        expected = normalize_text(activity.answer_text)
        if not submitted:
            raise HTTPException(status_code=400, detail="Answer text is required")
        is_correct = submitted == expected if expected else False
        score = 1 if is_correct else 0

    answer = ActivityAnswer(
        session_id=session_id,
        activity_id=payload.activity_id,
        selected_option_index=payload.selected_option_index,
        submitted_answer_text=payload.submitted_answer_text,
        is_correct=is_correct,
        score=score,
    )
    session.add(answer)

    play_session.correct_count += 1 if is_correct else 0
    play_session.score += score
    session.add(play_session)
    await session.commit()
    await session.refresh(play_session)

    _answered_count_result = await session.exec(
        select(func.count(ActivityAnswer.id)).where(
            ActivityAnswer.session_id == session_id
        )
    )
    answered_count = _answered_count_result.first()
    if isinstance(answered_count, tuple):
        answered_count = answered_count[0]

    completed = answered_count >= play_session.total_questions
    if completed:
        play_session.status = "completed"
        play_session.completed_at = datetime.now(UTC)
        session.add(play_session)
        await session.commit()
        await session.refresh(play_session)

    _answered_ids_result = await session.exec(
        select(ActivityAnswer.activity_id).where(ActivityAnswer.session_id == session_id)
    )
    answered_ids = [
        row[0] if isinstance(row, tuple) else row
        for row in _answered_ids_result.all()
    ]

    next_query = select(ChapterActivity).where(
        ChapterActivity.chapter_id == play_session.chapter_id,
        ChapterActivity.is_published == True,
    )
    if answered_ids:
        next_query = next_query.where(~ChapterActivity.id.in_(answered_ids))
    next_query = next_query.order_by(*sort_ordering(ChapterActivity)).limit(1)
    _next_result = await session.exec(next_query)
    next_activity = _next_result.first()

    correct_answer = None
    if activity.type == "mcq":
        correct_answer = {
            "correct_option_index": activity.correct_option_index,
            "correct_option_text": (
                activity.options[activity.correct_option_index - 1]
                if activity.options and activity.correct_option_index
                else None
            ),
            "answer_description": activity.answer_description,
        }
    else:
        correct_answer = {
            "answer_text": activity.answer_text,
            "answer_description": activity.answer_description,
        }

    return {
        "data": {
            "is_correct": is_correct,
            "score": score,
            "correct_answer": correct_answer,
            "answer_image_url": activity.answer_image_url,
            "next_activity": next_activity.dict() if next_activity else None,
            "completed": completed,
            "session": play_session.dict(),
        }
    }


@router.get("/sessions/{session_id}/report")
async def get_session_report(
    session_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    play_session = await session.get(ActivityPlaySession, session_id)
    if not play_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if play_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")

    _result = await session.exec(
        select(ActivityAnswer, ChapterActivity)
        .join(ChapterActivity, ChapterActivity.id == ActivityAnswer.activity_id)
        .where(ActivityAnswer.session_id == session_id)
        .order_by(*sort_ordering(ChapterActivity))
    )
    answers = []
    for answer, activity in _result.all():
        if activity.type == "mcq":
            correct = {
                "correct_option_index": activity.correct_option_index,
                "correct_option_text": (
                    activity.options[activity.correct_option_index - 1]
                    if activity.options and activity.correct_option_index
                    else None
                ),
                "answer_description": activity.answer_description,
            }
            submitted = {"selected_option_index": answer.selected_option_index}
        else:
            correct = {
                "answer_text": activity.answer_text,
                "answer_description": activity.answer_description,
            }
            submitted = {"submitted_answer_text": answer.submitted_answer_text}

        answers.append(
            {
                "activity_id": activity.id,
                "type": activity.type,
                "question_text": activity.question_text,
                "options": activity.options,
                "answer_image_url": activity.answer_image_url,
                "submitted": submitted,
                "correct": correct,
                "is_correct": answer.is_correct,
                "score": answer.score,
            }
        )

    return {
        "data": {
            "session": play_session.dict(),
            "answers": answers,
        }
    }


@router.post("/publish")
async def publish_activities(
    payload: PublishRequest,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    if not payload.ids:
        raise HTTPException(status_code=400, detail="No ids provided")

    _result = await session.exec(
        select(ChapterActivity).where(ChapterActivity.id.in_(payload.ids))
    )
    activities = _result.all()
    if len(activities) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid activity ids")

    for activity in activities:
        activity.is_published = payload.is_published
        session.add(activity)

    await session.commit()
    return {"message": "Activities updated"}


@router.post("/ai/topics")
async def ai_generate_topics(
    payload: TopicsRequest,
    background_tasks: BackgroundTasks,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    chapter = await session.get(Chapter, payload.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    job = ActivityGenerationJob(
        job_type="topics",
        status="pending",
        payload={"chapter_id": payload.chapter_id},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    background_tasks.add_task(enqueue_activity_job, job.id)
    return {"data": {"job_id": job.id, "status": job.status}}


@router.post("/ai/generate")
async def ai_generate_activities(
    payload: GenerateActivitiesRequest,
    background_tasks: BackgroundTasks,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    chapter = await session.get(Chapter, payload.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    mcq_count = min(max(payload.mcq_count, 0), 20)
    descriptive_count = min(max(payload.descriptive_count, 0), 20)
    if mcq_count + descriptive_count == 0:
        raise HTTPException(status_code=400, detail="Counts cannot both be zero")

    job = ActivityGenerationJob(
        job_type="activities",
        status="pending",
        payload={
            "chapter_id": payload.chapter_id,
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
async def get_ai_job(
    job_id: int,
    _: Admin = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    job = await session.get(ActivityGenerationJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"data": job.dict()}
