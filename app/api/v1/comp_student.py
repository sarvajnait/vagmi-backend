from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.auth import get_current_user
from app.models.comp_activities import (
    CompActivityPlaySession, CompChapterActivity, CompActivityAnswer,
)
from app.models.comp_streak import UserStreak
from app.models.comp_study_time import StudyTimeLog
from app.models.comp_wrong_answers import WrongAnswerEntry
from app.models.user_notifications import UserNotification
from app.models.user import User
from app.models.user_comp_profile import UserCompProfile
from app.models.competitive_hierarchy import ExamCategory, Exam, CompExamMedium, Level
from app.models.comp_student_content import CompPreviousYearPaper
from app.services.comp_performance_service import (
    get_chapter_detail, get_subject_chapters, get_performance_dashboard,
)
from app.services.comp_streak_service import record_activity, get_streak
from app.services.comp_wrong_answer_service import (
    get_wrong_answers, create_retry_session,
)
from app.services.database import get_session

router = APIRouter()


# ============================================================
# Chapter List & Detail
# ============================================================

@router.get("/subjects/{subject_id}/chapters")
async def student_chapter_list(
    subject_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await get_subject_chapters(current_user.id, subject_id, session)
    if result is None:
        raise HTTPException(status_code=404, detail="Subject not found")
    return {"data": result}


@router.get("/chapters/{chapter_id}")
async def student_chapter_detail(
    chapter_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await get_chapter_detail(current_user.id, chapter_id, session)
    if result is None:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return {"data": result}


# ============================================================
# Streak
# ============================================================

@router.post("/streak/activity")
async def record_streak_activity(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await record_activity(current_user.id, session)
    return {"data": result}


@router.get("/streak")
async def get_streak_data(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await get_streak(current_user.id, session)
    return {"data": result}


# ============================================================
# Study Time
# ============================================================

class StudyTimePayload(BaseModel):
    duration_seconds: int
    date: Optional[date] = None


@router.post("/study-time")
async def log_study_time(
    payload: StudyTimePayload,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if payload.duration_seconds <= 0:
        raise HTTPException(status_code=400, detail="duration_seconds must be positive")

    logged_date = payload.date or date.today()

    existing_result = await session.exec(
        select(StudyTimeLog).where(
            StudyTimeLog.user_id == current_user.id,
            StudyTimeLog.logged_date == logged_date,
        )
    )
    log = existing_result.first()
    if log:
        log.duration_seconds += payload.duration_seconds
        session.add(log)
    else:
        log = StudyTimeLog(
            user_id=current_user.id,
            logged_date=logged_date,
            duration_seconds=payload.duration_seconds,
        )
        session.add(log)
    await session.commit()
    await session.refresh(log)

    total_result = await session.exec(
        select(func.coalesce(func.sum(StudyTimeLog.duration_seconds), 0)).where(
            StudyTimeLog.user_id == current_user.id
        )
    )
    total = total_result.first() or 0
    if isinstance(total, tuple):
        total = total[0]

    return {"data": {"total_today_seconds": log.duration_seconds, "total_all_time_seconds": int(total)}}


# ============================================================
# Wrong Answer Notebook
# ============================================================

class RetrySessionPayload(BaseModel):
    comp_chapter_id: Optional[int] = None
    activity_ids: Optional[list[int]] = None


@router.get("/wrong-answers")
async def list_wrong_answers(
    comp_chapter_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await get_wrong_answers(current_user.id, session, comp_chapter_id=comp_chapter_id)
    return {"data": result}


@router.post("/wrong-answers/retry-session")
async def start_wrong_answer_retry(
    payload: RetrySessionPayload,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if payload.comp_chapter_id is None and not payload.activity_ids:
        raise HTTPException(status_code=400, detail="Provide comp_chapter_id or activity_ids")
    try:
        session_id, activity_ids = await create_retry_session(
            user_id=current_user.id,
            db=session,
            comp_chapter_id=payload.comp_chapter_id,
            activity_ids=payload.activity_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch the actual activities so frontend can display them immediately
    activities_result = await session.exec(
        select(CompChapterActivity).where(CompChapterActivity.id.in_(activity_ids))
    )
    activities = [a.dict() for a in activities_result.all()]

    return {"data": {"session_id": session_id, "activities": activities, "total_questions": len(activity_ids)}}


# ============================================================
# Performance Dashboard
# ============================================================

@router.get("/performance")
async def performance_dashboard(
    level_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await get_performance_dashboard(current_user.id, session, level_id=level_id)
    return {"data": result}


# ============================================================
# Profile Stats
# ============================================================

@router.get("/profile-stats")
async def profile_stats(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    streak_result = await session.exec(
        select(UserStreak).where(UserStreak.user_id == current_user.id)
    )
    streak = streak_result.first()
    current_streak = streak.current_streak if streak else 0

    # L-6: use distinct activity counts to avoid inflating numbers on replay
    stats_result = await session.exec(
        select(
            func.count(func.distinct(CompActivityAnswer.activity_id)),
            func.count(func.distinct(
                case((CompActivityAnswer.is_correct == True, CompActivityAnswer.activity_id), else_=None)
            )),
        )
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == current_user.id,
            CompActivityPlaySession.status == "completed",
        )
    )
    _stats = stats_result.first()
    total_answered = _stats[0] if _stats else 0
    total_correct = _stats[1] if _stats else 0
    accuracy_pct = round(total_correct / total_answered * 100) if total_answered > 0 else 0

    wrong_count_result = await session.exec(
        select(func.count(WrongAnswerEntry.id)).where(
            WrongAnswerEntry.user_id == current_user.id,
            WrongAnswerEntry.is_mastered == False,
        )
    )
    wrong_pending = wrong_count_result.first() or 0
    if isinstance(wrong_pending, tuple):
        wrong_pending = wrong_pending[0]

    return {
        "data": {
            "current_streak": current_streak,
            "questions_done": int(total_answered),
            "accuracy_pct": accuracy_pct,
            "wrong_answers_pending_revision": int(wrong_pending),
        }
    }


# ============================================================
# Notification Inbox
# ============================================================

class MarkReadPayload(BaseModel):
    ids: list[int]


def _group_notifications(notifications: list) -> list[dict]:
    from datetime import timedelta
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    groups: dict[str, list] = {"Today": [], "Yesterday": [], "Earlier": []}
    for n in notifications:
        created = n.created_at
        if created:
            # created_at is timezone-aware from BaseModel server_default
            d = created.date() if hasattr(created, "date") else created
            if d == today:
                groups["Today"].append(n)
            elif d == yesterday:
                groups["Yesterday"].append(n)
            else:
                groups["Earlier"].append(n)
        else:
            groups["Earlier"].append(n)

    result = []
    for label in ("Today", "Yesterday", "Earlier"):
        items = groups[label]
        if items:
            result.append({
                "label": label,
                "items": [
                    {
                        "id": n.id,
                        "title": n.title,
                        "body": n.body,
                        "notif_type": n.notif_type,
                        "icon_emoji": n.icon_emoji,
                        "is_read": n.is_read,
                        "created_at": n.created_at.isoformat() if n.created_at else None,
                    }
                    for n in items
                ],
            })
    return result


@router.get("/notifications")
async def get_notifications(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    notifs_result = await session.exec(
        select(UserNotification)
        .where(UserNotification.user_id == current_user.id)
        .order_by(UserNotification.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    notifications = notifs_result.all()

    unread_result = await session.exec(
        select(func.count(UserNotification.id)).where(
            UserNotification.user_id == current_user.id,
            UserNotification.is_read == False,
        )
    )
    unread_count = unread_result.first() or 0
    if isinstance(unread_count, tuple):
        unread_count = unread_count[0]

    return {
        "data": {
            "unread_count": int(unread_count),
            "groups": _group_notifications(notifications),
        }
    }


@router.get("/notifications/unread-count")
async def notifications_unread_count(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.exec(
        select(func.count(UserNotification.id)).where(
            UserNotification.user_id == current_user.id,
            UserNotification.is_read == False,
        )
    )
    count = result.first() or 0
    if isinstance(count, tuple):
        count = count[0]
    return {"data": {"count": int(count)}}


@router.post("/notifications/mark-read")
async def mark_notifications_read(
    payload: MarkReadPayload,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not payload.ids:
        return {"data": {"updated": 0}}
    notifs_result = await session.exec(
        select(UserNotification).where(
            UserNotification.user_id == current_user.id,
            UserNotification.id.in_(payload.ids),
        )
    )
    updated = 0
    for n in notifs_result.all():
        n.is_read = True
        session.add(n)
        updated += 1
    await session.commit()
    return {"data": {"updated": updated}}


@router.post("/notifications/mark-all-read")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    notifs_result = await session.exec(
        select(UserNotification).where(
            UserNotification.user_id == current_user.id,
            UserNotification.is_read == False,
        )
    )
    count = 0
    for n in notifs_result.all():
        n.is_read = True
        session.add(n)
        count += 1
    await session.commit()
    return {"data": {"updated": count}}


# ============================================================
# Hierarchy — Read-Only (for onboarding dropdowns)
# ============================================================

@router.get("/exam-categories")
async def list_exam_categories(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.exec(
        select(ExamCategory).order_by(
            case((ExamCategory.sort_order == None, 1), else_=0),
            ExamCategory.sort_order,
            ExamCategory.name,
        )
    )
    categories = result.all()
    return {"data": [{"id": c.id, "name": c.name} for c in categories]}


@router.get("/exams")
async def list_exams(
    category_id: Optional[int] = Query(default=None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    query = select(Exam)
    if category_id is not None:
        query = query.where(Exam.exam_category_id == category_id)
    query = query.order_by(
        case((Exam.sort_order == None, 1), else_=0),
        Exam.sort_order,
        Exam.name,
    )
    result = await session.exec(query)
    exams = result.all()
    return {"data": [{"id": e.id, "name": e.name, "exam_category_id": e.exam_category_id} for e in exams]}


@router.get("/exam-mediums")
async def list_exam_mediums(
    exam_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.exec(
        select(CompExamMedium)
        .where(CompExamMedium.exam_id == exam_id)
        .order_by(
            case((CompExamMedium.sort_order == None, 1), else_=0),
            CompExamMedium.sort_order,
            CompExamMedium.name,
        )
    )
    mediums = result.all()
    return {"data": [{"id": m.id, "name": m.name, "exam_id": m.exam_id} for m in mediums]}


@router.get("/exam-levels")
async def list_exam_levels(
    medium_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.exec(
        select(Level)
        .where(Level.medium_id == medium_id)
        .order_by(
            case((Level.sort_order == None, 1), else_=0),
            Level.sort_order,
            Level.name,
        )
    )
    levels = result.all()
    return {"data": [{"id": l.id, "name": l.name, "medium_id": l.medium_id} for l in levels]}


# ============================================================
# Onboarding
# ============================================================

class OnboardingPayload(BaseModel):
    exam_id: Optional[int] = None
    comp_medium_id: Optional[int] = None
    level_id: Optional[int] = None
    exam_date: Optional[date] = None
    daily_commitment_hours: Optional[int] = None


def _build_onboarding_response(profile: UserCompProfile, exam=None, medium=None, level=None) -> dict:
    days_until_exam = None
    if profile.exam_date:
        delta = profile.exam_date - date.today()
        days_until_exam = max(0, delta.days)
    return {
        "exam_id": profile.exam_id,
        "exam_name": exam.name if exam else None,
        "comp_medium_id": profile.comp_medium_id,
        "medium_name": medium.name if medium else None,
        "level_id": profile.level_id,
        "level_name": level.name if level else None,
        "exam_date": profile.exam_date.isoformat() if profile.exam_date else None,
        "daily_commitment_hours": profile.daily_commitment_hours,
        "days_until_exam": days_until_exam,
    }


@router.post("/onboarding")
async def upsert_onboarding(
    payload: OnboardingPayload,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.exec(
        select(UserCompProfile).where(UserCompProfile.user_id == current_user.id)
    )
    profile = result.first()

    if profile is None:
        profile = UserCompProfile(user_id=current_user.id)

    for field, value in payload.dict(exclude_unset=True).items():
        setattr(profile, field, value)

    session.add(profile)
    await session.commit()
    await session.refresh(profile)

    exam = await session.get(Exam, profile.exam_id) if profile.exam_id else None
    medium = await session.get(CompExamMedium, profile.comp_medium_id) if profile.comp_medium_id else None
    level = await session.get(Level, profile.level_id) if profile.level_id else None

    return {"data": _build_onboarding_response(profile, exam, medium, level)}


@router.get("/onboarding")
async def get_onboarding(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.exec(
        select(UserCompProfile).where(UserCompProfile.user_id == current_user.id)
    )
    profile = result.first()
    if not profile:
        raise HTTPException(status_code=404, detail="Onboarding not completed")

    exam = await session.get(Exam, profile.exam_id) if profile.exam_id else None
    medium = await session.get(CompExamMedium, profile.comp_medium_id) if profile.comp_medium_id else None
    level = await session.get(Level, profile.level_id) if profile.level_id else None

    return {"data": _build_onboarding_response(profile, exam, medium, level)}


# ============================================================
# Previous Year Papers (student-facing, read-only)
# ============================================================

@router.get("/pyq")
async def get_pyq(
    level_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Return enabled previous year papers for a level."""
    result = await session.exec(
        select(CompPreviousYearPaper)
        .where(
            CompPreviousYearPaper.level_id == level_id,
            CompPreviousYearPaper.enabled == True,
        )
        .order_by(
            case((CompPreviousYearPaper.sort_order == None, 1), else_=0),
            CompPreviousYearPaper.sort_order,
            CompPreviousYearPaper.year.desc().nulls_last(),
        )
    )
    papers = result.all()
    return {
        "data": [
            {
                "id": p.id,
                "title": p.title,
                "year": p.year,
                "num_questions": p.num_questions,
                "num_pages": p.num_pages,
                "file_url": p.file_url,
                "is_premium": p.is_premium,
            }
            for p in papers
        ]
    }
