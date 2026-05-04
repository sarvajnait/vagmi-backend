from datetime import date, timedelta
from typing import Optional
from sqlalchemy import func, case
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.comp_activities import (
    CompActivityAnswer, CompActivityPlaySession, CompActivityGroup, CompChapterActivity,
)
from app.models.competitive_hierarchy import CompChapter, CompSubject, Level
from app.models.comp_student_content import CompStudentNote, CompStudentVideo
from app.models.comp_study_time import StudyTimeLog


async def get_subject_chapters(user_id: int, subject_id: int, db: AsyncSession) -> dict:
    subject = await db.get(CompSubject, subject_id)
    if not subject:
        return None

    chapters_result = await db.exec(
        select(CompChapter).where(CompChapter.subject_id == subject_id)
        .order_by(
            case((CompChapter.sort_order == None, 1), else_=0),
            CompChapter.sort_order,
        )
    )
    chapters = chapters_result.all()

    chapter_ids = [c.id for c in chapters]

    # Distinct questions answered per chapter (replay-safe)
    distinct_result = await db.exec(
        select(
            CompChapterActivity.comp_chapter_id,
            func.count(func.distinct(CompActivityAnswer.activity_id)).label("answered"),
            func.count(func.distinct(
                case((CompActivityAnswer.is_correct == True, CompActivityAnswer.activity_id), else_=None)
            )).label("correct"),
        )
        .join(CompActivityAnswer, CompActivityAnswer.activity_id == CompChapterActivity.id)
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == user_id,
            CompChapterActivity.comp_chapter_id.in_(chapter_ids),
        )
        .group_by(CompChapterActivity.comp_chapter_id)
    )
    chapter_stats: dict[int, dict] = {
        row[0]: {"answered": row[1], "correct": row[2], "last_at": None}
        for row in distinct_result.all()
    }

    # last_active_at still comes from sessions
    sessions_result = await db.exec(
        select(CompActivityPlaySession.comp_chapter_id, CompActivityPlaySession.completed_at).where(
            CompActivityPlaySession.user_id == user_id,
            CompActivityPlaySession.comp_chapter_id.in_(chapter_ids),
            CompActivityPlaySession.completed_at.is_not(None),
        )
    )
    for cid, completed_at in sessions_result.all():
        if cid in chapter_stats:
            if chapter_stats[cid]["last_at"] is None or completed_at > chapter_stats[cid]["last_at"]:
                chapter_stats[cid]["last_at"] = completed_at

    # Total published questions per chapter
    q_counts_result = await db.exec(
        select(CompChapterActivity.comp_chapter_id, func.count(CompChapterActivity.id))
        .where(
            CompChapterActivity.comp_chapter_id.in_(chapter_ids),
            CompChapterActivity.is_published == True,
        )
        .group_by(CompChapterActivity.comp_chapter_id)
    )
    q_counts = {row[0]: row[1] for row in q_counts_result.all()}

    # Activity group counts per chapter
    group_counts_result = await db.exec(
        select(CompActivityGroup.comp_chapter_id, func.count(CompActivityGroup.id))
        .where(CompActivityGroup.comp_chapter_id.in_(chapter_ids))
        .group_by(CompActivityGroup.comp_chapter_id)
    )
    group_counts = {row[0]: row[1] for row in group_counts_result.all()}

    chapter_list = []
    chapters_completed = 0
    total_questions = sum(q_counts.values())
    last_active = None
    last_active_chapter = None

    for chapter in chapters:
        cid = chapter.id
        q_total = q_counts.get(cid, 0)
        stats = chapter_stats.get(cid, {})
        answered = stats.get("answered", 0)
        progress_pct = round(answered / q_total * 100) if q_total > 0 else 0

        if progress_pct >= 100:
            status = "completed"
            chapters_completed += 1
        elif answered > 0:
            status = "in_progress"
        else:
            status = "not_started"

        last_at = stats.get("last_at")
        if last_at and (last_active is None or last_at > last_active):
            last_active = last_at
            last_active_chapter = {
                "chapter_id": cid,
                "chapter_name": chapter.name,
                "progress_pct": progress_pct,
            }

        chapter_list.append({
            "id": cid,
            "title": chapter.name,
            "sort_order": chapter.sort_order,
            "question_count": q_total,
            "progress_pct": progress_pct,
            "status": status,
            "activity_group_count": group_counts.get(cid, 0),
        })

    # Study time for this subject's chapters
    study_time_result = await db.exec(
        select(func.coalesce(func.sum(StudyTimeLog.duration_seconds), 0)).where(
            StudyTimeLog.user_id == user_id
        )
    )
    study_time_seconds = study_time_result.first() or 0
    if isinstance(study_time_seconds, tuple):
        study_time_seconds = study_time_seconds[0]

    return {
        "subject": {"id": subject_id, "name": subject.name},
        "stats": {
            "chapters_completed": chapters_completed,
            "total_chapters": len(chapters),
            "total_questions": total_questions,
            "study_time_seconds": int(study_time_seconds),
        },
        "last_active_chapter": last_active_chapter,
        "chapters": chapter_list,
    }


async def get_chapter_detail(user_id: int, chapter_id: int, db: AsyncSession) -> Optional[dict]:
    chapter = await db.get(CompChapter, chapter_id)
    if not chapter:
        return None

    # Get activity groups for this chapter
    groups_result = await db.exec(
        select(CompActivityGroup).where(CompActivityGroup.comp_chapter_id == chapter_id)
        .order_by(
            case((CompActivityGroup.sort_order == None, 1), else_=0),
            CompActivityGroup.sort_order,
        )
    )
    groups = groups_result.all()
    group_ids = [g.id for g in groups]

    # Total published questions per group
    q_counts_result = await db.exec(
        select(CompChapterActivity.activity_group_id, func.count(CompChapterActivity.id))
        .where(
            CompChapterActivity.activity_group_id.in_(group_ids),
            CompChapterActivity.is_published == True,
        )
        .group_by(CompChapterActivity.activity_group_id)
    )
    q_counts = {row[0]: row[1] for row in q_counts_result.all()}

    # Distinct questions answered/correct per group (replay-safe)
    user_answers_result = await db.exec(
        select(
            CompChapterActivity.activity_group_id,
            func.count(func.distinct(CompActivityAnswer.activity_id)).label("answered"),
            func.count(func.distinct(
                case((CompActivityAnswer.is_correct == True, CompActivityAnswer.activity_id), else_=None)
            )).label("correct"),
        )
        .join(CompChapterActivity, CompChapterActivity.id == CompActivityAnswer.activity_id)
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == user_id,
            CompChapterActivity.activity_group_id.in_(group_ids),
        )
        .group_by(CompChapterActivity.activity_group_id)
    )
    user_stats = {row[0]: {"answered": row[1], "correct": row[2]} for row in user_answers_result.all()}

    # Overall chapter progress
    total_q = sum(q_counts.values())
    total_answered = sum(s["answered"] for s in user_stats.values())
    overall_progress_pct = round(total_answered / total_q * 100) if total_q > 0 else 0

    activity_groups = []
    for group in groups:
        gid = group.id
        q_total = q_counts.get(gid, 0)
        stats = user_stats.get(gid, {"answered": 0, "correct": 0})
        answered = stats["answered"]
        correct = stats["correct"] or 0

        accuracy_pct = round(correct / answered * 100) if answered > 0 else None

        if answered >= q_total and q_total > 0:
            status = "completed"
        elif answered > 0:
            status = "in_progress"
        else:
            status = "not_started"

        activity_groups.append({
            "id": gid,
            "name": group.name,
            "question_count": q_total,
            "user_questions_done": answered,
            "user_accuracy_pct": accuracy_pct,
            "status": status,
        })

    # Notes and videos for this chapter
    notes_result = await db.exec(
        select(CompStudentNote).where(
            CompStudentNote.comp_chapter_id == chapter_id,
            CompStudentNote.is_published == True,
        )
    )
    notes = [{"id": n.id, "title": n.title, "language": n.language, "file_url": n.file_url} for n in notes_result.all()]

    videos_result = await db.exec(
        select(CompStudentVideo).where(CompStudentVideo.comp_chapter_id == chapter_id)
    )
    videos = [{"id": v.id, "title": v.title, "file_url": v.file_url} for v in videos_result.all()]

    return {
        "chapter": {"id": chapter_id, "title": chapter.name},
        "overall_progress_pct": overall_progress_pct,
        "activity_groups": activity_groups,
        "notes": notes,
        "videos": videos,
    }


async def get_performance_dashboard(user_id: int, db: AsyncSession, level_id: Optional[int] = None) -> dict:
    today = date.today()
    week_ago = today - timedelta(days=7)

    # Get all chapters in scope (optionally filtered by level)
    chapter_query = select(CompChapter)
    if level_id is not None:
        chapter_query = chapter_query.join(CompSubject, CompSubject.id == CompChapter.subject_id).where(
            CompSubject.level_id == level_id
        )
    chapters_result = await db.exec(chapter_query)
    chapters = chapters_result.all()
    chapter_ids = [c.id for c in chapters]
    chapter_subject_map = {c.id: c.subject_id for c in chapters}

    if not chapter_ids:
        return _empty_performance()

    # Distinct questions answered/correct overall and this week (replay-safe)
    overall_result = await db.exec(
        select(
            func.count(func.distinct(CompActivityAnswer.activity_id)).label("answered"),
            func.count(func.distinct(
                case((CompActivityAnswer.is_correct == True, CompActivityAnswer.activity_id), else_=None)
            )).label("correct"),
        )
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == user_id,
            CompActivityPlaySession.comp_chapter_id.in_(chapter_ids),
            CompActivityPlaySession.status == "completed",
        )
    )
    _ov = overall_result.first()
    total_answered = _ov[0] if _ov else 0
    total_correct = _ov[1] if _ov else 0
    accuracy_pct = round(total_correct / total_answered * 100) if total_answered > 0 else 0

    week_result = await db.exec(
        select(
            func.count(func.distinct(CompActivityAnswer.activity_id)).label("answered"),
            func.count(func.distinct(
                case((CompActivityAnswer.is_correct == True, CompActivityAnswer.activity_id), else_=None)
            )).label("correct"),
        )
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == user_id,
            CompActivityPlaySession.comp_chapter_id.in_(chapter_ids),
            CompActivityPlaySession.status == "completed",
            CompActivityPlaySession.completed_at >= week_ago,
        )
    )
    _wk = week_result.first()
    week_answered = _wk[0] if _wk else 0
    week_correct = _wk[1] if _wk else 0

    prev_result = await db.exec(
        select(
            func.count(func.distinct(CompActivityAnswer.activity_id)).label("answered"),
            func.count(func.distinct(
                case((CompActivityAnswer.is_correct == True, CompActivityAnswer.activity_id), else_=None)
            )).label("correct"),
        )
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == user_id,
            CompActivityPlaySession.comp_chapter_id.in_(chapter_ids),
            CompActivityPlaySession.status == "completed",
            CompActivityPlaySession.completed_at < week_ago,
        )
    )
    _pv = prev_result.first()
    prev_total = _pv[0] if _pv else 0
    prev_correct = _pv[1] if _pv else 0
    prev_accuracy = round(prev_correct / prev_total * 100) if prev_total > 0 else 0
    accuracy_delta_week = accuracy_pct - prev_accuracy

    # Sessions still needed for subject-level breakdown
    sessions_result = await db.exec(
        select(CompActivityPlaySession).where(
            CompActivityPlaySession.user_id == user_id,
            CompActivityPlaySession.comp_chapter_id.in_(chapter_ids),
            CompActivityPlaySession.status == "completed",
        )
    )
    sessions = sessions_result.all()

    # Study time
    study_result = await db.exec(
        select(func.coalesce(func.sum(StudyTimeLog.duration_seconds), 0)).where(
            StudyTimeLog.user_id == user_id
        )
    )
    total_study_seconds = study_result.first() or 0
    if isinstance(total_study_seconds, tuple):
        total_study_seconds = total_study_seconds[0]

    week_study_result = await db.exec(
        select(func.coalesce(func.sum(StudyTimeLog.duration_seconds), 0)).where(
            StudyTimeLog.user_id == user_id,
            StudyTimeLog.logged_date >= week_ago,
        )
    )
    week_study_seconds = week_study_result.first() or 0
    if isinstance(week_study_seconds, tuple):
        week_study_seconds = week_study_seconds[0]

    # Per-subject breakdown
    subjects_result = await db.exec(
        select(CompSubject).where(CompSubject.id.in_(set(chapter_subject_map.values())))
        .order_by(case((CompSubject.sort_order == None, 1), else_=0), CompSubject.sort_order)
    )
    subjects = subjects_result.all()

    # Distinct questions answered/correct per subject (replay-safe)
    subject_ids = [s.id for s in subjects]
    subj_answers_result = await db.exec(
        select(
            CompSubject.id,
            func.count(func.distinct(CompActivityAnswer.activity_id)).label("answered"),
            func.count(func.distinct(
                case((CompActivityAnswer.is_correct == True, CompActivityAnswer.activity_id), else_=None)
            )).label("correct"),
        )
        .join(CompChapter, CompChapter.subject_id == CompSubject.id)
        .join(CompChapterActivity, CompChapterActivity.comp_chapter_id == CompChapter.id)
        .join(CompActivityAnswer, CompActivityAnswer.activity_id == CompChapterActivity.id)
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == user_id,
            CompActivityPlaySession.status == "completed",
            CompSubject.id.in_(subject_ids),
        )
        .group_by(CompSubject.id)
    )
    subject_stats: dict[int, dict] = {
        row[0]: {"answered": row[1], "correct": row[2]}
        for row in subj_answers_result.all()
    }

    # Activity groups per subject (topics_done = groups where user answered all questions)
    groups_result = await db.exec(
        select(CompActivityGroup).where(CompActivityGroup.comp_chapter_id.in_(chapter_ids))
    )
    groups = groups_result.all()
    group_to_subject = {g.id: chapter_subject_map.get(g.comp_chapter_id) for g in groups}
    total_groups_per_subject: dict[int, int] = {}
    for g in groups:
        sid = group_to_subject.get(g.id)
        if sid:
            total_groups_per_subject[sid] = total_groups_per_subject.get(sid, 0) + 1

    # M-9: completed groups = groups where distinct answered >= total published questions
    group_answered_result = await db.exec(
        select(
            CompChapterActivity.activity_group_id,
            func.count(func.distinct(CompActivityAnswer.activity_id)).label("answered"),
        )
        .join(CompActivityAnswer, CompActivityAnswer.activity_id == CompChapterActivity.id)
        .join(CompActivityPlaySession, CompActivityPlaySession.id == CompActivityAnswer.session_id)
        .where(
            CompActivityPlaySession.user_id == user_id,
            CompChapterActivity.comp_chapter_id.in_(chapter_ids),
            CompChapterActivity.is_published == True,
        )
        .group_by(CompChapterActivity.activity_group_id)
    )
    group_answered_map = {row[0]: row[1] for row in group_answered_result.all() if row[0]}

    group_total_result = await db.exec(
        select(CompChapterActivity.activity_group_id, func.count(CompChapterActivity.id))
        .where(
            CompChapterActivity.activity_group_id.in_(list(group_answered_map.keys())),
            CompChapterActivity.is_published == True,
        )
        .group_by(CompChapterActivity.activity_group_id)
    )
    group_total_map = {row[0]: row[1] for row in group_total_result.all()}

    completed_group_ids = {
        gid for gid, answered in group_answered_map.items()
        if answered >= group_total_map.get(gid, float("inf"))
    }
    done_groups_per_subject: dict[int, int] = {}
    for gid in completed_group_ids:
        sid = group_to_subject.get(gid)
        if sid:
            done_groups_per_subject[sid] = done_groups_per_subject.get(sid, 0) + 1

    subject_list = []
    for subject in subjects:
        sid = subject.id
        stats = subject_stats.get(sid, {"correct": 0, "answered": 0})
        answered = stats["answered"]
        correct = stats["correct"]
        subj_accuracy = round(correct / answered * 100) if answered > 0 else 0
        subject_list.append({
            "subject_id": sid,
            "subject_name": subject.name,
            "accuracy_pct": subj_accuracy,
            "questions_done": answered,
            "topics_done": done_groups_per_subject.get(sid, 0),
            "total_topics": total_groups_per_subject.get(sid, 0),
        })

    return {
        "overall": {
            "accuracy_pct": accuracy_pct,
            "accuracy_delta_week": accuracy_delta_week,
            "questions_done": total_answered,
            "questions_done_week": week_answered,
            "study_time_seconds": int(total_study_seconds),
            "study_time_seconds_week": int(week_study_seconds),
        },
        "subjects": subject_list,
    }


def _empty_performance() -> dict:
    return {
        "overall": {
            "accuracy_pct": 0,
            "accuracy_delta_week": 0,
            "questions_done": 0,
            "questions_done_week": 0,
            "study_time_seconds": 0,
            "study_time_seconds_week": 0,
        },
        "subjects": [],
    }
