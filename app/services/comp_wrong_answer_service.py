from datetime import datetime, timezone
from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.comp_activities import (
    CompActivityAnswer, CompActivityPlaySession, CompChapterActivity, CompActivityGroup,
)
from app.models.comp_wrong_answers import WrongAnswerEntry
from app.models.competitive_hierarchy import CompChapter


async def process_session_answers(session_id: int, user_id: int, db: AsyncSession) -> None:
    answers_result = await db.exec(
        select(CompActivityAnswer, CompChapterActivity)
        .join(CompChapterActivity, CompChapterActivity.id == CompActivityAnswer.activity_id)
        .where(CompActivityAnswer.session_id == session_id)
    )
    now = datetime.now(timezone.utc)

    for answer, activity in answers_result.all():
        existing_result = await db.exec(
            select(WrongAnswerEntry).where(
                WrongAnswerEntry.user_id == user_id,
                WrongAnswerEntry.activity_id == activity.id,
            )
        )
        existing = existing_result.first()

        if answer.is_correct:
            if existing and not existing.is_mastered:
                existing.is_mastered = True
                db.add(existing)
        else:
            if existing:
                existing.times_attempted += 1
                existing.last_wrong_at = now
                existing.is_mastered = False
                db.add(existing)
            else:
                db.add(WrongAnswerEntry(
                    user_id=user_id,
                    activity_id=activity.id,
                    comp_chapter_id=activity.comp_chapter_id,
                    activity_group_id=activity.activity_group_id,
                    times_attempted=1,
                    last_wrong_at=now,
                ))

    await db.commit()


async def get_wrong_answers(
    user_id: int,
    db: AsyncSession,
    comp_chapter_id: Optional[int] = None,
    subject_id: Optional[int] = None,
) -> dict:
    query = (
        select(WrongAnswerEntry, CompChapterActivity, CompActivityGroup, CompChapter)
        .join(CompChapterActivity, CompChapterActivity.id == WrongAnswerEntry.activity_id)
        .join(CompActivityGroup, CompActivityGroup.id == WrongAnswerEntry.activity_group_id, isouter=True)
        .join(CompChapter, CompChapter.id == WrongAnswerEntry.comp_chapter_id, isouter=True)
        .where(
            WrongAnswerEntry.user_id == user_id,
            WrongAnswerEntry.is_mastered == False,
        )
    )
    if comp_chapter_id is not None:
        query = query.where(WrongAnswerEntry.comp_chapter_id == comp_chapter_id)

    result = await db.exec(query)
    rows = result.all()

    items = []
    for entry, activity, group, chapter in rows:
        items.append({
            "activity_id": activity.id,
            "question_text": activity.question_text,
            "options": activity.options,
            "correct_option_index": activity.correct_option_index,
            "answer_description": activity.answer_description,
            "comp_chapter_id": entry.comp_chapter_id,
            "comp_chapter_title": chapter.name if chapter else None,
            "activity_group_id": entry.activity_group_id,
            "activity_group_name": group.name if group else None,
            "times_attempted": entry.times_attempted,
            "last_wrong_at": entry.last_wrong_at.isoformat() if entry.last_wrong_at else None,
        })

    return {"total_count": len(items), "items": items}


async def create_retry_session(
    user_id: int,
    db: AsyncSession,
    comp_chapter_id: Optional[int] = None,
    activity_ids: Optional[list[int]] = None,
) -> tuple[int, list[int]]:
    if activity_ids:
        query = select(WrongAnswerEntry).where(
            WrongAnswerEntry.user_id == user_id,
            WrongAnswerEntry.activity_id.in_(activity_ids),
            WrongAnswerEntry.is_mastered == False,
        )
    elif comp_chapter_id is not None:
        query = select(WrongAnswerEntry).where(
            WrongAnswerEntry.user_id == user_id,
            WrongAnswerEntry.comp_chapter_id == comp_chapter_id,
            WrongAnswerEntry.is_mastered == False,
        )
    else:
        raise ValueError("Either comp_chapter_id or activity_ids must be provided")

    entries_result = await db.exec(query)
    entries = entries_result.all()
    if not entries:
        raise ValueError("No wrong answers found to retry")

    wrong_activity_ids = [e.activity_id for e in entries]

    # M-7: derive scope from entries, not from caller-provided value
    derived_chapter_id = next((e.comp_chapter_id for e in entries if e.comp_chapter_id), None)

    from app.models.comp_activities import CompActivityPlaySession
    play_session = CompActivityPlaySession(
        user_id=user_id,
        comp_chapter_id=derived_chapter_id,
        total_questions=len(wrong_activity_ids),
    )
    db.add(play_session)
    await db.commit()
    await db.refresh(play_session)
    return play_session.id, wrong_activity_ids
