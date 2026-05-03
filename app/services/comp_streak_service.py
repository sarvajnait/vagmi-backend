from datetime import date, datetime, timezone
from typing import Optional
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.comp_streak import UserStreak, UserStreakDay, UserMilestone

MILESTONES = [
    (7, "Week Warrior"),
    (14, "Fortnight Fighter"),
    (21, "3-Week Champion"),
    (30, "Monthly Master"),
    (60, "60-Day Legend"),
    (100, "Centurion"),
]


async def record_activity(user_id: int, session: AsyncSession) -> dict:
    today = date.today()

    # Try insert today's streak day (ignore if already exists)
    existing_day = await session.exec(
        select(UserStreakDay).where(
            UserStreakDay.user_id == user_id,
            UserStreakDay.activity_date == today,
        )
    )
    was_new_day = existing_day.first() is None

    if was_new_day:
        session.add(UserStreakDay(user_id=user_id, activity_date=today))
        await session.flush()

    # Recompute current streak (consecutive days back from today)
    current_streak = await _compute_current_streak(user_id, today, session)

    # Get or create UserStreak row
    streak_row = await session.exec(
        select(UserStreak).where(UserStreak.user_id == user_id)
    )
    streak = streak_row.first()
    if streak is None:
        streak = UserStreak(user_id=user_id, current_streak=0, longest_streak=0)
        session.add(streak)

    streak.current_streak = current_streak
    if current_streak > streak.longest_streak:
        streak.longest_streak = current_streak
    streak.last_activity_date = today
    session.add(streak)

    # Check for new milestones
    milestone_achieved = None
    if was_new_day:
        milestone_achieved = await _check_milestones(user_id, current_streak, session)

    await session.commit()
    await session.refresh(streak)

    return {
        "current_streak": streak.current_streak,
        "was_new_day": was_new_day,
        "milestone_achieved": milestone_achieved,
    }


async def get_streak(user_id: int, session: AsyncSession) -> dict:
    streak_row = await session.exec(
        select(UserStreak).where(UserStreak.user_id == user_id)
    )
    streak = streak_row.first()
    current_streak = streak.current_streak if streak else 0
    longest_streak = streak.longest_streak if streak else 0
    last_activity_date = streak.last_activity_date if streak else None

    # Last 60 days calendar
    from datetime import timedelta
    today = date.today()
    days_result = await session.exec(
        select(UserStreakDay.activity_date).where(
            UserStreakDay.user_id == user_id,
            UserStreakDay.activity_date >= today - timedelta(days=59),
        )
    )
    active_dates = {row for row in days_result.all()}
    calendar = []
    for i in range(59, -1, -1):
        d = today - timedelta(days=i)
        calendar.append({"date": d.isoformat(), "active": d in active_dates})

    # Milestones
    achieved_rows = await session.exec(
        select(UserMilestone).where(UserMilestone.user_id == user_id)
    )
    achieved_map = {m.milestone_days: m.achieved_at for m in achieved_rows.all()}
    milestones = []
    for days, name in MILESTONES:
        achieved = days in achieved_map
        entry = {
            "days": days,
            "name": name,
            "achieved": achieved,
        }
        if achieved:
            entry["achieved_at"] = achieved_map[days].isoformat()
        else:
            entry["days_remaining"] = max(0, days - current_streak)
        milestones.append(entry)

    return {
        "current_streak": current_streak,
        "longest_streak": longest_streak,
        "last_activity_date": last_activity_date.isoformat() if last_activity_date else None,
        "calendar": calendar,
        "milestones": milestones,
    }


async def _compute_current_streak(user_id: int, today: date, session: AsyncSession) -> int:
    from datetime import timedelta
    # Walk backwards from today counting consecutive active days
    all_days_result = await session.exec(
        select(UserStreakDay.activity_date).where(
            UserStreakDay.user_id == user_id,
        ).order_by(UserStreakDay.activity_date.desc())
    )
    active_dates = {row for row in all_days_result.all()}

    streak = 0
    check = today
    while check in active_dates:
        streak += 1
        check = check - timedelta(days=1)
    return streak


async def _check_milestones(user_id: int, current_streak: int, session: AsyncSession) -> Optional[dict]:
    from app.services.notification_service import notify_milestone
    now = datetime.now(timezone.utc)
    milestone_achieved = None
    for days, name in MILESTONES:
        if current_streak >= days:
            existing = await session.exec(
                select(UserMilestone).where(
                    UserMilestone.user_id == user_id,
                    UserMilestone.milestone_days == days,
                )
            )
            if existing.first() is None:
                session.add(UserMilestone(user_id=user_id, milestone_days=days, achieved_at=now))
                milestone_achieved = {"days": days, "name": name}
                try:
                    await notify_milestone(user_id, days, name, session)
                except Exception:
                    pass
    return milestone_achieved
