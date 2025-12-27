from datetime import datetime, timedelta, timezone
from typing import Dict
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.models import LLMUsage


def get_day_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    if now is None:
        now = datetime.now(timezone.utc)
    day_start = datetime(
        year=now.year,
        month=now.month,
        day=now.day,
        tzinfo=timezone.utc,
    )
    day_end = day_start + timedelta(days=1)
    return day_start, day_end


async def get_user_daily_total(session: AsyncSession, user_id: int) -> int:
    day_start, day_end = get_day_window()
    _result = await session.exec(
        select(func.coalesce(func.sum(LLMUsage.total_tokens), 0)).where(
            LLMUsage.user_id == user_id,
            LLMUsage.created_at >= day_start,
            LLMUsage.created_at < day_end,
        )
    )
    result = _result.one()
    return int(result or 0)


async def record_usage_metadata(
    session: AsyncSession, user_id: int, usage_metadata: Dict[str, dict]
) -> None:
    for model_name, usage in usage_metadata.items():
        session.add(
            LLMUsage(
                user_id=user_id,
                model_name=model_name,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                input_token_details=usage.get("input_token_details"),
                output_token_details=usage.get("output_token_details"),
            )
        )
    await session.commit()
