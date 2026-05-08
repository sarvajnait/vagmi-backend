from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, delete
from app.services.database import get_session
from app.models.user import User
from app.models.llm_usage import LLMUsage
from app.models.subscriptions import UserSubscription, RazorpayOrder
from app.models.activities import ActivityPlaySession
from app.models.comp_activities import CompActivityPlaySession
from app.models.comp_streak import UserStreak, UserStreakDay, UserMilestone
from app.models.comp_wrong_answers import WrongAnswerEntry
from app.models.comp_study_time import StudyTimeLog
from app.models.user_notifications import UserNotification

router = APIRouter()

_TEST_PHONE = "7406832289"


@router.delete("/delete-test-user")
async def delete_test_user(db: AsyncSession = Depends(get_session)):
    result = await db.exec(select(User).where(User.phone == _TEST_PHONE))
    user = result.first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with phone {_TEST_PHONE} not found")

    user_id = user.id

    # Delete non-CASCADE tables first
    await db.exec(delete(LLMUsage).where(LLMUsage.user_id == user_id))
    await db.exec(delete(UserSubscription).where(UserSubscription.user_id == user_id))
    await db.exec(delete(RazorpayOrder).where(RazorpayOrder.user_id == user_id))

    # Delete user — DB CASCADE handles the rest
    await db.delete(user)
    await db.commit()

    return {"deleted": True, "phone": _TEST_PHONE, "user_id": user_id}
