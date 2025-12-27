from datetime import date
from typing import Optional, TYPE_CHECKING

from sqlmodel import Field, Relationship, SQLModel

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.user import User


# --------------------
# Subscription Plan
# --------------------
class SubscriptionPlanBase(SQLModel):
    name: str
    class_level_id: int = Field(foreign_key="class_levels.id")
    board_id: int = Field(foreign_key="boards.id")
    medium_id: int = Field(foreign_key="mediums.id")
    amount_inr: int = Field(default=99)
    is_active: bool = Field(default=True)
    description: Optional[str] = None


class SubscriptionPlan(SubscriptionPlanBase, BaseModel, table=True):
    __tablename__ = "subscription_plans"

    id: Optional[int] = Field(default=None, primary_key=True)
    subscriptions: list["UserSubscription"] = Relationship(
        back_populates="plan", cascade_delete=True
    )


class SubscriptionPlanCreate(SubscriptionPlanBase):
    pass


class SubscriptionPlanUpdate(SQLModel):
    name: Optional[str] = None
    class_level_id: Optional[int] = None
    board_id: Optional[int] = None
    medium_id: Optional[int] = None
    amount_inr: Optional[int] = None
    is_active: Optional[bool] = None
    description: Optional[str] = None


class SubscriptionPlanRead(SubscriptionPlanBase):
    id: int
    class_level_name: Optional[str] = None
    board_name: Optional[str] = None
    medium_name: Optional[str] = None


# --------------------
# User Subscription
# --------------------
class UserSubscriptionBase(SQLModel):
    user_id: int = Field(foreign_key="user.id")
    plan_id: int = Field(foreign_key="subscription_plans.id")
    starts_at: date = Field(default=date(2026, 1, 1))
    ends_at: date = Field(default=date(2026, 2, 1))
    status: str = Field(default="active")
    notes: Optional[str] = None


class UserSubscription(UserSubscriptionBase, BaseModel, table=True):
    __tablename__ = "user_subscriptions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user: "User" = Relationship()
    plan: SubscriptionPlan = Relationship(back_populates="subscriptions")


class UserSubscriptionCreate(UserSubscriptionBase):
    pass


class UserSubscriptionUpdate(SQLModel):
    plan_id: Optional[int] = None
    starts_at: Optional[date] = None
    ends_at: Optional[date] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class UserSubscriptionRead(UserSubscriptionBase):
    id: int
    plan_name: Optional[str] = None
    user_name: Optional[str] = None
