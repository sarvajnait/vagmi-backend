"""Razorpay payment integration endpoints."""

from datetime import date, timedelta
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.config import settings
from app.models import User
from app.models.subscriptions import RazorpayOrder, SubscriptionPlan, UserSubscription
from app.services.database import get_session
from app.services.razorpay_service import create_order, verify_signature
from app.services.subscriptions import get_active_subscription_summary

router = APIRouter()


# --------------------
# Schemas
# --------------------
class CreateOrderRequest(BaseModel):
    plan_id: int


class CreateOrderResponse(BaseModel):
    razorpay_order_id: str
    amount: int  # paise
    currency: str
    key_id: str
    plan_name: str
    duration_days: int


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


# --------------------
# Endpoints
# --------------------
@router.post("/create-order", response_model=Dict[str, CreateOrderResponse])
async def create_payment_order(
    body: CreateOrderRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    plan = await session.get(SubscriptionPlan, body.plan_id)
    if not plan or not plan.is_active:
        raise HTTPException(status_code=404, detail="Plan not found or inactive")

    receipt = f"rcpt_{current_user.id}_{plan.id}_{int(date.today().strftime('%Y%m%d'))}"

    try:
        rz_order = create_order(
            amount_inr=plan.amount_inr,
            receipt=receipt,
            notes={"user_id": str(current_user.id), "plan_id": str(plan.id)},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Razorpay error: {str(e)}")

    db_order = RazorpayOrder(
        user_id=current_user.id,
        plan_id=plan.id,
        razorpay_order_id=rz_order["id"],
        amount=rz_order["amount"],
        currency=rz_order["currency"],
        receipt=receipt,
        status="created",
    )
    session.add(db_order)
    await session.commit()

    return {
        "data": CreateOrderResponse(
            razorpay_order_id=rz_order["id"],
            amount=rz_order["amount"],
            currency=rz_order["currency"],
            key_id=settings.RAZORPAY_KEY_ID,
            plan_name=plan.name,
            duration_days=plan.duration_days,
        )
    }


@router.post("/verify")
async def verify_payment(
    body: VerifyPaymentRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    # Find our order record
    result = await session.exec(
        select(RazorpayOrder).where(
            RazorpayOrder.razorpay_order_id == body.razorpay_order_id,
            RazorpayOrder.user_id == current_user.id,
        )
    )
    db_order = result.first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")

    if db_order.status == "paid":
        raise HTTPException(status_code=400, detail="Order already processed")

    # Verify signature
    is_valid = verify_signature(
        razorpay_order_id=body.razorpay_order_id,
        razorpay_payment_id=body.razorpay_payment_id,
        razorpay_signature=body.razorpay_signature,
    )
    if not is_valid:
        db_order.status = "failed"
        session.add(db_order)
        await session.commit()
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    # Update order
    db_order.status = "paid"
    db_order.razorpay_payment_id = body.razorpay_payment_id
    db_order.razorpay_signature = body.razorpay_signature
    session.add(db_order)

    # Create subscription
    plan = await session.get(SubscriptionPlan, db_order.plan_id)
    starts_at = date.today()
    ends_at = plan.fixed_end_date if plan.fixed_end_date else starts_at + timedelta(days=plan.duration_days)

    subscription = UserSubscription(
        user_id=current_user.id,
        plan_id=plan.id,
        starts_at=starts_at,
        ends_at=ends_at,
        status="active",
        notes=f"Razorpay payment {body.razorpay_payment_id}",
    )
    session.add(subscription)
    await session.commit()

    # Return updated subscription summary
    active_sub = await get_active_subscription_summary(session, current_user.id)
    return {
        "message": "Payment successful",
        "subscription": active_sub,
    }


@router.post("/webhook")
async def razorpay_webhook(request: Request, session: AsyncSession = Depends(get_session)):
    """
    Razorpay webhook handler — backup for cases where frontend verify call fails.
    Configure this URL in Razorpay dashboard under Webhooks.
    Events: payment.captured
    """
    import hashlib
    import hmac

    payload_bytes = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    expected = hmac.new(
        settings.RAZORPAY_KEY_SECRET.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    import json
    payload = json.loads(payload_bytes)
    event = payload.get("event")

    if event == "payment.captured":
        payment = payload["payload"]["payment"]["entity"]
        order_id = payment.get("order_id")
        payment_id = payment.get("id")

        result = await session.exec(
            select(RazorpayOrder).where(RazorpayOrder.razorpay_order_id == order_id)
        )
        db_order = result.first()

        if db_order and db_order.status != "paid":
            db_order.status = "paid"
            db_order.razorpay_payment_id = payment_id
            session.add(db_order)

            # Create subscription if not already created (idempotent)
            existing = await session.exec(
                select(UserSubscription).where(
                    UserSubscription.user_id == db_order.user_id,
                    UserSubscription.plan_id == db_order.plan_id,
                    UserSubscription.notes.contains(payment_id),
                )
            )
            if not existing.first():
                plan = await session.get(SubscriptionPlan, db_order.plan_id)
                starts_at = date.today()
                ends_at = plan.fixed_end_date if plan.fixed_end_date else starts_at + timedelta(days=plan.duration_days)
                session.add(UserSubscription(
                    user_id=db_order.user_id,
                    plan_id=plan.id,
                    starts_at=starts_at,
                    ends_at=ends_at,
                    status="active",
                    notes=f"Razorpay payment {payment_id}",
                ))

            await session.commit()

    return {"status": "ok"}
