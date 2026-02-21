import hashlib
import hmac

import razorpay

from app.core.config import settings


def _client() -> razorpay.Client:
    return razorpay.Client(
        auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
    )


def create_order(amount_inr: int, receipt: str, notes: dict | None = None) -> dict:
    """Create a Razorpay order. amount_inr is in rupees — converted to paise internally."""
    client = _client()
    data = {
        "amount": amount_inr * 100,  # paise
        "currency": "INR",
        "receipt": receipt,
        "notes": notes or {},
    }
    return client.order.create(data=data)


def verify_signature(razorpay_order_id: str, razorpay_payment_id: str, razorpay_signature: str) -> bool:
    """Verify Razorpay payment signature using HMAC-SHA256."""
    message = f"{razorpay_order_id}|{razorpay_payment_id}"
    expected = hmac.new(
        settings.RAZORPAY_KEY_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, razorpay_signature)
