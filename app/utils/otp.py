import pyotp
import base64
import os
import requests
from app.core.config import settings


def generate_otp():
    """Generate a 5-minute OTP and a secret key"""
    secret = base64.b32encode(os.urandom(20)).decode("utf-8")
    totp = pyotp.TOTP(secret, interval=300)
    otp = totp.now()
    return otp, secret


def validate_otp(otp: str, secret: str) -> bool:
    """Validate the OTP against the secret"""
    totp = pyotp.TOTP(secret, interval=300)
    return totp.verify(otp, valid_window=1)


def send_sms(phone: str, otp: str):
    """Send OTP SMS using MSG91 Flow API"""

    url = "https://control.msg91.com/api/v5/flow"

    payload = {
        "template_id": settings.MSG91_TEMPLATE_ID,  # Replace with MSG91 SMS template ID
        "short_url": "0",  # Disable short URLs (set "1" to enable)
        "recipients": [
            {
                "mobiles": f"91{phone}",
                "otp": otp,  # Match variable names in your MSG91 template
            }
        ],
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authkey": settings.MSG91_AUTH_KEY,  # Replace with your MSG91 Auth Key
    }

    response = requests.post(url, json=payload, headers=headers)

    return response.json()
