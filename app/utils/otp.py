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

def send_sms(phone: str, message: str):
    """Send SMS using Textlocal API"""
    numbers = f"[91{phone}]"
    payload = {
        "apikey": settings.TEXTLOCAL_API_KEY,
        "numbers": numbers,
        "message": message,
        "sender": settings.TEXTLOCAL_SENDER,
    }
    requests.post("https://api.textlocal.in/send/", data=payload)
    return True

def prepare_otp_sms(phone: str, otp: str, app_signature: str = "") -> str:
    return (
        f"{otp} is the OTP to verify your mobile number for Cherri Learn. "
        f"This OTP is valid for 5 minutes. pls do not share it with anyone.\n{app_signature}"
    )
