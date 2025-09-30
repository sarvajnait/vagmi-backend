from fastapi import APIRouter, Depends, Body, Request, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import select, Session
from typing import Optional
from loguru import logger
import pyotp
import base64
import os

from app.models.user import User
from app.schemas.auth import UserCreate, TokenPair, AuthResponse, UserResponse
from app.services.database import get_session
from app.utils.auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    verify_refresh_token,
)
from app.utils.sanitization import (
    sanitize_phone,
    sanitize_string,
    validate_password_strength,
)
from app.core.config import settings
from app.core.limiter import limiter

router = APIRouter()
security = HTTPBearer()


# -----------------------
# Current User Dependency
# -----------------------
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session),
) -> User:
    try:
        token = sanitize_string(credentials.credentials)
        user_id = verify_token(token)
        if not user_id:
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials"
            )

        user = session.get(User, int(user_id))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except ValueError as ve:
        logger.error("token_validation_failed", error=str(ve), exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid token format")


# -----------------------
# Register Endpoint
# -----------------------
@router.post("/register", response_model=AuthResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["register"][0])
async def register_user(
    request: Request, user_data: UserCreate, session: Session = Depends(get_session)
):
    phone = sanitize_phone(user_data.phone)
    password = user_data.password
    board = user_data.board
    medium = user_data.medium
    grade = user_data.grade

    validate_password_strength(password)

    existing_user = session.exec(select(User).where(User.phone == phone)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Phone already registered")

    user = User(
        phone=phone,
        hashed_password=User.hash_password(password),
        board_id=board,
        medium_id=medium,
        class_level_id=grade,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return AuthResponse(
        user=UserResponse(
            id=user.id,
            phone=user.phone,
            board_id=board,
            medium_id=medium,
            class_level_id=grade,
        ),
        tokens=TokenPair(
            access_token=access_token.access_token,
            refresh_token=refresh_token.access_token,
            token_type="bearer",
            expires_at=access_token.expires_at,
            refresh_expires_at=refresh_token.expires_at,
        ),
    )


# -----------------------
# Login Endpoint
# -----------------------
@router.post("/login", response_model=AuthResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["login"][0])
async def login(
    request: Request,
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
    session: Session = Depends(get_session),
):
    username = sanitize_phone(username)
    password = sanitize_string(password)

    user = session.exec(select(User).where(User.phone == username)).first()
    if not user or not user.verify_password(password):
        raise HTTPException(status_code=401, detail="Incorrect phone or password")

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return AuthResponse(
        user=UserResponse(
            id=user.id,
            phone=user.phone,
            board_id=user.board_id,
            medium_id=user.medium_id,
            class_level_id=user.class_level_id,
        ),
        tokens=TokenPair(
            access_token=access_token.access_token,
            refresh_token=refresh_token.access_token,
            token_type="bearer",
            expires_at=access_token.expires_at,
            refresh_expires_at=refresh_token.expires_at,
        ),
    )


# -----------------------
# Login via OTP
# -----------------------
@router.post("/login-otp", response_model=AuthResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["login"][0])
async def login_otp(
    request: Request,
    phone: str = Body(..., embed=True),
    otp: str = Body(..., embed=True),
    otp_secret: str = Body(..., embed=True),
    session: Session = Depends(get_session),
):
    phone = sanitize_phone(phone)
    otp = sanitize_string(otp)
    otp_secret = sanitize_string(otp_secret)

    if not validate_otp(otp, otp_secret):
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")

    user = session.exec(select(User).where(User.phone == phone)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return AuthResponse(
        user=UserResponse(id=user.id, phone=user.phone),
        tokens=TokenPair(
            access_token=access_token.access_token,
            refresh_token=refresh_token.access_token,
            token_type="bearer",
            expires_at=access_token.expires_at,
            refresh_expires_at=refresh_token.expires_at,
        ),
    )


# -----------------------
# Refresh Token Endpoint
# -----------------------
@router.post("/refresh", response_model=TokenPair)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["refresh"][0])
async def refresh_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session),
):
    token = sanitize_string(credentials.credentials)
    user_id = verify_refresh_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = session.get(User, int(user_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return TokenPair(
        access_token=access_token.access_token,
        refresh_token=refresh_token.access_token,
        token_type="bearer",
        expires_at=access_token.expires_at,
        refresh_expires_at=refresh_token.expires_at,
    )


# -----------------------
# OTP Utilities
# -----------------------
def generate_otp():
    secret = base64.b32encode(os.urandom(20)).decode("utf-8")
    totp = pyotp.TOTP(secret, interval=300)  # 5 minutes
    otp = totp.now()
    return otp, secret


def validate_otp(otp: str, secret: str) -> bool:
    totp = pyotp.TOTP(secret, interval=300)
    return totp.verify(otp, valid_window=1)


def send_sms(phone: str, message: str):
    numbers = f"[91{phone}]"
    payload = {
        "apikey": settings.TEXTLOCAL_API_KEY,
        "numbers": numbers,
        "message": message,
        "sender": settings.TEXTLOCAL_SENDER,
    }
    # Actual sending logic commented out
    # encoded_payload = urllib.parse.urlencode(payload)
    # requests.post("https://api.textlocal.in/send/", data=encoded_payload)
    return True


# -----------------------
# Check User & Send OTP
# -----------------------
@router.post("/check-user")
async def check_user_exists(
    phone: str = Body(..., embed=True),
    app_signature: str = Body(default=""),
    session: Session = Depends(get_session),
):
    phone = sanitize_phone(phone)
    user = session.exec(select(User).where(User.phone == phone)).first()

    if user:
        return {
            "status": "success",
            "message": "User already exists",
            "data": {"exists": True},
        }

    otp, secret = generate_otp()
    print(otp)
    send_sms(phone, f"{otp} is your OTP. {app_signature}")
    return {
        "status": "success",
        "message": "OTP sent",
        "data": {"exists": False, "otp_secret": secret},
    }


@router.post("/send-otp")
async def send_otp(
    phone: str = Body(..., embed=True),
    app_signature: str = Body(default=""),
):
    phone = sanitize_phone(phone)
    otp, secret = generate_otp()
    print(otp)
    sms_text = f"{otp} is the OTP to verify your mobile number for Cherri Learn. This OTP is valid for 5 minutes. pls do not share it with anyone.\n {app_signature}"
    send_sms(phone, sms_text)
    return {
        "status": "success",
        "message": "OTP sent",
        "data": {"otp_secret": secret},
    }


@router.post("/verify-otp")
async def verify_otp_endpoint(
    otp: str = Body(..., embed=True),
    otp_secret: str = Body(..., embed=True),
):
    if not validate_otp(otp, otp_secret):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    return {"status": "success", "message": "OTP verified"}
