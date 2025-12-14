from fastapi import APIRouter, Depends, Body, Request, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import select, Session
from loguru import logger
from app.utils.otp import validate_otp

from app.models.admin import Admin
from app.schemas.auth import (
    AdminCreate,
    TokenPair,
    AdminResponse,
    AdminAuthResponse,
)
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
) -> Admin:
    try:
        token = sanitize_string(credentials.credentials)
        user_id = verify_token(token)
        if not user_id:
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials"
            )

        user = session.get(Admin, int(user_id))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except ValueError as ve:
        logger.error("token_validation_failed", error=str(ve), exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid token format")


# -----------------------
# Register Endpoint
# -----------------------
@router.post("/register", response_model=AdminAuthResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["register"][0])
async def register_user(
    request: Request, user_data: AdminCreate, session: Session = Depends(get_session)
):

    phone = sanitize_phone(user_data.phone)

    existing_user = session.exec(select(Admin).where(Admin.phone == phone)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Phone already registered")

    user = Admin(
        phone=phone,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return AdminAuthResponse(
        user=AdminResponse(
            id=user.id,
            phone=user.phone,
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
@router.post("/login-otp", response_model=AdminAuthResponse)
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

    user = session.exec(select(Admin).where(Admin.phone == phone)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return AdminAuthResponse(
        user=AdminResponse(
            id=user.id,
            phone=user.phone,
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

    user = session.get(Admin, int(user_id))
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


@router.post("/check-user")
async def check_user_exists(
    phone: str = Body(..., embed=True),
    app_signature: str = Body(default=""),
    session: Session = Depends(get_session),
):
    phone = sanitize_phone(phone)
    user = session.exec(select(Admin).where(Admin.phone == phone)).first()
    if user:
        return {
            "status": "success",
            "message": "Admin already exists",
            "data": {"exists": True},
        }

    return {
        "status": "success",
        "message": "Admin does not exist",
        "data": {"exists": False},
    }
