"""Authentication and authorization endpoints for the API.

This module provides endpoints for user registration, login, session management,
and token verification.
"""

import uuid
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
)
from fastapi.security import (
    HTTPBearer,
)

from app.core.logging import logger
from app.models.session import Session
from app.models.user import User
from app.schemas.session import SessionResponse, AutoNameSessionRequest
from .auth import get_current_user
from app.services.database import DatabaseService
from app.utils.sanitization import (
    sanitize_string,
)
from app.core.langgraph.general_agent import GeneralAgent
from app.utils.files import delete_session_folder

router = APIRouter()
security = HTTPBearer()
db_service = DatabaseService()


async def get_session_by_id(
    session_id: str, user: User = Depends(get_current_user)
) -> Session:
    """Get the session details by ID and verify it belongs to the authenticated user.

    Args:
        session_id: The ID of the session to retrieve.
        user: The authenticated user (from JWT).

    Returns:
        Session: The session details.

    Raises:
        HTTPException: If the session is missing or does not belong to the user.
    """
    try:
        if not session_id:
            logger.error("session_id_missing")
            raise HTTPException(
                status_code=400,
                detail="Session ID is required",
            )

        # Sanitize session_id before using it
        sanitized_session_id = sanitize_string(session_id)

        # Fetch the session from DB
        session = await db_service.get_session(sanitized_session_id)
        if session is None or session.user_id != user.id:
            logger.error(
                "session_not_found_or_not_owned",
                session_id=sanitized_session_id,
                user_id=user.id,
            )
            raise HTTPException(
                status_code=404,
                detail="Session not found or not authorized",
            )

        return session

    except ValueError as ve:
        logger.error("session_validation_failed", error=str(ve), exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Invalid session ID format",
        )


@router.post("", response_model=SessionResponse)
async def create_session(
    user: User = Depends(get_current_user),
):
    try:
        session_id = str(uuid.uuid4())

        # Always set default name
        session_name = "New Session"

        # Create in DB
        session = await db_service.create_session(
            session_id=session_id,
            user_id=user.id,
            name=session_name,
        )

        return SessionResponse(session_id=session_id, name=session.name)
    except Exception as e:
        logger.error("session_creation_failed", error=str(e), user_id=user.id)
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.patch("/{session_id}/name", response_model=SessionResponse)
async def update_session_name(
    session_id: str,
    name: str = Form(...),
    session: Session = Depends(get_session_by_id),
):
    """Update a session's name.

    Args:
        session_id: The ID of the session to update
        name: The new name for the session
        session: The verified sessiond details

    Returns:
        SessionResponse: The updated session information
    """
    try:
        # Sanitize inputs
        sanitized_session_id = sanitize_string(session_id)
        sanitized_name = sanitize_string(name)
        sanitized_current_session = sanitize_string(session.id)

        # Verify the session ID matches the authenticated session
        if sanitized_session_id != sanitized_current_session:
            raise HTTPException(status_code=403, detail="Cannot modify other sessions")

        # Update the session name
        session = await db_service.update_session_name(
            sanitized_session_id, sanitized_name
        )

        return SessionResponse(session_id=sanitized_session_id, name=session.name)
    except ValueError as ve:
        logger.error(
            "session_update_validation_failed",
            error=str(ve),
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(status_code=422, detail=str(ve))


@router.post("/{session_id}/auto-name", response_model=SessionResponse)
async def auto_name_session(
    session_id: str,
    body: AutoNameSessionRequest,
    session: Session = Depends(get_session_by_id),
):
    """Generate and update a session name using the initial user prompt."""
    try:
        sanitized_session_id = sanitize_string(session_id)
        sanitized_current_session = sanitize_string(session.id)
        sanitized_prompt = sanitize_string(body.prompt)

        if sanitized_session_id != sanitized_current_session:
            raise HTTPException(status_code=403, detail="Cannot modify other sessions")

        general_agent = GeneralAgent()
        generated_name = await general_agent.generate_session_name(sanitized_prompt)

        updated = await db_service.update_session_name(
            sanitized_session_id, generated_name
        )

        return SessionResponse(session_id=sanitized_session_id, name=updated.name)

    except ValueError as ve:
        logger.error(
            "auto_name_session_validation_failed",
            error=str(ve),
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(
            "auto_name_session_failed",
            error=str(e),
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to auto-generate session name"
        )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str, session: Session = Depends(get_session_by_id)
):
    """Delete a session for the authenticated user.

    Args:
        session_id: The ID of the session to delete
        session: The verified sessiond details

    Returns:
        None
    """
    try:
        # Sanitize inputs
        sanitized_session_id = sanitize_string(session_id)
        sanitized_current_session = sanitize_string(session.id)

        # Verify the session ID matches the authenticated session
        if sanitized_session_id != sanitized_current_session:
            raise HTTPException(status_code=403, detail="Cannot delete other sessions")

        # Delete the session
        await db_service.delete_session(sanitized_session_id)

        delete_session_folder(sanitized_session_id)

        logger.info("session_deleted", session_id=session_id, user_id=session.user_id)

        return {"message": f"Session '{sanitized_session_id}' deleted successfully."}

    except ValueError as ve:
        logger.error(
            "session_deletion_validation_failed",
            error=str(ve),
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(status_code=422, detail=str(ve))


@router.delete("", summary="Delete all sessions for the authenticated user")
async def delete_all_sessions(user: User = Depends(get_current_user)):
    deleted_ids = await db_service.delete_user_sessions(user.id)

    for sid in deleted_ids:
        delete_session_folder(sid)

    return {"message": f"Deleted {len(deleted_ids)} sessions."}


@router.get("", response_model=List[SessionResponse])
async def get_user_sessions(user: User = Depends(get_current_user)):
    """Get all session IDs for the authenticated user.

    Args:
        user: The authenticated user

    Returns:
        List[SessionResponse]: List of session IDs
    """
    try:
        sessions = await db_service.get_user_sessions(user.id)
        return [
            SessionResponse(
                session_id=sanitize_string(session.id),
                name=sanitize_string(session.name),
            )
            for session in sessions
        ]
    except ValueError as ve:
        logger.error(
            "get_sessions_validation_failed",
            user_id=user.id,
            error=str(ve),
            exc_info=True,
        )
        raise HTTPException(status_code=422, detail=str(ve))
