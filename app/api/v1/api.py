"""API v1 router configuration.

This module sets up the main API router and includes all sub-routers for different
endpoints like authentication and chatbot functionality.
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.agent import router as agent_router
from app.api.v1.files import router as files_router
from app.api.v1.visualizations import router as visualizations_router
from app.api.v1.session import router as session_router
from loguru import logger


api_router = APIRouter()

# Include routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(session_router, prefix="/session", tags=["session"])
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(files_router, prefix="/files", tags=["files"])
api_router.include_router(
    visualizations_router, prefix="/visualizations", tags=["visualizations"]
)


@api_router.get("/health")
def health_check():
    """Health check endpoint.

    Returns:
        dict: Health status information.
    """
    logger.info("health_check_called")
    return {"status": "healthy", "version": settings.VERSION}
