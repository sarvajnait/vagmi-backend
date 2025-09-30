from fastapi import APIRouter
from loguru import logger
from app.core.config import settings

from app.api.v1.agent import router as agent_router
from app.api.v1.documents import router as documents_router
from app.api.v1.transcribe import router as transcribe_router
from app.api.v1.hierarchy import router as hierarchy_router
from app.api.v1.class_levels import router as class_levels_router
from app.api.v1.boards import router as boards_router
from app.api.v1.mediums import router as mediums_router
from app.api.v1.subjects import router as subjects_router
from app.api.v1.chapters import router as chapters_router
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router

api_router = APIRouter()

# Include routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(transcribe_router, prefix="/transcribe", tags=["transcribe"])
api_router.include_router(
    hierarchy_router, prefix="/hierarchy-options", tags=["hierarchy"]
)
api_router.include_router(
    class_levels_router, prefix="/class-levels", tags=["class-levels"]
)
api_router.include_router(boards_router, prefix="/boards", tags=["boards"])
api_router.include_router(mediums_router, prefix="/mediums", tags=["mediums"])
api_router.include_router(subjects_router, prefix="/subjects", tags=["subjects"])
api_router.include_router(chapters_router, prefix="/chapters", tags=["chapters"])


@api_router.get("/health")
def health_check():
    logger.info("health_check_called")
    return {"status": "healthy", "version": settings.VERSION}
