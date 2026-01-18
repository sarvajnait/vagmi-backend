from fastapi import APIRouter
from loguru import logger
from app.core.config import settings

from app.api.v1.agent import router as agent_router
from app.api.v1.transcribe import router as transcribe_router
from app.api.v1.hierarchy import router as hierarchy_router
from app.api.v1.class_levels import router as class_levels_router
from app.api.v1.boards import router as boards_router
from app.api.v1.mediums import router as mediums_router
from app.api.v1.subjects import router as subjects_router
from app.api.v1.chapters import router as chapters_router
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.admin.auth import router as admin_router
from app.api.v1.admin.users import router as admin_users_router
from app.api.v1.files import router as files_router
from app.api.v1.llm_resources import router as llm_resources_router
from app.api.v1.student_content import router as student_content_router
from app.api.v1.subscriptions import router as subscriptions_router
from app.api.v1.activities import router as activities_router
from app.api.v1.activity_groups import router as activity_groups_router

api_router = APIRouter()

# Include routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
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
api_router.include_router(files_router, prefix="/files", tags=["files"])
api_router.include_router(
    llm_resources_router, prefix="/llm_resources", tags=["llm_resources"]
)
api_router.include_router(
    student_content_router, prefix="/student_content", tags=["student_content"]
)
api_router.include_router(
    subscriptions_router, prefix="/subscriptions", tags=["subscriptions"]
)
api_router.include_router(activities_router, prefix="/activities", tags=["activities"])
api_router.include_router(activity_groups_router, prefix="/activity-groups", tags=["activity-groups"])
api_router.include_router(admin_router, prefix="/admin/auth", tags=["admin"])
api_router.include_router(admin_users_router, prefix="/admin/users", tags=["admin"])


@api_router.get("/health")
def health_check():
    logger.info("health_check_called")
    return {"status": "healthy", "version": settings.VERSION}
