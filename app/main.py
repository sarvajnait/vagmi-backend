from fastapi import FastAPI
from loguru import logger
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.api.v1.api import api_router
from app.core.config import settings
from app.models import *
from app.schemas import *
from fastapi.middleware.cors import CORSMiddleware
from app.services.database import engine

# Load environment variables
load_dotenv()


async def _cleanup_stale_jobs():
    """
    On startup, mark any jobs left in 'running' state as 'failed'.
    These are jobs that were interrupted by a server restart mid-execution.
    Also resets the associated record status so the frontend never shows
    a permanently stuck 'processing' badge.
    """
    from sqlmodel import select
    from app.services.database import async_session_maker
    from app.models import ActivityGenerationJob, StudentTextbook, StudentNotes, ChapterArtifact

    async with async_session_maker() as session:
        result = await session.exec(
            select(ActivityGenerationJob).where(ActivityGenerationJob.status == "running")
        )
        stale_jobs = result.all()

        if not stale_jobs:
            return

        for job in stale_jobs:
            job.status = "failed"
            job.error = "Server restarted while job was running"
            session.add(job)

            # Reset the associated resource status
            try:
                if job.job_type == "audio_generation":
                    resource_type = (job.payload or {}).get("resource_type")
                    resource_id = (job.payload or {}).get("resource_id")
                    if resource_type == "textbook" and resource_id:
                        record = await session.get(StudentTextbook, resource_id)
                    elif resource_type == "notes" and resource_id:
                        record = await session.get(StudentNotes, resource_id)
                    else:
                        record = None
                    if record and record.audio_status == "processing":
                        record.audio_status = "failed"
                        session.add(record)

                elif job.job_type == "textbook_process":
                    chapter_id = (job.payload or {}).get("chapter_id")
                    if chapter_id:
                        art_result = await session.exec(
                            select(ChapterArtifact).where(
                                ChapterArtifact.chapter_id == chapter_id,
                                ChapterArtifact.artifact_type == "chapter_summary",
                                ChapterArtifact.status == "processing",
                            )
                        )
                        artifact = art_result.first()
                        if artifact:
                            artifact.status = "failed"
                            artifact.error = "Server restarted while processing"
                            session.add(artifact)
            except Exception as e:
                logger.warning(f"Could not reset resource for stale job {job.id}: {e}")

        await session.commit()
        logger.info(f"Cleaned up {len(stale_jobs)} stale job(s) from previous run")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _cleanup_stale_jobs()
    yield
    await engine.dispose()
    logger.info("Shutting down application")


app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.get("/")
def root():
    return {
        "message": "Welcome to VAGMI",
    }
