from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from typing import Dict, List, Optional
from loguru import logger

from app.models import Subject, Medium
from app.models import SubjectCreate, SubjectRead
from app.services.database import get_session
from app.utils.cleanup import cleanup_subject_resources

router = APIRouter()


@router.get("/", response_model=Dict[str, List[SubjectRead]])
async def get_subjects(
    medium_id: Optional[int] = Query(None), session: AsyncSession = Depends(get_session)
):
    try:
        query = select(Subject, Medium).join(Medium)
        if medium_id:
            query = query.where(Subject.medium_id == medium_id)
        _result = await session.exec(query.order_by(Medium.name, Subject.name))
        results = _result.all()

        subjects = [
            SubjectRead(
                id=subject.id,
                name=subject.name,
                medium_id=subject.medium_id,
                medium_name=medium.name,
            )
            for subject, medium in results
        ]
        return {"data": subjects}
    except Exception as e:
        logger.error(f"Error getting subjects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, SubjectRead])
async def create_subject(
    subject: SubjectCreate, session: AsyncSession = Depends(get_session)
):
    try:
        medium = await session.get(Medium, subject.medium_id)
        if not medium:
            raise HTTPException(status_code=400, detail="Medium not found")

        _result = await session.exec(
            select(Subject).where(
                Subject.name == subject.name, Subject.medium_id == subject.medium_id
            )
        )
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Subject already exists for this medium"
            )

        db_subject = Subject.model_validate(subject)
        session.add(db_subject)
        await session.commit()
        await session.refresh(db_subject)
        return {
            "data": SubjectRead(
                id=db_subject.id,
                name=db_subject.name,
                medium_id=db_subject.medium_id,
                medium_name=medium.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error creating subject: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{subject_id}", response_model=Dict[str, SubjectRead])
async def update_subject(
    subject_id: int,
    subject_data: SubjectCreate,
    session: AsyncSession = Depends(get_session),
):
    try:
        db_subject = await session.get(Subject, subject_id)
        if not db_subject:
            raise HTTPException(status_code=404, detail="Subject not found")

        medium = await session.get(Medium, subject_data.medium_id)
        if not medium:
            raise HTTPException(status_code=400, detail="Medium not found")

        # Check for duplicate name within same medium (excluding current subject)
        _result = await session.exec(
            select(Subject).where(
                Subject.name == subject_data.name,
                Subject.medium_id == subject_data.medium_id,
                Subject.id != subject_id,
            ))
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Another subject with this name already exists for this medium",
            )

        # Update name and medium
        db_subject.name = subject_data.name
        db_subject.medium_id = subject_data.medium_id

        session.add(db_subject)
        await session.commit()
        await session.refresh(db_subject)

        return {
            "data": SubjectRead(
                id=db_subject.id,
                name=db_subject.name,
                medium_id=db_subject.medium_id,
                medium_name=medium.name,
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error updating subject: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{subject_id}")
async def delete_subject(subject_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a subject and all its related resources (chapters, files, embeddings, DB records)."""
    try:
        subject = await session.get(Subject, subject_id)
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

        # Clean up all related resources for all chapters in this subject
        cleanup_stats = await cleanup_subject_resources(session, subject_id)

        # Delete the subject (cascade_delete relationships will handle related DB records)
        await session.delete(subject)
        await session.commit()

        return {
            "message": "Subject deleted successfully",
            "cleanup_stats": {
                "chapters_cleaned": cleanup_stats["chapters_cleaned"],
                "total_files_deleted": cleanup_stats["total_files_deleted"],
                "total_embeddings_deleted": cleanup_stats["total_embeddings_deleted"],
                "errors": cleanup_stats["errors"] if cleanup_stats["errors"] else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error deleting subject: {e}")
        raise HTTPException(status_code=500, detail=str(e))
