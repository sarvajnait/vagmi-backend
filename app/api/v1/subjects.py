from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from typing import Dict, List, Optional
from loguru import logger

from app.models import Subject, Medium
from app.models import SubjectCreate, SubjectRead
from app.services.database import get_session

router = APIRouter()


@router.get("/", response_model=Dict[str, List[SubjectRead]])
async def get_subjects(
    medium_id: Optional[int] = Query(None), session: Session = Depends(get_session)
):
    try:
        query = select(Subject, Medium).join(Medium)
        if medium_id:
            query = query.where(Subject.medium_id == medium_id)
        results = session.exec(query.order_by(Medium.name, Subject.name)).all()

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
    subject: SubjectCreate, session: Session = Depends(get_session)
):
    try:
        medium = session.get(Medium, subject.medium_id)
        if not medium:
            raise HTTPException(status_code=400, detail="Medium not found")

        existing = session.exec(
            select(Subject).where(
                Subject.name == subject.name, Subject.medium_id == subject.medium_id
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Subject already exists for this medium"
            )

        db_subject = Subject.model_validate(subject)
        session.add(db_subject)
        session.commit()
        session.refresh(db_subject)
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
        session.rollback()
        logger.error(f"Error creating subject: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{subject_id}")
async def delete_subject(subject_id: int, session: Session = Depends(get_session)):
    try:
        subject = session.get(Subject, subject_id)
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")
        session.delete(subject)
        session.commit()
        return {"message": "Subject deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting subject: {e}")
        raise HTTPException(status_code=500, detail=str(e))
