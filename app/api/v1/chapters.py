from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from typing import Dict, List, Optional
from loguru import logger

from app.models import Chapter, Subject
from app.models import ChapterCreate, ChapterRead
from app.services.database import get_session

router = APIRouter()

@router.get("/", response_model=Dict[str, List[ChapterRead]])
async def get_chapters(subject_id: Optional[int] = Query(None), session: Session = Depends(get_session)):
    try:
        query = select(Chapter, Subject).join(Subject)
        if subject_id:
            query = query.where(Chapter.subject_id == subject_id)
        results = session.exec(query.order_by(Subject.name, Chapter.name)).all()

        chapters = [
            ChapterRead(
                id=chapter.id,
                name=chapter.name,
                subject_id=chapter.subject_id,
                subject_name=subject.name,
            )
            for chapter, subject in results
        ]
        return {"data": chapters}
    except Exception as e:
        logger.error(f"Error getting chapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Dict[str, ChapterRead])
async def create_chapter(chapter: ChapterCreate, session: Session = Depends(get_session)):
    try:
        subject = session.get(Subject, chapter.subject_id)
        if not subject:
            raise HTTPException(status_code=400, detail="Subject not found")

        existing = session.exec(
            select(Chapter).where(Chapter.name == chapter.name, Chapter.subject_id == chapter.subject_id)
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Chapter already exists for this subject")

        db_chapter = Chapter.model_validate(chapter)
        session.add(db_chapter)
        session.commit()
        session.refresh(db_chapter)
        return {
            "data": ChapterRead(
                id=db_chapter.id,
                name=db_chapter.name,
                subject_id=db_chapter.subject_id,
                subject_name=subject.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating chapter: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{chapter_id}")
async def delete_chapter(chapter_id: int, session: Session = Depends(get_session)):
    try:
        chapter = session.get(Chapter, chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")
        session.delete(chapter)
        session.commit()
        return {"message": "Chapter deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting chapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))
