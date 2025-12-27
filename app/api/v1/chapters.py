from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from typing import Dict, List, Optional
from loguru import logger

from app.models import Chapter, Subject
from app.models import ChapterCreate, ChapterRead, ChapterUpdate
from app.models import (
    LLMTextbook, AdditionalNotes, LLMImage, LLMNote, QAPattern,
    StudentTextbook, StudentNotes, StudentVideo
)
from app.services.database import get_session
from app.utils.cleanup import cleanup_chapter_resources

router = APIRouter()


@router.get("/{chapter_id}", response_model=Dict[str, ChapterRead])
async def get_chapter(chapter_id: int, session: AsyncSession = Depends(get_session)):
    try:
        # Fetch the chapter
        db_chapter = await session.get(Chapter, chapter_id)
        if not db_chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        # Fetch the associated subject
        subject = await session.get(Subject, db_chapter.subject_id)
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

        return {
            "data": ChapterRead(
                id=db_chapter.id,
                name=db_chapter.name,
                subject_id=db_chapter.subject_id,
                subject_name=subject.name,
                chapter_number=db_chapter.chapter_number,
                enabled=db_chapter.enabled,
                is_premium=db_chapter.is_premium,
                icon_url=db_chapter.icon_url,
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chapter {chapter_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=Dict[str, List[ChapterRead]])
async def get_chapters(
    subject_id: Optional[int] = Query(None), session: AsyncSession = Depends(get_session)
):
    try:
        query = select(Chapter, Subject).join(Subject)
        if subject_id:
            query = query.where(Chapter.subject_id == subject_id)
        _result = await session.exec(
            query.order_by(Subject.name, Chapter.chapter_number))
        results = _result.all()
        chapters = [
            ChapterRead(
                id=chapter.id,
                name=chapter.name,
                subject_id=chapter.subject_id,
                subject_name=subject.name,
                chapter_number=chapter.chapter_number,
                enabled=chapter.enabled,
                is_premium=chapter.is_premium,
                icon_url=chapter.icon_url,
            )
            for chapter, subject in results
        ]
        return {"data": chapters}
    except Exception as e:
        logger.error(f"Error getting chapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, ChapterRead])
async def create_chapter(
    chapter: ChapterCreate, session: AsyncSession = Depends(get_session)
):
    try:
        subject = await session.get(Subject, chapter.subject_id)
        if not subject:
            raise HTTPException(status_code=400, detail="Subject not found")

        _result = await session.exec(
            select(Chapter).where(
                Chapter.name == chapter.name, Chapter.subject_id == chapter.subject_id
            ))

        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Chapter already exists for this subject"
            )

        db_chapter = Chapter.model_validate(chapter)
        session.add(db_chapter)
        await session.commit()
        await session.refresh(db_chapter)
        return {
            "data": ChapterRead(
                id=db_chapter.id,
                name=db_chapter.name,
                subject_id=db_chapter.subject_id,
                subject_name=subject.name,
                chapter_number=db_chapter.chapter_number,
                enabled=db_chapter.enabled,
                is_premium=db_chapter.is_premium,
                icon_url=db_chapter.icon_url,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error creating chapter: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{chapter_id}", response_model=Dict[str, ChapterRead])
async def update_chapter(
    chapter_id: int,
    chapter_data: ChapterUpdate,
    session: AsyncSession = Depends(get_session),
):
    try:
        db_chapter = await session.get(Chapter, chapter_id)
        if not db_chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        subject = await session.get(Subject, chapter_data.subject_id)
        if not subject:
            raise HTTPException(status_code=400, detail="Subject not found")

        # Check for duplicate name within the same subject (excluding current chapter)
        _result = await session.exec(
            select(Chapter).where(
                Chapter.name == chapter_data.name,
                Chapter.subject_id == chapter_data.subject_id,
                Chapter.id != chapter_id,
            ))
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Another chapter with this name already exists"
            )

        # Update only provided fields
        update_data = chapter_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_chapter, field, value)

        session.add(db_chapter)
        await session.commit()
        await session.refresh(db_chapter)

        return {
            "data": ChapterRead(
                id=db_chapter.id,
                name=db_chapter.name,
                subject_id=db_chapter.subject_id,
                subject_name=subject.name,
                chapter_number=db_chapter.chapter_number,
                enabled=db_chapter.enabled,
                is_premium=db_chapter.is_premium,
                icon_url=db_chapter.icon_url,
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error updating chapter: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{chapter_id}")
async def delete_chapter(chapter_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a chapter and all its related resources (files, embeddings, DB records)."""
    try:
        chapter = await session.get(Chapter, chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        # Clean up all related resources (files and embeddings) before deleting the chapter
        cleanup_stats = await cleanup_chapter_resources(session, chapter_id)
        
        # Delete the chapter (cascade_delete relationships will handle related DB records)
        await session.delete(chapter)
        await session.commit()

        return {
            "message": "Chapter deleted successfully",
            "cleanup_stats": {
                "files_deleted": cleanup_stats["files_deleted"],
                "embeddings_deleted": cleanup_stats["embeddings_deleted"],
                "errors": cleanup_stats["errors"] if cleanup_stats["errors"] else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error deleting chapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chapter_id}/llm-resources")
async def get_chapter_llm_resources(
    chapter_id: int, session: AsyncSession = Depends(get_session)
):
    """Get all LLM resources for a specific chapter."""
    try:
        # Verify chapter exists
        chapter = await session.get(Chapter, chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        # Fetch all LLM resources
        _result = await session.exec(
            select(LLMTextbook).where(LLMTextbook.chapter_id == chapter_id))
        textbooks = _result.all()
        _result = await session.exec(
            select(AdditionalNotes).where(AdditionalNotes.chapter_id == chapter_id))

        additional_notes = _result.all()
        _result = await session.exec(
            select(LLMImage).where(LLMImage.chapter_id == chapter_id))

        images = _result.all()
        _result = await session.exec(
            select(LLMNote).where(LLMNote.chapter_id == chapter_id))

        llm_notes = _result.all()
        _result = await session.exec(
            select(QAPattern).where(QAPattern.chapter_id == chapter_id))

        qa_patterns = _result.all()
        return {
            "data": {
                "chapter_id": chapter_id,
                "chapter_name": chapter.name,
                "textbooks": [t.dict() for t in textbooks],
                "additional_notes": [n.dict() for n in additional_notes],
                "images": [img.dict() for img in images],
                "llm_notes": [note.dict() for note in llm_notes],
                "qa_patterns": [qa.dict() for qa in qa_patterns],
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching LLM resources for chapter {chapter_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chapter_id}/student-content")
async def get_chapter_student_content(
    chapter_id: int, session: AsyncSession = Depends(get_session)
):
    """Get all student content for a specific chapter."""
    try:
        # Verify chapter exists
        chapter = await session.get(Chapter, chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        # Fetch all student content
        _result = await session.exec(
            select(StudentTextbook).where(StudentTextbook.chapter_id == chapter_id))
        textbooks = _result.all()
        _result = await session.exec(
            select(StudentNotes).where(StudentNotes.chapter_id == chapter_id))

        notes = _result.all()
        _result = await session.exec(
            select(StudentVideo).where(StudentVideo.chapter_id == chapter_id))

        videos = _result.all()
        return {
            "data": {
                "chapter_id": chapter_id,
                "chapter_name": chapter.name,
                "textbooks": [t.dict() for t in textbooks],
                "notes": [n.dict() for n in notes],
                "videos": [v.dict() for v in videos],
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching student content for chapter {chapter_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chapter_id}/all-content")
async def get_chapter_all_content(
    chapter_id: int, session: AsyncSession = Depends(get_session)
):
    """Get all content (both LLM resources and student content) for a specific chapter."""
    try:
        # Verify chapter exists
        chapter = await session.get(Chapter, chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        # Fetch all LLM resources
        _result = await session.exec(
            select(LLMTextbook).where(LLMTextbook.chapter_id == chapter_id))
        llm_textbooks = _result.all()
        _result = await session.exec(
            select(AdditionalNotes).where(AdditionalNotes.chapter_id == chapter_id))

        additional_notes = _result.all()
        _result = await session.exec(
            select(LLMImage).where(LLMImage.chapter_id == chapter_id))

        images = _result.all()
        _result = await session.exec(
            select(LLMNote).where(LLMNote.chapter_id == chapter_id))

        llm_notes = _result.all()
        _result = await session.exec(
            select(QAPattern).where(QAPattern.chapter_id == chapter_id))

        qa_patterns = _result.all()
        # Fetch all student content
        _result = await session.exec(
            select(StudentTextbook).where(StudentTextbook.chapter_id == chapter_id))
        student_textbooks = _result.all()
        _result = await session.exec(
            select(StudentNotes).where(StudentNotes.chapter_id == chapter_id))

        student_notes = _result.all()
        _result = await session.exec(
            select(StudentVideo).where(StudentVideo.chapter_id == chapter_id))

        student_videos = _result.all()
        return {
            "data": {
                "chapter_id": chapter_id,
                "chapter_name": chapter.name,
                "llm_resources": {
                    "textbooks": [t.dict() for t in llm_textbooks],
                    "additional_notes": [n.dict() for n in additional_notes],
                    "images": [img.dict() for img in images],
                    "llm_notes": [note.dict() for note in llm_notes],
                    "qa_patterns": [qa.dict() for qa in qa_patterns],
                },
                "student_content": {
                    "textbooks": [t.dict() for t in student_textbooks],
                    "notes": [n.dict() for n in student_notes],
                    "videos": [v.dict() for v in student_videos],
                },
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching all content for chapter {chapter_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
