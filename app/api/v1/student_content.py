from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from loguru import logger
from sqlmodel import Session
from app.models import (
    StudentTextbook,
    StudentNotes,
    StudentVideo,
    PreviousYearQuestionPaper,
    Subject,
)
from app.services.database import get_session
from app.utils.files import upload_to_do, delete_from_do

router = APIRouter()


# Textbook Endpoints
# ============================================================
@router.post("/textbook")
async def upload_textbook(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    session: Session = Depends(get_session),
):
    """Upload a textbook directly to DigitalOcean and add to DB/vector store."""
    try:

    
        do_path = f"chapters/{chapter_id}/student-content/textbooks"
        file_url = upload_to_do(file, do_path)

        textbook = StudentTextbook(
            chapter_id=chapter_id,
            title=file.filename,
            description=None,
            file_url=file_url,
        )
        session.add(textbook)
        session.commit()
        session.refresh(textbook)

        return {
            "message": "Document uploaded",
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/textbook")
async def get_textbooks(
    chapter_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """Get textbooks filtered by chapter."""
    try:
        query = session.query(StudentTextbook)
        if chapter_id is not None:
            query = query.filter(StudentTextbook.chapter_id == chapter_id)
        textbooks = query.all()
        return {"data": [t.dict() for t in textbooks]}

    except Exception as e:
        logger.error(f"Error fetching textbooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/textbook/{textbook_id}")
async def delete_textbook(textbook_id: int, session: Session = Depends(get_session)):
    """Delete a textbook from DB, vector store, and DigitalOcean Spaces."""
    try:
        textbook = session.get(StudentTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        if textbook.file_url:
            delete_from_do(textbook.file_url)

        session.delete(textbook)
        session.commit()

        return {"message": f"Textbook '{textbook.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting textbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Notes Endpoints
# ============================================================
@router.post("/notes")
async def upload_note(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """
    Upload a note file directly to DigitalOcean and add to DB.
    """
    try:
        do_path = f"chapters/{chapter_id}/student-content/notes"
        file_url = upload_to_do(file, do_path)

        note = StudentNotes(
            chapter_id=chapter_id,
            title=title,
            description=description,
            file_url=file_url,
        )
        session.add(note)
        session.commit()
        session.refresh(note)

        return {"message": "Note uploaded", "data": note.dict()}

    except Exception as e:
        logger.error(f"Error uploading note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notes")
async def get_notes(
    chapter_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """Get all notes or filter by chapter."""
    try:
        query = session.query(StudentNotes)
        if chapter_id is not None:
            query = query.filter(StudentNotes.chapter_id == chapter_id)
        notes = query.all()
        return {"data": [n.dict() for n in notes]}

    except Exception as e:
        logger.error(f"Error fetching notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/notes/{note_id}")
async def delete_note(note_id: int, session: Session = Depends(get_session)):
    """Delete a note from DB and DigitalOcean Spaces."""
    try:
        note = session.get(StudentNotes, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        if note.file_url:
            try:
                delete_from_do(note.file_url)
            except Exception as e:
                logger.error(f"Error deleting note file from DO: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        session.delete(note)
        session.commit()

        return {"message": f"Note '{note.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Video Endpoints
# ============================================================
@router.post("/videos")
async def upload_video(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """
    Upload a video file directly to DigitalOcean and add to DB.
    """
    try:
        do_path = f"chapters/{chapter_id}/student-content/videos"
        file_url = upload_to_do(file, do_path)

        video = StudentVideo(
            chapter_id=chapter_id,
            title=title,
            description=description,
            file_url=file_url,
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        return {"message": "Video uploaded", "data": video.model_dump()}

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos")
async def get_videos(
    chapter_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """Get all videos or filter by chapter."""
    try:
        query = session.query(StudentVideo)
        if chapter_id is not None:
            query = query.filter(StudentVideo.chapter_id == chapter_id)
        notes = query.all()
        return {"data": [n.dict() for n in notes]}

    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/videos/{video_id}")
async def delete_video(video_id: int, session: Session = Depends(get_session)):
    """Delete a video from DB and DigitalOcean Spaces."""
    try:
        video = session.get(StudentVideo, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if video.file_url:
            try:
                delete_from_do(video.file_url)
            except Exception as e:
                logger.error(f"Error deleting video file from DO: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        session.delete(video)
        session.commit()

        return {"message": f"Video '{video.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Previous Year Question Papers (Subject-level)
# ============================================================
@router.post("/previous-year-question-papers")
async def upload_previous_year_question_paper(
    file: UploadFile = File(...),
    subject_id: int = Form(...),
    title: str = Form(...),
    num_pages: int = Form(...),
    is_premium: bool = Form(False),
    enabled: bool = Form(True),
    session: Session = Depends(get_session),
):
    """Upload a previous year question paper and store it against a subject."""
    try:
        subject = session.get(Subject, subject_id)
        if not subject:
            raise HTTPException(status_code=400, detail="Subject not found")

        do_path = f"subjects/{subject_id}/student-content/previous-year-question-papers"
        file_url = upload_to_do(file, do_path)

        paper = PreviousYearQuestionPaper(
            subject_id=subject_id,
            title=title,
            num_pages=num_pages,
            file_url=file_url,
            is_premium=is_premium,
            enabled=enabled,
        )
        session.add(paper)
        session.commit()
        session.refresh(paper)

        return {
            "message": "Previous year question paper uploaded",
            "data": paper.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error uploading previous year paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/previous-year-question-papers")
async def get_previous_year_question_papers(
    subject_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """List previous year question papers, optionally filtered by subject."""
    try:
        query = session.query(PreviousYearQuestionPaper)
        if subject_id is not None:
            query = query.filter(PreviousYearQuestionPaper.subject_id == subject_id)
        papers = query.all()
        return {"data": [p.dict() for p in papers]}

    except Exception as e:
        logger.error(f"Error fetching previous year papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/previous-year-question-papers/{paper_id}")
async def get_previous_year_question_paper(
    paper_id: int, session: Session = Depends(get_session)
):
    """Get a single previous year question paper by ID."""
    try:
        paper = session.get(PreviousYearQuestionPaper, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Previous year paper not found")
        return {"data": paper.dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching previous year paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/previous-year-question-papers/{paper_id}")
async def update_previous_year_question_paper(
    paper_id: int,
    file: UploadFile | None = File(None),
    subject_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    num_pages: Optional[int] = Form(None),
    is_premium: Optional[bool] = Form(None),
    enabled: Optional[bool] = Form(None),
    session: Session = Depends(get_session),
):
    """Update metadata, file, or subject for a previous year question paper."""
    try:
        paper = session.get(PreviousYearQuestionPaper, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Previous year paper not found")

        if subject_id is not None:
            subject = session.get(Subject, subject_id)
            if not subject:
                raise HTTPException(status_code=400, detail="Subject not found")
            paper.subject_id = subject_id

        if title is not None:
            paper.title = title
        if num_pages is not None:
            paper.num_pages = num_pages
        if is_premium is not None:
            paper.is_premium = is_premium
        if enabled is not None:
            paper.enabled = enabled

        if file:
            do_path = f"subjects/{paper.subject_id}/student-content/previous-year-question-papers"
            new_file_url = upload_to_do(file, do_path)

            if paper.file_url:
                try:
                    delete_from_do(paper.file_url)
                except Exception as e:
                    logger.error(
                        f"Error deleting old previous year paper file from DO: {e}"
                    )
                    raise HTTPException(
                        status_code=500, detail=f"Error deleting file: {e}"
                    )

            paper.file_url = new_file_url

        session.add(paper)
        session.commit()
        session.refresh(paper)

        return {
            "message": "Previous year question paper updated",
            "data": paper.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating previous year paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/previous-year-question-papers/{paper_id}")
async def delete_previous_year_question_paper(
    paper_id: int, session: Session = Depends(get_session)
):
    """Delete a previous year question paper and its file."""
    try:
        paper = session.get(PreviousYearQuestionPaper, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Previous year paper not found")

        if paper.file_url:
            try:
                delete_from_do(paper.file_url)
            except Exception as e:
                logger.error(
                    f"Error deleting previous year paper file from DO: {e}"
                )
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        session.delete(paper)
        session.commit()
        return {"message": f"Previous year paper '{paper.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting previous year paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
