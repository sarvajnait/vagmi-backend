from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from loguru import logger
from sqlmodel import Session
from app.models import StudentTextbook,StudentNotes
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
        do_path = f"chapters/{chapter_id}/textbooks"
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
        do_path = f"chapters/{chapter_id}/notes"
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

        return {
            "message": "Note uploaded",
            "data": note.dict()
        }

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