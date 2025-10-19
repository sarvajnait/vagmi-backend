from typing import Dict, Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from loguru import logger
from sqlmodel import Session, text
from app.models import LLMTextbook, AdditionalNotes
from app.services.database import get_session
from app.core.langgraph.graph import EducationPlatform
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.files import upload_to_do, delete_from_do

router = APIRouter()
platform = EducationPlatform()
COLLECTION_NAME = "llm_textbooks"


# ============================================================
# Textbook Processing
# ============================================================
def process_textbook_upload(file_url: str, metadata: Dict[str, str]) -> int:
    """Upload document to vector store with hierarchical metadata."""
    try:
        loader = PyPDFLoader(file_url)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
        )
        documents = loader.load_and_split(text_splitter)

        for doc in documents:
            doc.metadata.update(
                {
                    "chapter_id": str(metadata["chapter_id"]),
                    "source_file": metadata["source_file"],
                    "file_url": metadata["file_url"],
                    "textbook_id": str(metadata["textbook_id"]),
                }
            )

        platform.vector_store.add_documents(documents)
        logger.info(f"Successfully uploaded {len(documents)} document chunks")
        return len(documents)

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise


# ============================================================
# Textbook Endpoints
# ============================================================
@router.post("/textbook")
async def upload_textbook(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    session: Session = Depends(get_session),
):
    """Upload a textbook to DigitalOcean and add to DB/vector store."""
    try:
        # Upload to DigitalOcean using utility
        do_path = f"chapters/{chapter_id}/textbooks"
        file_url = upload_to_do(file, do_path)

        # Add textbook to DB
        textbook = LLMTextbook(
            chapter_id=chapter_id,
            title=file.filename,
            description=None,
            file_url=file_url,
        )
        session.add(textbook)
        session.commit()
        session.refresh(textbook)

        metadata = {
            "chapter_id": chapter_id,
            "source_file": file.filename,
            "file_url": file_url,
            "textbook_id": textbook.id,
        }

        doc_count = process_textbook_upload(file_url, metadata)

        return {
            "message": "Document uploaded",
            "metadata": metadata,
            "documents_processed": doc_count,
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
        query = session.query(LLMTextbook)
        if chapter_id is not None:
            query = query.filter(LLMTextbook.chapter_id == chapter_id)
        textbooks = query.all()
        return {"data": [t.dict() for t in textbooks]}

    except Exception as e:
        logger.error(f"Error fetching textbooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/textbook/{textbook_id}")
async def delete_textbook(textbook_id: int, session: Session = Depends(get_session)):
    """Delete a textbook from DB, vector store, and DigitalOcean Spaces."""
    try:
        textbook = session.get(LLMTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        # Delete from vector store
        query = """
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
        )
        AND cmetadata->>'textbook_id' = :textbook_id
        """
        result = session.execute(
            text(query),
            {"collection_name": COLLECTION_NAME, "textbook_id": str(textbook_id)},
        )
        deleted_count = result.rowcount
        session.commit()

        # Delete file from DigitalOcean
        if textbook.file_url:
            try:
                delete_from_do(textbook.file_url)
                logger.info(f"Deleted file from DigitalOcean: {textbook.file_url}")
            except Exception as e:
                logger.error(f"Error deleting file from DigitalOcean: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        # Delete from DB
        session.delete(textbook)
        session.commit()

        return {
            "message": f"Textbook '{textbook.title}' deleted successfully",
            "vector_chunks_deleted": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting textbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LLM Notes Endpoints
# ============================================================
@router.post("/additional-notes")
async def create_additional_note(
    chapter_id: int = Form(...),
    note: str = Form(...),
    session: Session = Depends(get_session),
):
    """Create a new additional note for a chapter."""
    try:
        additional_note = AdditionalNotes(chapter_id=chapter_id, note=note)
        session.add(additional_note)
        session.commit()
        session.refresh(additional_note)
        return {"message": "Note added", "data": additional_note.dict()}

    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/additional-notes")
async def get_additional_notes(
    chapter_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """Get all notes or filter by chapter."""
    try:
        query = session.query(AdditionalNotes)
        if chapter_id is not None:
            query = query.filter(AdditionalNotes.chapter_id == chapter_id)
        notes = query.all()
        return {"data": [n.dict() for n in notes]}

    except Exception as e:
        logger.error(f"Error fetching notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/additional-notes/{note_id}")
async def delete_additional_note(note_id: int, session: Session = Depends(get_session)):
    """Delete a specific note by ID."""
    try:
        note = session.get(AdditionalNotes, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        session.delete(note)
        session.commit()
        return {"message": f"Note with id {note_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        raise HTTPException(status_code=500, detail=str(e))
