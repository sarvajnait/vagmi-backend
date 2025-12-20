from typing import Dict, Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from loguru import logger
from sqlmodel import Session
from app.models import LLMTextbook, AdditionalNotes, LLMImage, LLMNote, QAPattern
from app.services.database import get_session
from app.core.agents.graph import EducationPlatform
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.files import upload_to_do, delete_from_do
from app.utils.cleanup import delete_embeddings_by_resource_id
import uuid
from app.utils.files import compress_image

router = APIRouter()
platform = EducationPlatform()
COLLECTION_NAME_TEXTBOOKS = "llm_textbooks"
COLLECTION_NAME_NOTES = "llm_notes"
COLLECTION_NAME_QA = "qa_patterns"
COLLECTION_NAME_IMAGES = "llm_images"
COLLECTION_NAME = COLLECTION_NAME_TEXTBOOKS  # For backwards compatibility


# ============================================================
# Textbook Processing
# ============================================================
def process_textbook_upload(file_url: str, metadata: Dict[str, str]) -> int:
    """Upload textbook to vector store with clean educational chunks."""
    try:
        loader = PyPDFLoader(file_url)

        pages = loader.load()

        # 3. Chunk ACROSS pages (not page-by-page)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len,
        )

        documents = text_splitter.split_documents(pages)

        if not documents:
            logger.warning("PDF contains no extractable text. Skipping vector storage.")
            return 0

        # 4. Attach metadata
        for doc in documents:
            doc.metadata.update(
                {
                    "chapter_id": str(metadata["chapter_id"]),
                    "source_file": metadata["source_file"],
                    "file_url": metadata["file_url"],
                    "textbook_id": str(metadata["textbook_id"]),
                    "content_type": "textbook",
                }
            )

        platform.vector_store_textbooks.add_documents(
            documents,
            ids=[str(uuid.uuid4()) for _ in documents],
        )

        logger.info(f"Uploaded {len(documents)} clean textbook chunks")
        return len(documents)

    except Exception as e:
        logger.error(f"Error uploading textbook: {e}", exc_info=True)
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
        do_path = f"chapters/{chapter_id}/llm-resources/textbooks"
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

        # Delete from vector store using utility function
        deleted_count = delete_embeddings_by_resource_id(
            session, textbook_id, COLLECTION_NAME_TEXTBOOKS, "textbook_id"
        )

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


# ============================================================
# LLM Image Endpoints
# ============================================================
@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """Upload an image to DigitalOcean, add to DB, and add to vector store with embeddings."""
    try:
        # Upload to DigitalOcean using utility

        compressed_file = compress_image(
            file,
            max_width=1600,
            max_height=1600,
            quality=80,
        )

        do_path = f"chapters/{chapter_id}/llm-resources/images"
        file_url = upload_to_do(compressed_file, do_path)

        # Parse tags from comma-separated string
        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Add image to DB
        image = LLMImage(
            chapter_id=chapter_id,
            title=file.filename,
            description=description,
            file_url=file_url,
            tags=tags_list,
        )
        session.add(image)
        session.commit()
        session.refresh(image)

        # Create embeddings for the image based on title + description + tags
        # This allows semantic search on image metadata
        try:
            from langchain_core.documents import Document

            # Combine title, description, and tags into searchable text
            searchable_text = file.filename
            if description:
                searchable_text += f"\n{description}"
            if tags_list:
                searchable_text += f"\nTags: {', '.join(tags_list)}"

            # Create a document with the searchable text and metadata
            doc = Document(
                page_content=searchable_text,
                metadata={
                    "chapter_id": str(chapter_id),
                    "image_id": str(image.id),
                    "title": file.filename,
                    "description": description or "",
                    "file_url": file_url,
                    "tags": tags_list or [],
                    "content_type": "image",
                },
            )

            # Add to vector store for semantic search
            platform.vector_store_images.add_documents(
                [doc],
                ids=[str(uuid.uuid4())],
            )
            logger.info(
                f"Added image to vector store: {file.filename} (ID: {image.id})"
            )

        except Exception as e:
            logger.error(f"Error adding image to vector store: {e}")
            # Don't fail the upload if vector store fails, just log it

        return {
            "message": "Image uploaded successfully and added to vector store",
            "data": image.dict(),
        }

    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image")
async def get_images(
    chapter_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """Get images filtered by chapter."""
    try:
        query = session.query(LLMImage)
        if chapter_id is not None:
            query = query.filter(LLMImage.chapter_id == chapter_id)
        images = query.all()
        return {"data": [img.dict() for img in images]}

    except Exception as e:
        logger.error(f"Error fetching images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/image/{image_id}")
async def delete_image(image_id: int, session: Session = Depends(get_session)):
    """Delete an image from DB, DigitalOcean Spaces, and vector store."""
    try:
        image = session.get(LLMImage, image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Delete from vector store using utility function
        try:
            deleted_count = delete_embeddings_by_resource_id(
                session, image_id, COLLECTION_NAME_IMAGES, "image_id"
            )
            logger.info(
                f"Deleted {deleted_count} embeddings from vector store: image_id={image_id}"
            )
        except Exception as e:
            logger.error(f"Error deleting image from vector store: {e}")
            # Continue with deletion even if vector store cleanup fails

        # Delete file from DigitalOcean
        if image.file_url:
            try:
                delete_from_do(image.file_url)
                logger.info(f"Deleted file from DigitalOcean: {image.file_url}")
            except Exception as e:
                logger.error(f"Error deleting file from DigitalOcean: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        # Delete from DB
        session.delete(image)
        session.commit()

        return {
            "message": f"Image '{image.title}' deleted successfully from all stores",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LLM Note Endpoints (PDF-based, for RAG)
# ============================================================
@router.post("/llm-note")
async def upload_llm_note(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    description: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """Upload an LLM note PDF and store clean chunks for RAG."""
    try:
        do_path = f"chapters/{chapter_id}/llm-resources/llm_notes"
        file_url = upload_to_do(file, do_path)

        note = LLMNote(
            chapter_id=chapter_id,
            title=file.filename,
            description=description,
            file_url=file_url,
        )
        session.add(note)
        session.commit()
        session.refresh(note)

        loader = PyPDFLoader(file_url)
        pages = loader.load()

        filtered_pages = [p for p in pages]

        if not filtered_pages:
            logger.warning("No valid LLM note content found")
            return {"message": "LLM note uploaded but no usable content found"}

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
            separators=["\n\n", "\n", ".", " "],
            length_function=len,
        )

        documents = text_splitter.split_documents(filtered_pages)

        if not documents:
            logger.warning("PDF contains no extractable text. Skipping vector storage.")
            return 0

        for doc in documents:
            doc.metadata.update(
                {
                    "chapter_id": str(chapter_id),
                    "source_file": file.filename,
                    "file_url": file_url,
                    "note_id": str(note.id),
                    "content_type": "llm_note",
                }
            )

        platform.vector_store_notes.add_documents(
            documents,
            ids=[str(uuid.uuid4()) for _ in documents],
        )

        logger.info(f"Uploaded {len(documents)} clean LLM note chunks")

        return {
            "message": "LLM note uploaded successfully",
            "data": note.dict(),
            "documents_processed": len(documents),
        }

    except Exception as e:
        logger.error(f"Error uploading LLM note: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-note")
async def get_llm_notes(
    chapter_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """Get LLM notes filtered by chapter."""
    try:
        query = session.query(LLMNote)
        if chapter_id is not None:
            query = query.filter(LLMNote.chapter_id == chapter_id)
        notes = query.all()
        return {"data": [note.dict() for note in notes]}

    except Exception as e:
        logger.error(f"Error fetching LLM notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/llm-note/{note_id}")
async def delete_llm_note(note_id: int, session: Session = Depends(get_session)):
    """Delete an LLM note from DB, vector store, and DigitalOcean."""
    try:
        note = session.get(LLMNote, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="LLM note not found")

        # Delete from vector store using utility function
        deleted_count = delete_embeddings_by_resource_id(
            session, note_id, COLLECTION_NAME_NOTES, "note_id"
        )

        # Delete file from DigitalOcean
        if note.file_url:
            try:
                delete_from_do(note.file_url)
                logger.info(f"Deleted file from DigitalOcean: {note.file_url}")
            except Exception as e:
                logger.error(f"Error deleting file from DigitalOcean: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        # Delete from DB
        session.delete(note)
        session.commit()

        return {
            "message": f"LLM note '{note.title}' deleted successfully",
            "vector_chunks_deleted": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting LLM note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Q&A Pattern Endpoints (PDF-based, for RAG)
# ============================================================
@router.post("/qa-pattern")
async def upload_qa_pattern(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    description: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """Upload Q&A pattern PDF with clean, reasoning-safe chunks."""
    try:
        do_path = f"chapters/{chapter_id}/llm-resources/qa_patterns"
        file_url = upload_to_do(file, do_path)

        pattern = QAPattern(
            chapter_id=chapter_id,
            title=file.filename,
            description=description,
            file_url=file_url,
        )
        session.add(pattern)
        session.commit()
        session.refresh(pattern)

        loader = PyPDFLoader(file_url)
        pages = loader.load()

        # ---- quality filter (Q&A tolerant but safe) ----
        def is_valid_qa_page(text: str) -> bool:
            text = text.strip()
            if len(text) < 50:
                return False
            digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
            if digit_ratio > 0.45:
                return False
            return True

        filtered_pages = [p for p in pages if is_valid_qa_page(p.page_content)]

        if not filtered_pages:
            logger.warning("No valid Q&A content found")
            return {"message": "Q&A uploaded but no usable content found"}

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=650,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
            length_function=len,
        )

        documents = text_splitter.split_documents(filtered_pages)

        if not documents:
            logger.warning("PDF contains no extractable text. Skipping vector storage.")
            return 0

        for doc in documents:
            doc.metadata.update(
                {
                    "chapter_id": str(chapter_id),
                    "source_file": file.filename,
                    "file_url": file_url,
                    "pattern_id": str(pattern.id),
                    "content_type": "qa_pattern",
                }
            )

        platform.vector_store_qa.add_documents(
            documents,
            ids=[str(uuid.uuid4()) for _ in documents],
        )

        logger.info(f"Uploaded {len(documents)} clean Q&A chunks")

        return {
            "message": "Q&A pattern uploaded successfully",
            "data": pattern.dict(),
            "documents_processed": len(documents),
        }

    except Exception as e:
        logger.error(f"Error uploading Q&A pattern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qa-pattern")
async def get_qa_patterns(
    chapter_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """Get Q&A patterns filtered by chapter."""
    try:
        query = session.query(QAPattern)
        if chapter_id is not None:
            query = query.filter(QAPattern.chapter_id == chapter_id)
        patterns = query.all()
        return {"data": [pattern.dict() for pattern in patterns]}

    except Exception as e:
        logger.error(f"Error fetching Q&A patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/qa-pattern/{pattern_id}")
async def delete_qa_pattern(pattern_id: int, session: Session = Depends(get_session)):
    """Delete a Q&A pattern from DB, vector store, and DigitalOcean."""
    try:
        pattern = session.get(QAPattern, pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail="Q&A pattern not found")

        # Delete from vector store using utility function
        deleted_count = delete_embeddings_by_resource_id(
            session, pattern_id, COLLECTION_NAME_QA, "pattern_id"
        )

        # Delete file from DigitalOcean
        if pattern.file_url:
            try:
                delete_from_do(pattern.file_url)
                logger.info(f"Deleted file from DigitalOcean: {pattern.file_url}")
            except Exception as e:
                logger.error(f"Error deleting file from DigitalOcean: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        # Delete from DB
        session.delete(pattern)
        session.commit()

        return {
            "message": f"Q&A pattern '{pattern.title}' deleted successfully",
            "vector_chunks_deleted": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting Q&A pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))
