from typing import Dict, Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Query
from pydantic import BaseModel
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from sqlalchemy import func, case
from app.models import LLMTextbook, AdditionalNotes, LLMImage, LLMNote, QAPattern, Chapter
from app.services.database import get_session
from app.core.agents.graph import EducationPlatform
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.files import upload_to_do, delete_from_do
from app.utils.cleanup import delete_embeddings_by_resource_id
from app.utils.kannada_converter import convert_kannada_text
import uuid
from app.utils.files import compress_image
from langchain_core.documents import Document

router = APIRouter()
platform = EducationPlatform()
COLLECTION_NAME_TEXTBOOKS = "llm_textbooks"
COLLECTION_NAME_NOTES = "llm_notes"
COLLECTION_NAME_QA = "qa_patterns"
COLLECTION_NAME_IMAGES = "llm_images"
COLLECTION_NAME = COLLECTION_NAME_TEXTBOOKS  # For backwards compatibility


class OrderUpdate(BaseModel):
    ids: list[int]


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


# ============================================================
# Textbook Processing
# ============================================================
def process_textbook_upload(file_url: str, metadata: Dict[str, str]) -> int:
    """Upload textbook to vector store with clean educational chunks."""
    try:
        loader = PyPDFLoader(file_url)

        pages = loader.load()

        # Convert Kannada text from legacy ASCII to Unicode
        for page in pages:
            page.page_content = convert_kannada_text(page.page_content)

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
        for idx, doc in enumerate(documents):
            doc.metadata.update(
                {
                    "chapter_id": str(metadata["chapter_id"]),
                    "source_file": metadata["source_file"],
                    "file_url": metadata["file_url"],
                    "textbook_id": str(metadata["textbook_id"]),
                    "content_type": "textbook",
                    "chunk_index": idx,
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


def build_image_document(
    *,
    chapter_id: int,
    image_id: int,
    title: str,
    description: Optional[str],
    file_url: str,
    tags: Optional[list[str]],
):
    searchable_text = title
    if description:
        searchable_text += f"\n{description}"
    if tags:
        searchable_text += f"\nTags: {', '.join(tags)}"

    return Document(
        page_content=searchable_text,
        metadata={
            "chapter_id": str(chapter_id),
            "image_id": str(image_id),
            "title": title,
            "description": description or "",
            "file_url": file_url,
            "tags": tags or [],
            "content_type": "image",
        },
    )


def parse_tags(tags: Optional[str]) -> Optional[list[str]]:
    if tags is None:
        return None
    tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return tags_list or []


# ============================================================
# Textbook Endpoints
# ============================================================
@router.post("/textbook")
async def upload_textbook(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Upload a textbook to DigitalOcean and add to DB/vector store."""
    try:

        # Upload to DigitalOcean using utility
        do_path = f"chapters/{chapter_id}/llm-resources/textbooks"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(
            select(func.max(LLMTextbook.sort_order)).where(
                LLMTextbook.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        # Add textbook to DB
        textbook = LLMTextbook(
            chapter_id=chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(textbook)
        await session.commit()
        await session.refresh(textbook)

        metadata = {
            "chapter_id": chapter_id,
            "source_file": textbook.original_filename or textbook.title,
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
    session: AsyncSession = Depends(get_session),
):
    """Get textbooks filtered by chapter."""
    try:
        query = select(LLMTextbook)
        if chapter_id is not None:
            query = query.where(LLMTextbook.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(LLMTextbook))
        _result = await session.exec(query)
        textbooks = _result.all()
        return {"data": [t.dict() for t in textbooks]}

    except Exception as e:
        logger.error(f"Error fetching textbooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/textbook/order")
async def reorder_textbooks(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update textbook order within a chapter."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(LLMTextbook).where(
            LLMTextbook.chapter_id == chapter_id,
            LLMTextbook.id.in_(payload.ids),
        )
    )
    textbooks = _result.all()
    if len(textbooks) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid textbook ids for chapter")

    textbook_map = {t.id: t for t in textbooks}
    for index, textbook_id in enumerate(payload.ids, start=1):
        textbook_map[textbook_id].sort_order = index
        session.add(textbook_map[textbook_id])

    await session.commit()
    return {"message": "Textbook order updated"}


@router.delete("/textbook/{textbook_id}")
async def delete_textbook(textbook_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a textbook from DB, vector store, and DigitalOcean Spaces."""
    try:
        textbook = await session.get(LLMTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        # Delete from vector store using utility function
        deleted_count = await delete_embeddings_by_resource_id(
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
        await session.delete(textbook)
        await session.commit()

        return {
            "message": f"Textbook '{textbook.title}' deleted successfully",
            "vector_chunks_deleted": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting textbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/textbook/{textbook_id}")
async def update_textbook(
    textbook_id: int,
    file: UploadFile | None = File(None),
    chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update a textbook's metadata or file and reindex embeddings when needed."""
    try:
        textbook = await session.get(LLMTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        should_reindex = False

        if chapter_id is not None and chapter_id != textbook.chapter_id:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            textbook.chapter_id = chapter_id
            should_reindex = True

        if title is not None:
            textbook.title = title
        if description is not None:
            textbook.description = description

        if file:
            do_path = f"chapters/{textbook.chapter_id}/llm-resources/textbooks"
            new_file_url = upload_to_do(file, do_path)

            if textbook.file_url:
                delete_from_do(textbook.file_url)

            textbook.file_url = new_file_url
            textbook.original_filename = file.filename
            if title is None:
                textbook.title = file.filename
            should_reindex = True

        if should_reindex:
            await delete_embeddings_by_resource_id(
                session, textbook_id, COLLECTION_NAME_TEXTBOOKS, "textbook_id"
            )
            metadata = {
                "chapter_id": textbook.chapter_id,
                "source_file": textbook.original_filename or textbook.title,
                "file_url": textbook.file_url,
                "textbook_id": textbook.id,
            }
            process_textbook_upload(textbook.file_url, metadata)

        session.add(textbook)
        await session.commit()
        await session.refresh(textbook)

        return {"message": "Textbook updated", "data": textbook.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating textbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ============================================================
# LLM Notes Endpoints
# ============================================================
@router.post("/additional-notes")
async def create_additional_note(
    chapter_id: int = Form(...),
    note: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    """Create a new additional note for a chapter."""
    try:
        _result = await session.exec(
            select(func.max(AdditionalNotes.sort_order)).where(
                AdditionalNotes.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        additional_note = AdditionalNotes(
            chapter_id=chapter_id,
            note=note,
            sort_order=next_order,
        )
        session.add(additional_note)
        await session.commit()
        await session.refresh(additional_note)
        return {"message": "Note added", "data": additional_note.dict()}

    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/additional-notes")
async def get_additional_notes(
    chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """Get all notes or filter by chapter."""
    try:
        query = select(AdditionalNotes)
        if chapter_id is not None:
            query = query.where(AdditionalNotes.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(AdditionalNotes))
        _result = await session.exec(query)
        notes = _result.all()
        return {"data": [n.dict() for n in notes]}

    except Exception as e:
        logger.error(f"Error fetching notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/additional-notes/order")
async def reorder_additional_notes(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update additional notes order within a chapter."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(AdditionalNotes).where(
            AdditionalNotes.chapter_id == chapter_id,
            AdditionalNotes.id.in_(payload.ids),
        )
    )
    notes = _result.all()
    if len(notes) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid note ids for chapter")

    note_map = {n.id: n for n in notes}
    for index, note_id in enumerate(payload.ids, start=1):
        note_map[note_id].sort_order = index
        session.add(note_map[note_id])

    await session.commit()
    return {"message": "Note order updated"}


@router.delete("/additional-notes/{note_id}")
async def delete_additional_note(note_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a specific note by ID."""
    try:
        note = await session.get(AdditionalNotes, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        await session.delete(note)
        await session.commit()
        return {"message": f"Note with id {note_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/additional-notes/{note_id}")
async def update_additional_note(
    note_id: int,
    chapter_id: Optional[int] = Form(None),
    note: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update an additional note."""
    try:
        additional_note = await session.get(AdditionalNotes, note_id)
        if not additional_note:
            raise HTTPException(status_code=404, detail="Note not found")

        if chapter_id is not None and chapter_id != additional_note.chapter_id:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            additional_note.chapter_id = chapter_id
        if note is not None:
            additional_note.note = note

        session.add(additional_note)
        await session.commit()
        await session.refresh(additional_note)

        return {"message": "Note updated", "data": additional_note.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ============================================================
# LLM Image Endpoints
# ============================================================
@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
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
        tags_list = parse_tags(tags)
        image_title = title or file.filename

        _result = await session.exec(
            select(func.max(LLMImage.sort_order)).where(
                LLMImage.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        # Add image to DB
        image = LLMImage(
            chapter_id=chapter_id,
            title=image_title,
            description=description,
            file_url=file_url,
            tags=tags_list,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(image)
        await session.commit()
        await session.refresh(image)

        # Create embeddings for the image based on title + description + tags
        # This allows semantic search on image metadata
        try:
            doc = build_image_document(
                chapter_id=chapter_id,
                image_id=image.id,
                title=image_title,
                description=description,
                file_url=file_url,
                tags=tags_list,
            )
            platform.vector_store_images.add_documents(
                [doc],
                ids=[str(uuid.uuid4())],
            )
            logger.info(f"Added image to vector store: {image_title} (ID: {image.id})")

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
    session: AsyncSession = Depends(get_session),
):
    """Get images filtered by chapter."""
    try:
        query = select(LLMImage)
        if chapter_id is not None:
            query = query.where(LLMImage.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(LLMImage))
        _result = await session.exec(query)
        images = _result.all()
        return {"data": [img.dict() for img in images]}

    except Exception as e:
        logger.error(f"Error fetching images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/image/order")
async def reorder_images(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update image order within a chapter."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(LLMImage).where(
            LLMImage.chapter_id == chapter_id,
            LLMImage.id.in_(payload.ids),
        )
    )
    images = _result.all()
    if len(images) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid image ids for chapter")

    image_map = {img.id: img for img in images}
    for index, image_id in enumerate(payload.ids, start=1):
        image_map[image_id].sort_order = index
        session.add(image_map[image_id])

    await session.commit()
    return {"message": "Image order updated"}


@router.delete("/image/{image_id}")
async def delete_image(image_id: int, session: AsyncSession = Depends(get_session)):
    """Delete an image from DB, DigitalOcean Spaces, and vector store."""
    try:
        image = await session.get(LLMImage, image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Delete from vector store using utility function
        try:
            deleted_count = await delete_embeddings_by_resource_id(
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
        await session.delete(image)
        await session.commit()

        return {
            "message": f"Image '{image.title}' deleted successfully from all stores",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/image/{image_id}")
async def update_image(
    image_id: int,
    file: UploadFile | None = File(None),
    chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update an image's metadata or file and refresh vector store."""
    try:
        image = await session.get(LLMImage, image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        should_reindex = False

        if chapter_id is not None and chapter_id != image.chapter_id:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            image.chapter_id = chapter_id
            should_reindex = True

        if title is not None:
            image.title = title
            should_reindex = True
        if description is not None:
            image.description = description
            should_reindex = True
        if tags is not None:
            image.tags = parse_tags(tags)
            should_reindex = True

        if file:
            compressed_file = compress_image(
                file,
                max_width=1600,
                max_height=1600,
                quality=80,
            )
            do_path = f"chapters/{image.chapter_id}/llm-resources/images"
            new_file_url = upload_to_do(compressed_file, do_path)

            if image.file_url:
                delete_from_do(image.file_url)

            image.file_url = new_file_url
            image.original_filename = file.filename
            if title is None:
                image.title = file.filename
            should_reindex = True

        if should_reindex:
            try:
                await delete_embeddings_by_resource_id(
                    session, image_id, COLLECTION_NAME_IMAGES, "image_id"
                )
            except Exception as e:
                logger.error(f"Error deleting image embeddings: {e}")

            try:
                doc = build_image_document(
                    chapter_id=image.chapter_id,
                    image_id=image.id,
                    title=image.title,
                    description=image.description,
                    file_url=image.file_url,
                    tags=image.tags,
                )
                platform.vector_store_images.add_documents(
                    [doc],
                    ids=[str(uuid.uuid4())],
                )
            except Exception as e:
                logger.error(f"Error reindexing image embeddings: {e}")

        session.add(image)
        await session.commit()
        await session.refresh(image)

        return {"message": "Image updated", "data": image.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ============================================================
# LLM Note Endpoints (PDF-based, for RAG)
# ============================================================
@router.post("/llm-note")
async def upload_llm_note(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Upload an LLM note PDF and store clean chunks for RAG."""
    try:
        do_path = f"chapters/{chapter_id}/llm-resources/llm_notes"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(
            select(func.max(LLMNote.sort_order)).where(
                LLMNote.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        note = LLMNote(
            chapter_id=chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)

        loader = PyPDFLoader(file_url)
        pages = loader.load()

        # Convert Kannada text from legacy ASCII to Unicode
        for page in pages:
            page.page_content = convert_kannada_text(page.page_content)

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
                    "source_file": note.original_filename or note.title,
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
    session: AsyncSession = Depends(get_session),
):
    """Get LLM notes filtered by chapter."""
    try:
        query = select(LLMNote)
        if chapter_id is not None:
            query = query.where(LLMNote.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(LLMNote))
        _result = await session.exec(query)
        notes = _result.all()
        return {"data": [note.dict() for note in notes]}

    except Exception as e:
        logger.error(f"Error fetching LLM notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/llm-note/order")
async def reorder_llm_notes(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update LLM note order within a chapter."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(LLMNote).where(
            LLMNote.chapter_id == chapter_id,
            LLMNote.id.in_(payload.ids),
        )
    )
    notes = _result.all()
    if len(notes) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid note ids for chapter")

    note_map = {n.id: n for n in notes}
    for index, note_id in enumerate(payload.ids, start=1):
        note_map[note_id].sort_order = index
        session.add(note_map[note_id])

    await session.commit()
    return {"message": "LLM note order updated"}


@router.delete("/llm-note/{note_id}")
async def delete_llm_note(note_id: int, session: AsyncSession = Depends(get_session)):
    """Delete an LLM note from DB, vector store, and DigitalOcean."""
    try:
        note = await session.get(LLMNote, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="LLM note not found")

        # Delete from vector store using utility function
        deleted_count = await delete_embeddings_by_resource_id(
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
        await session.delete(note)
        await session.commit()

        return {
            "message": f"LLM note '{note.title}' deleted successfully",
            "vector_chunks_deleted": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting LLM note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/llm-note/{note_id}")
async def update_llm_note(
    note_id: int,
    file: UploadFile | None = File(None),
    chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update an LLM note and reindex when file or chapter changes."""
    try:
        note = await session.get(LLMNote, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="LLM note not found")

        should_reindex = False
        if chapter_id is not None and chapter_id != note.chapter_id:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            note.chapter_id = chapter_id
            should_reindex = True

        if title is not None:
            note.title = title
        if description is not None:
            note.description = description

        if file:
            do_path = f"chapters/{note.chapter_id}/llm-resources/llm_notes"
            new_file_url = upload_to_do(file, do_path)

            if note.file_url:
                delete_from_do(note.file_url)

            note.file_url = new_file_url
            note.original_filename = file.filename
            if title is None:
                note.title = file.filename
            should_reindex = True

        if should_reindex:
            await delete_embeddings_by_resource_id(
                session, note_id, COLLECTION_NAME_NOTES, "note_id"
            )

            loader = PyPDFLoader(note.file_url)
            pages = loader.load()

            # Convert Kannada text from legacy ASCII to Unicode
            for page in pages:
                page.page_content = convert_kannada_text(page.page_content)

            filtered_pages = [p for p in pages]
            if not filtered_pages:
                logger.warning("No valid LLM note content found")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=900,
                    chunk_overlap=120,
                    separators=["\n\n", "\n", ".", " "],
                    length_function=len,
                )
                documents = text_splitter.split_documents(filtered_pages)
                for doc in documents:
                    doc.metadata.update(
                        {
                            "chapter_id": str(note.chapter_id),
                            "source_file": note.original_filename or note.title,
                            "file_url": note.file_url,
                            "note_id": str(note.id),
                            "content_type": "llm_note",
                        }
                    )
                platform.vector_store_notes.add_documents(
                    documents,
                    ids=[str(uuid.uuid4()) for _ in documents],
                )

        session.add(note)
        await session.commit()
        await session.refresh(note)

        return {"message": "LLM note updated", "data": note.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating LLM note: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ============================================================
# Q&A Pattern Endpoints (PDF-based, for RAG)
# ============================================================
@router.post("/qa-pattern")
async def upload_qa_pattern(
    file: UploadFile = File(...),
    chapter_id: int = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Upload Q&A pattern PDF with clean, reasoning-safe chunks."""
    try:
        do_path = f"chapters/{chapter_id}/llm-resources/qa_patterns"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(
            select(func.max(QAPattern.sort_order)).where(
                QAPattern.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        pattern = QAPattern(
            chapter_id=chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(pattern)
        await session.commit()
        await session.refresh(pattern)

        loader = PyPDFLoader(file_url)
        pages = loader.load()

        # Convert Kannada text from legacy ASCII to Unicode
        for page in pages:
            page.page_content = convert_kannada_text(page.page_content)

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
                    "source_file": pattern.original_filename or pattern.title,
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
    session: AsyncSession = Depends(get_session),
):
    """Get Q&A patterns filtered by chapter."""
    try:
        query = select(QAPattern)
        if chapter_id is not None:
            query = query.where(QAPattern.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(QAPattern))
        _result = await session.exec(query)
        patterns = _result.all()
        return {"data": [pattern.dict() for pattern in patterns]}

    except Exception as e:
        logger.error(f"Error fetching Q&A patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/qa-pattern/order")
async def reorder_qa_patterns(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update Q&A pattern order within a chapter."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(QAPattern).where(
            QAPattern.chapter_id == chapter_id,
            QAPattern.id.in_(payload.ids),
        )
    )
    patterns = _result.all()
    if len(patterns) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid pattern ids for chapter")

    pattern_map = {p.id: p for p in patterns}
    for index, pattern_id in enumerate(payload.ids, start=1):
        pattern_map[pattern_id].sort_order = index
        session.add(pattern_map[pattern_id])

    await session.commit()
    return {"message": "Q&A pattern order updated"}


@router.delete("/qa-pattern/{pattern_id}")
async def delete_qa_pattern(pattern_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a Q&A pattern from DB, vector store, and DigitalOcean."""
    try:
        pattern = await session.get(QAPattern, pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail="Q&A pattern not found")

        # Delete from vector store using utility function
        deleted_count = await delete_embeddings_by_resource_id(
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
        await session.delete(pattern)
        await session.commit()

        return {
            "message": f"Q&A pattern '{pattern.title}' deleted successfully",
            "vector_chunks_deleted": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting Q&A pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/qa-pattern/{pattern_id}")
async def update_qa_pattern(
    pattern_id: int,
    file: UploadFile | None = File(None),
    chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update a Q&A pattern and reindex when file or chapter changes."""
    try:
        pattern = await session.get(QAPattern, pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail="Q&A pattern not found")

        should_reindex = False
        if chapter_id is not None and chapter_id != pattern.chapter_id:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            pattern.chapter_id = chapter_id
            should_reindex = True

        if title is not None:
            pattern.title = title
        if description is not None:
            pattern.description = description

        if file:
            do_path = f"chapters/{pattern.chapter_id}/llm-resources/qa_patterns"
            new_file_url = upload_to_do(file, do_path)

            if pattern.file_url:
                delete_from_do(pattern.file_url)

            pattern.file_url = new_file_url
            pattern.original_filename = file.filename
            if title is None:
                pattern.title = file.filename
            should_reindex = True

        if should_reindex:
            await delete_embeddings_by_resource_id(
                session, pattern_id, COLLECTION_NAME_QA, "pattern_id"
            )

            loader = PyPDFLoader(pattern.file_url)
            pages = loader.load()

            # Convert Kannada text from legacy ASCII to Unicode
            for page in pages:
                page.page_content = convert_kannada_text(page.page_content)

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
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=650,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", " "],
                    length_function=len,
                )
                documents = text_splitter.split_documents(filtered_pages)
                for doc in documents:
                    doc.metadata.update(
                        {
                            "chapter_id": str(pattern.chapter_id),
                            "source_file": pattern.original_filename or pattern.title,
                            "file_url": pattern.file_url,
                            "pattern_id": str(pattern.id),
                            "content_type": "qa_pattern",
                        }
                    )
                platform.vector_store_qa.add_documents(
                    documents,
                    ids=[str(uuid.uuid4()) for _ in documents],
                )

        session.add(pattern)
        await session.commit()
        await session.refresh(pattern)

        return {"message": "Q&A pattern updated", "data": pattern.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating Q&A pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

