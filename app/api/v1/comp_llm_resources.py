from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, HTTPException, Form, Query
from pydantic import BaseModel
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from sqlalchemy import func, case
from app.models.comp_llm_resources import (
    CompLLMTextbook, CompLLMNote, CompAdditionalNote, CompQAPattern, CompLLMImage,
)
from app.models.comp_artifacts import CompChapterArtifact
from app.models import ActivityGenerationJob
from app.services.database import get_session
from app.utils.files import upload_to_do, delete_from_do
from app.utils.cleanup import delete_embeddings_by_resource_id
from app.utils.files import compress_image

router = APIRouter()

COMP_COLLECTION_TEXTBOOKS = "comp_llm_textbooks"
COMP_COLLECTION_NOTES = "comp_llm_notes"
COMP_COLLECTION_QA = "comp_qa_patterns"
COMP_COLLECTION_IMAGES = "comp_llm_images"


class OrderUpdate(BaseModel):
    ids: list[int]


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


def _resolve_fk(comp_chapter_id: Optional[int], sub_chapter_id: Optional[int]):
    """Ensure exactly one FK is set."""
    if comp_chapter_id is None and sub_chapter_id is None:
        raise HTTPException(status_code=400, detail="Either comp_chapter_id or sub_chapter_id must be provided")
    return comp_chapter_id, sub_chapter_id


# ============================================================
# Textbook Endpoints
# ============================================================

@router.post("/textbook")
async def upload_comp_textbook(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    try:
        comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
        folder_id = comp_chapter_id or sub_chapter_id
        do_path = f"comp/chapters/{folder_id}/llm-resources/textbooks"
        file_url = upload_to_do(file, do_path)

        filter_col = CompLLMTextbook.comp_chapter_id if comp_chapter_id else CompLLMTextbook.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _result = await session.exec(select(func.max(CompLLMTextbook.sort_order)).where(filter_col == filter_val))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        textbook = CompLLMTextbook(
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(textbook)
        await session.commit()
        await session.refresh(textbook)

        job = ActivityGenerationJob(
            job_type="comp_textbook_process",
            status="pending",
            payload={
                "comp_chapter_id": comp_chapter_id,
                "sub_chapter_id": sub_chapter_id,
                "textbook_id": textbook.id,
                "file_url": file_url,
                "source_file": textbook.original_filename or textbook.title,
            },
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        from app.services.activity_jobs import enqueue_activity_job
        background_tasks.add_task(enqueue_activity_job, job.id)

        return {
            "message": "Textbook uploaded. Processing in progress.",
            "textbook_id": textbook.id,
            "job_id": job.id,
            "status": "processing",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading comp textbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/textbook")
async def get_comp_textbooks(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    try:
        query = select(CompLLMTextbook)
        if comp_chapter_id is not None:
            query = query.where(CompLLMTextbook.comp_chapter_id == comp_chapter_id)
        elif sub_chapter_id is not None:
            query = query.where(CompLLMTextbook.sub_chapter_id == sub_chapter_id)
        query = query.order_by(*sort_ordering(CompLLMTextbook))
        _result = await session.exec(query)
        textbooks = _result.all()

        # Fetch artifact status
        chapter_id_val = comp_chapter_id or sub_chapter_id
        artifact_status = None
        if chapter_id_val:
            filter_col = CompChapterArtifact.comp_chapter_id if comp_chapter_id else CompChapterArtifact.sub_chapter_id
            art_result = await session.exec(
                select(CompChapterArtifact).where(
                    filter_col == chapter_id_val,
                    CompChapterArtifact.artifact_type == "chapter_summary",
                )
            )
            art = art_result.first()
            artifact_status = art.status if art else None

        data = []
        for t in textbooks:
            row = t.dict()
            row["artifact_status"] = artifact_status
            data.append(row)
        return {"data": data}
    except Exception as e:
        logger.error(f"Error fetching comp textbooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/textbook/order")
async def reorder_comp_textbooks(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")
    filter_col = CompLLMTextbook.comp_chapter_id if comp_chapter_id else CompLLMTextbook.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(
        select(CompLLMTextbook).where(filter_col == filter_val, CompLLMTextbook.id.in_(payload.ids))
    )
    textbooks = _result.all()
    if len(textbooks) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid textbook ids")
    tb_map = {t.id: t for t in textbooks}
    for index, tid in enumerate(payload.ids, start=1):
        tb_map[tid].sort_order = index
        session.add(tb_map[tid])
    await session.commit()
    return {"message": "Textbook order updated"}


@router.delete("/textbook/{textbook_id}")
async def delete_comp_textbook(textbook_id: int, session: AsyncSession = Depends(get_session)):
    try:
        textbook = await session.get(CompLLMTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")
        await delete_embeddings_by_resource_id(session, textbook_id, COMP_COLLECTION_TEXTBOOKS, "textbook_id")
        if textbook.file_url:
            delete_from_do(textbook.file_url)
        await session.delete(textbook)
        await session.commit()
        return {"message": f"Textbook '{textbook.title}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/textbook/{textbook_id}")
async def update_comp_textbook(
    background_tasks: BackgroundTasks,
    textbook_id: int,
    file: UploadFile | None = File(None),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    try:
        textbook = await session.get(CompLLMTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")
        if title is not None:
            textbook.title = title
        if description is not None:
            textbook.description = description
        job_id = None
        if file:
            folder_id = textbook.comp_chapter_id or textbook.sub_chapter_id
            do_path = f"comp/chapters/{folder_id}/llm-resources/textbooks"
            new_file_url = upload_to_do(file, do_path)
            if textbook.file_url:
                delete_from_do(textbook.file_url)
            textbook.file_url = new_file_url
            textbook.original_filename = file.filename
            if title is None:
                textbook.title = file.filename
        session.add(textbook)
        await session.commit()
        await session.refresh(textbook)
        if file:
            job = ActivityGenerationJob(
                job_type="comp_textbook_process",
                status="pending",
                payload={
                    "comp_chapter_id": textbook.comp_chapter_id,
                    "sub_chapter_id": textbook.sub_chapter_id,
                    "textbook_id": textbook.id,
                    "file_url": textbook.file_url,
                    "source_file": textbook.original_filename or textbook.title,
                },
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            from app.services.activity_jobs import enqueue_activity_job
            background_tasks.add_task(enqueue_activity_job, job.id)
            job_id = job.id
        response = {"message": "Textbook updated", "data": textbook.dict()}
        if job_id:
            response["job_id"] = job_id
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LLM Notes Endpoints
# ============================================================

@router.post("/llm-note")
async def upload_comp_llm_note(
    file: UploadFile = File(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    try:
        comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
        folder_id = comp_chapter_id or sub_chapter_id
        do_path = f"comp/chapters/{folder_id}/llm-resources/notes"
        file_url = upload_to_do(file, do_path)

        filter_col = CompLLMNote.comp_chapter_id if comp_chapter_id else CompLLMNote.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _result = await session.exec(select(func.max(CompLLMNote.sort_order)).where(filter_col == filter_val))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]

        note = CompLLMNote(
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=(max_order or 0) + 1,
            original_filename=file.filename,
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)
        return {"message": "LLM note uploaded", "data": note.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-note")
async def get_comp_llm_notes(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(CompLLMNote)
    if comp_chapter_id is not None:
        query = query.where(CompLLMNote.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompLLMNote.sub_chapter_id == sub_chapter_id)
    query = query.order_by(*sort_ordering(CompLLMNote))
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}


@router.put("/llm-note/order")
async def reorder_comp_llm_notes(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    filter_col = CompLLMNote.comp_chapter_id if comp_chapter_id else CompLLMNote.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(CompLLMNote).where(filter_col == filter_val, CompLLMNote.id.in_(payload.ids)))
    notes = _result.all()
    if len(notes) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid note ids")
    note_map = {n.id: n for n in notes}
    for index, nid in enumerate(payload.ids, start=1):
        note_map[nid].sort_order = index
        session.add(note_map[nid])
    await session.commit()
    return {"message": "Note order updated"}


@router.delete("/llm-note/{note_id}")
async def delete_comp_llm_note(note_id: int, session: AsyncSession = Depends(get_session)):
    note = await session.get(CompLLMNote, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    await delete_embeddings_by_resource_id(session, note_id, COMP_COLLECTION_NOTES, "note_id")
    if note.file_url:
        delete_from_do(note.file_url)
    await session.delete(note)
    await session.commit()
    return {"message": "Note deleted"}


@router.put("/llm-note/{note_id}")
async def update_comp_llm_note(
    note_id: int,
    file: UploadFile | None = File(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    note = await session.get(CompLLMNote, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    if title is not None:
        note.title = title
    if description is not None:
        note.description = description
    if file:
        folder_id = note.comp_chapter_id or note.sub_chapter_id
        new_url = upload_to_do(file, f"comp/chapters/{folder_id}/llm-resources/notes")
        if note.file_url:
            delete_from_do(note.file_url)
        note.file_url = new_url
        note.original_filename = file.filename
    session.add(note)
    await session.commit()
    await session.refresh(note)
    return {"message": "Note updated", "data": note.dict()}


# ============================================================
# Additional Notes Endpoints
# ============================================================

@router.post("/additional-note")
async def create_comp_additional_note(
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    note: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
    filter_col = CompAdditionalNote.comp_chapter_id if comp_chapter_id else CompAdditionalNote.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(func.max(CompAdditionalNote.sort_order)).where(filter_col == filter_val))
    max_order = _result.first()
    if isinstance(max_order, tuple):
        max_order = max_order[0]
    obj = CompAdditionalNote(
        comp_chapter_id=comp_chapter_id,
        sub_chapter_id=sub_chapter_id,
        note=note,
        sort_order=(max_order or 0) + 1,
    )
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Additional note created", "data": obj.dict()}


@router.get("/additional-note")
async def get_comp_additional_notes(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(CompAdditionalNote)
    if comp_chapter_id is not None:
        query = query.where(CompAdditionalNote.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompAdditionalNote.sub_chapter_id == sub_chapter_id)
    query = query.order_by(*sort_ordering(CompAdditionalNote))
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}


@router.put("/additional-note/order")
async def reorder_comp_additional_notes(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    filter_col = CompAdditionalNote.comp_chapter_id if comp_chapter_id else CompAdditionalNote.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(CompAdditionalNote).where(filter_col == filter_val, CompAdditionalNote.id.in_(payload.ids)))
    notes = _result.all()
    if len(notes) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid note ids")
    note_map = {n.id: n for n in notes}
    for index, nid in enumerate(payload.ids, start=1):
        note_map[nid].sort_order = index
        session.add(note_map[nid])
    await session.commit()
    return {"message": "Additional note order updated"}


@router.delete("/additional-note/{note_id}")
async def delete_comp_additional_note(note_id: int, session: AsyncSession = Depends(get_session)):
    note = await session.get(CompAdditionalNote, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    await session.delete(note)
    await session.commit()
    return {"message": "Note deleted"}


@router.put("/additional-note/{note_id}")
async def update_comp_additional_note(
    note_id: int,
    note: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    obj = await session.get(CompAdditionalNote, note_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Note not found")
    if note is not None:
        obj.note = note
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Note updated", "data": obj.dict()}


# ============================================================
# QA Pattern Endpoints
# ============================================================

@router.post("/qa-pattern")
async def upload_comp_qa_pattern(
    file: UploadFile = File(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    try:
        comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
        folder_id = comp_chapter_id or sub_chapter_id
        do_path = f"comp/chapters/{folder_id}/llm-resources/qa-patterns"
        file_url = upload_to_do(file, do_path)

        filter_col = CompQAPattern.comp_chapter_id if comp_chapter_id else CompQAPattern.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _result = await session.exec(select(func.max(CompQAPattern.sort_order)).where(filter_col == filter_val))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]

        obj = CompQAPattern(
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=(max_order or 0) + 1,
            original_filename=file.filename,
        )
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "QA pattern uploaded", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qa-pattern")
async def get_comp_qa_patterns(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(CompQAPattern)
    if comp_chapter_id is not None:
        query = query.where(CompQAPattern.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompQAPattern.sub_chapter_id == sub_chapter_id)
    query = query.order_by(*sort_ordering(CompQAPattern))
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}


@router.put("/qa-pattern/order")
async def reorder_comp_qa_patterns(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    filter_col = CompQAPattern.comp_chapter_id if comp_chapter_id else CompQAPattern.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(CompQAPattern).where(filter_col == filter_val, CompQAPattern.id.in_(payload.ids)))
    patterns = _result.all()
    if len(patterns) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid pattern ids")
    p_map = {p.id: p for p in patterns}
    for index, pid in enumerate(payload.ids, start=1):
        p_map[pid].sort_order = index
        session.add(p_map[pid])
    await session.commit()
    return {"message": "QA pattern order updated"}


@router.delete("/qa-pattern/{pattern_id}")
async def delete_comp_qa_pattern(pattern_id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompQAPattern, pattern_id)
    if not obj:
        raise HTTPException(status_code=404, detail="QA pattern not found")
    await delete_embeddings_by_resource_id(session, pattern_id, COMP_COLLECTION_QA, "qa_id")
    if obj.file_url:
        delete_from_do(obj.file_url)
    await session.delete(obj)
    await session.commit()
    return {"message": "QA pattern deleted"}


@router.put("/qa-pattern/{pattern_id}")
async def update_comp_qa_pattern(
    pattern_id: int,
    file: UploadFile | None = File(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    obj = await session.get(CompQAPattern, pattern_id)
    if not obj:
        raise HTTPException(status_code=404, detail="QA pattern not found")
    if title is not None:
        obj.title = title
    if description is not None:
        obj.description = description
    if file:
        folder_id = obj.comp_chapter_id or obj.sub_chapter_id
        new_url = upload_to_do(file, f"comp/chapters/{folder_id}/llm-resources/qa-patterns")
        if obj.file_url:
            delete_from_do(obj.file_url)
        obj.file_url = new_url
        obj.original_filename = file.filename
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "QA pattern updated", "data": obj.dict()}


# ============================================================
# LLM Image Endpoints
# ============================================================

@router.post("/image")
async def upload_comp_image(
    file: UploadFile = File(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    try:
        comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
        folder_id = comp_chapter_id or sub_chapter_id
        do_path = f"comp/chapters/{folder_id}/llm-resources/images"
        compressed = compress_image(file)
        file_url = upload_to_do(compressed, do_path)

        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        filter_col = CompLLMImage.comp_chapter_id if comp_chapter_id else CompLLMImage.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _result = await session.exec(select(func.max(CompLLMImage.sort_order)).where(filter_col == filter_val))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]

        obj = CompLLMImage(
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            title=title,
            description=description,
            file_url=file_url,
            tags=tags_list,
            sort_order=(max_order or 0) + 1,
            original_filename=file.filename,
        )
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Image uploaded", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image")
async def get_comp_images(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(CompLLMImage)
    if comp_chapter_id is not None:
        query = query.where(CompLLMImage.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompLLMImage.sub_chapter_id == sub_chapter_id)
    query = query.order_by(*sort_ordering(CompLLMImage))
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}


@router.put("/image/order")
async def reorder_comp_images(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    filter_col = CompLLMImage.comp_chapter_id if comp_chapter_id else CompLLMImage.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(CompLLMImage).where(filter_col == filter_val, CompLLMImage.id.in_(payload.ids)))
    images = _result.all()
    if len(images) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid image ids")
    img_map = {i.id: i for i in images}
    for index, iid in enumerate(payload.ids, start=1):
        img_map[iid].sort_order = index
        session.add(img_map[iid])
    await session.commit()
    return {"message": "Image order updated"}


@router.delete("/image/{image_id}")
async def delete_comp_image(image_id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompLLMImage, image_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Image not found")
    await delete_embeddings_by_resource_id(session, image_id, COMP_COLLECTION_IMAGES, "image_id")
    if obj.file_url:
        delete_from_do(obj.file_url)
    await session.delete(obj)
    await session.commit()
    return {"message": "Image deleted"}


@router.put("/image/{image_id}")
async def update_comp_image(
    image_id: int,
    file: UploadFile | None = File(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    obj = await session.get(CompLLMImage, image_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Image not found")
    if title is not None:
        obj.title = title
    if description is not None:
        obj.description = description
    if tags is not None:
        obj.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if file:
        folder_id = obj.comp_chapter_id or obj.sub_chapter_id
        compressed = compress_image(file)
        new_url = upload_to_do(compressed, f"comp/chapters/{folder_id}/llm-resources/images")
        if obj.file_url:
            delete_from_do(obj.file_url)
        obj.file_url = new_url
        obj.original_filename = file.filename
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Image updated", "data": obj.dict()}


# ============================================================
# Artifacts Endpoints
# ============================================================

@router.get("/artifacts")
async def get_comp_artifacts(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(CompChapterArtifact)
    if comp_chapter_id is not None:
        query = query.where(CompChapterArtifact.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompChapterArtifact.sub_chapter_id == sub_chapter_id)
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}


@router.put("/artifacts/{artifact_id}")
async def update_comp_artifact(
    artifact_id: int,
    content: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    artifact = await session.get(CompChapterArtifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if content is not None:
        artifact.content = content
        artifact.status = "completed"
    session.add(artifact)
    await session.commit()
    await session.refresh(artifact)
    return {"message": "Artifact updated", "data": artifact.dict()}
