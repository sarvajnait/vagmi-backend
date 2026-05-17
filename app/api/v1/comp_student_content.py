from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, HTTPException, Form, Query
from pydantic import BaseModel
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from sqlalchemy import func, case
from app.models.comp_student_content import (
    CompStudentTextbook, CompStudentNote, CompStudentVideo, CompPreviousYearPaper,
)
from app.models import ActivityGenerationJob
from app.api.v1.admin.auth import get_current_user as get_current_admin
from app.services.database import get_session
from app.utils.files import upload_to_do, delete_from_do, delete_prefix_from_do

router = APIRouter()


class OrderUpdate(BaseModel):
    ids: list[int]


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


def _resolve_fk(comp_chapter_id: Optional[int], sub_chapter_id: Optional[int]):
    if comp_chapter_id is None and sub_chapter_id is None:
        raise HTTPException(status_code=400, detail="Either comp_chapter_id or sub_chapter_id must be provided")
    return comp_chapter_id, sub_chapter_id


# ============================================================
# Student Textbook Endpoints
# ============================================================

@router.post("/textbook")
async def upload_comp_student_textbook(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
        folder_id = comp_chapter_id or sub_chapter_id
        do_path = f"comp/chapters/{folder_id}/student-content/textbooks"
        file_url = upload_to_do(file, do_path)

        filter_col = CompStudentTextbook.comp_chapter_id if comp_chapter_id else CompStudentTextbook.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _result = await session.exec(select(func.max(CompStudentTextbook.sort_order)).where(filter_col == filter_val))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]

        textbook = CompStudentTextbook(
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=(max_order or 0) + 1,
            original_filename=file.filename,
            audio_status="processing",
        )
        session.add(textbook)
        await session.commit()
        await session.refresh(textbook)

        job = ActivityGenerationJob(
            job_type="comp_audio_generation",
            status="pending",
            payload={
                "resource_type": "textbook",
                "resource_id": textbook.id,
                "file_url": file_url,
                "comp_chapter_id": comp_chapter_id,
                "sub_chapter_id": sub_chapter_id,
            },
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        from app.services.activity_jobs import enqueue_activity_job
        background_tasks.add_task(enqueue_activity_job, job.id)

        return {
            "message": "Textbook uploaded. Audio generation in progress.",
            "data": textbook.dict(),
            "job_id": job.id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading comp student textbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/textbook")
async def get_comp_student_textbooks(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        query = select(CompStudentTextbook)
        if comp_chapter_id is not None:
            query = query.where(CompStudentTextbook.comp_chapter_id == comp_chapter_id)
        elif sub_chapter_id is not None:
            query = query.where(CompStudentTextbook.sub_chapter_id == sub_chapter_id)
        query = query.order_by(*sort_ordering(CompStudentTextbook))
        result = await session.exec(query)
        return {"data": [r.dict() for r in result.all()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/textbook/order")
async def reorder_comp_student_textbooks(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    filter_col = CompStudentTextbook.comp_chapter_id if comp_chapter_id else CompStudentTextbook.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(CompStudentTextbook).where(filter_col == filter_val, CompStudentTextbook.id.in_(payload.ids)))
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
async def delete_comp_student_textbook(textbook_id: int, _admin=Depends(get_current_admin), session: AsyncSession = Depends(get_session)):
    textbook = await session.get(CompStudentTextbook, textbook_id)
    if not textbook:
        raise HTTPException(status_code=404, detail="Textbook not found")
    if textbook.file_url:
        delete_from_do(textbook.file_url)
    if textbook.audio_url:
        try:
            delete_from_do(textbook.audio_url)
        except Exception as e:
            logger.warning(f"Could not delete audio: {e}")
    await session.delete(textbook)
    await session.commit()
    return {"message": "Textbook deleted"}


@router.put("/textbook/{textbook_id}")
async def update_comp_student_textbook(
    background_tasks: BackgroundTasks,
    textbook_id: int,
    file: UploadFile | None = File(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        textbook = await session.get(CompStudentTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")
        if title is not None:
            textbook.title = title
        if description is not None:
            textbook.description = description
        job_id = None
        if file:
            folder_id = textbook.comp_chapter_id or textbook.sub_chapter_id
            new_url = upload_to_do(file, f"comp/chapters/{folder_id}/student-content/textbooks")
            if textbook.file_url:
                delete_from_do(textbook.file_url)
            if textbook.audio_url:
                try:
                    delete_from_do(textbook.audio_url)
                except Exception:
                    pass
            textbook.file_url = new_url
            textbook.original_filename = file.filename
            if title is None:
                textbook.title = file.filename
            textbook.audio_url = None
            textbook.audio_status = "processing"
        session.add(textbook)
        await session.commit()
        await session.refresh(textbook)
        if file:
            job = ActivityGenerationJob(
                job_type="comp_audio_generation",
                status="pending",
                payload={
                    "resource_type": "textbook",
                    "resource_id": textbook.id,
                    "file_url": textbook.file_url,
                    "comp_chapter_id": textbook.comp_chapter_id,
                    "sub_chapter_id": textbook.sub_chapter_id,
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
# Student Notes Endpoints
# ============================================================

def _note_source(filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return "excel_upload"
    return "docx_upload"


@router.post("/notes")
async def upload_comp_student_note(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    language: str = Form("en"),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        filename = file.filename or ""
        if not (filename.lower().endswith(".docx") or filename.lower().endswith(".xlsx")):
            raise HTTPException(status_code=400, detail="Only .docx and .xlsx files are supported")

        comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
        folder_id = comp_chapter_id or sub_chapter_id
        do_path = f"comp/chapters/{folder_id}/student-content/notes"
        file_url = upload_to_do(file, do_path)
        source = _note_source(filename)

        filter_col = CompStudentNote.comp_chapter_id if comp_chapter_id else CompStudentNote.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _result = await session.exec(select(func.max(CompStudentNote.sort_order)).where(filter_col == filter_val))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]

        note = CompStudentNote(
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            title=title,
            description=description,
            file_url=file_url,
            original_filename=filename,
            sort_order=(max_order or 0) + 1,
            content_status="processing",
            is_published=False,
            source=source,
            language=language,
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)

        job = ActivityGenerationJob(
            job_type="comp_notes_convert",
            status="pending",
            payload={
                "note_id": note.id,
                "file_url": file_url,
                "source": source,
            },
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        from app.services.activity_jobs import enqueue_activity_job
        background_tasks.add_task(enqueue_activity_job, job.id)

        return {"message": "Note uploaded. Converting to markdown…", "data": note.dict(), "job_id": job.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading comp student note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notes")
async def get_comp_student_notes(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    query = select(CompStudentNote)
    if comp_chapter_id is not None:
        query = query.where(CompStudentNote.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompStudentNote.sub_chapter_id == sub_chapter_id)
    query = query.order_by(*sort_ordering(CompStudentNote))
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}




@router.get("/notes/published/{note_id}")
async def get_published_comp_note(note_id: int, session: AsyncSession = Depends(get_session)):
    """Student-facing: returns full note content including audio_url."""
    note = await session.get(CompStudentNote, note_id)
    if not note or not note.is_published:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"data": note.dict()}


@router.put("/notes/order")
async def reorder_comp_student_notes(
    payload: OrderUpdate,
    comp_chapter_id: Optional[int] = Query(None),
    sub_chapter_id: Optional[int] = Query(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    filter_col = CompStudentNote.comp_chapter_id if comp_chapter_id else CompStudentNote.sub_chapter_id
    filter_val = comp_chapter_id or sub_chapter_id
    _result = await session.exec(select(CompStudentNote).where(filter_col == filter_val, CompStudentNote.id.in_(payload.ids)))
    notes = _result.all()
    if len(notes) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid note ids")
    note_map = {n.id: n for n in notes}
    for index, nid in enumerate(payload.ids, start=1):
        note_map[nid].sort_order = index
        session.add(note_map[nid])
    await session.commit()
    return {"message": "Note order updated"}


@router.delete("/notes/{note_id}")
async def delete_comp_student_note(note_id: int, _admin=Depends(get_current_admin), session: AsyncSession = Depends(get_session)):
    note = await session.get(CompStudentNote, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    if note.file_url:
        try:
            delete_from_do(note.file_url)
        except Exception:
            pass
    if note.audio_url:
        try:
            delete_from_do(note.audio_url)
        except Exception:
            pass
    # Clean up inline images extracted from the DOCX
    try:
        delete_prefix_from_do(f"comp/notes/images/{note_id}/")
    except Exception:
        pass
    await session.delete(note)
    await session.commit()
    return {"message": "Note deleted"}


@router.put("/notes/{note_id}")
async def update_comp_student_note(
    background_tasks: BackgroundTasks,
    note_id: int,
    file: UploadFile | None = File(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    is_published: Optional[bool] = Form(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        note = await session.get(CompStudentNote, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        if title is not None:
            note.title = title
        if description is not None:
            note.description = description
        if language is not None:
            note.language = language
        if is_published is not None:
            note.is_published = is_published
        job_id = None
        if file:
            filename = file.filename or ""
            if not (filename.lower().endswith(".docx") or filename.lower().endswith(".xlsx")):
                raise HTTPException(status_code=400, detail="Only .docx and .xlsx files are supported")
            folder_id = note.comp_chapter_id or note.sub_chapter_id
            new_url = upload_to_do(file, f"comp/chapters/{folder_id}/student-content/notes")
            if note.file_url:
                try:
                    delete_from_do(note.file_url)
                except Exception:
                    pass
            source = _note_source(filename)
            note.file_url = new_url
            note.original_filename = filename
            note.source = source
            note.content = None
            note.content_status = "processing"
            note.version = (note.version or 1) + 1
            note.word_count = None
            note.read_time_min = None
        session.add(note)
        await session.commit()
        await session.refresh(note)
        if file:
            job = ActivityGenerationJob(
                job_type="comp_notes_convert",
                status="pending",
                payload={"note_id": note.id, "file_url": note.file_url, "source": note.source},
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            from app.services.activity_jobs import enqueue_activity_job
            background_tasks.add_task(enqueue_activity_job, job.id)
            job_id = job.id
        response = {"message": "Note updated", "data": note.dict()}
        if job_id:
            response["job_id"] = job_id
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notes/{note_id}/regenerate")
async def regenerate_comp_note_markdown(
    note_id: int,
    background_tasks: BackgroundTasks,
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    """Admin: re-trigger markdown conversion for an existing note using its stored file."""
    note = await session.get(CompStudentNote, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    if not note.file_url:
        raise HTTPException(status_code=400, detail="Note has no source file to reprocess")
    if note.content_status == "processing":
        raise HTTPException(status_code=409, detail="Conversion already in progress")

    # Backfill source for notes uploaded before the field existed
    source = note.source or _note_source(note.original_filename or note.file_url or "")
    note.source = source

    note.content_status = "processing"
    note.content = None
    note.word_count = None
    note.read_time_min = None
    note.is_published = False
    note.version = (note.version or 1) + 1
    session.add(note)

    job = ActivityGenerationJob(
        job_type="comp_notes_convert",
        status="pending",
        payload={"note_id": note.id, "file_url": note.file_url, "source": source},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    from app.services.activity_jobs import enqueue_activity_job
    background_tasks.add_task(enqueue_activity_job, job.id)

    return {"message": "Markdown regeneration started", "job_id": job.id}


@router.post("/notes/{note_id}/generate-audio")
async def generate_comp_note_audio(
    note_id: int,
    background_tasks: BackgroundTasks,
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    """Admin: enqueue ElevenLabs audio generation with word-level sync for a published note."""
    note = await session.get(CompStudentNote, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    if note.content_status != "completed" or not note.content:
        raise HTTPException(status_code=400, detail="Note content must be fully converted before generating audio")
    if note.audio_status == "processing":
        raise HTTPException(status_code=409, detail="Audio generation already in progress")

    note.audio_status = "processing"
    session.add(note)

    job = ActivityGenerationJob(
        job_type="comp_notes_audio",
        status="pending",
        payload={"note_id": note_id},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    from app.services.activity_jobs import enqueue_activity_job
    background_tasks.add_task(enqueue_activity_job, job.id)

    return {"message": "Audio generation started", "job_id": job.id}


# ============================================================
# Student Video Endpoints
# ============================================================

@router.post("/video")
async def upload_comp_student_video(
    file: UploadFile = File(...),
    comp_chapter_id: Optional[int] = Form(None),
    sub_chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        comp_chapter_id, sub_chapter_id = _resolve_fk(comp_chapter_id, sub_chapter_id)
        folder_id = comp_chapter_id or sub_chapter_id
        do_path = f"comp/chapters/{folder_id}/student-content/videos"
        file_url = upload_to_do(file, do_path)

        filter_col = CompStudentVideo.comp_chapter_id if comp_chapter_id else CompStudentVideo.sub_chapter_id
        filter_val = comp_chapter_id or sub_chapter_id
        _result = await session.exec(select(func.max(CompStudentVideo.sort_order)).where(filter_col == filter_val))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]

        video = CompStudentVideo(
            comp_chapter_id=comp_chapter_id,
            sub_chapter_id=sub_chapter_id,
            title=title or file.filename,
            description=description,
            file_url=file_url,
            sort_order=(max_order or 0) + 1,
            original_filename=file.filename,
        )
        session.add(video)
        await session.commit()
        await session.refresh(video)
        return {"message": "Video uploaded", "data": video.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video")
async def get_comp_student_videos(
    comp_chapter_id: Optional[int] = None,
    sub_chapter_id: Optional[int] = None,
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    query = select(CompStudentVideo)
    if comp_chapter_id is not None:
        query = query.where(CompStudentVideo.comp_chapter_id == comp_chapter_id)
    elif sub_chapter_id is not None:
        query = query.where(CompStudentVideo.sub_chapter_id == sub_chapter_id)
    query = query.order_by(*sort_ordering(CompStudentVideo))
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}


@router.delete("/video/{video_id}")
async def delete_comp_student_video(video_id: int, _admin=Depends(get_current_admin), session: AsyncSession = Depends(get_session)):
    video = await session.get(CompStudentVideo, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.file_url:
        delete_from_do(video.file_url)
    await session.delete(video)
    await session.commit()
    return {"message": "Video deleted"}


@router.put("/video/{video_id}")
async def update_comp_student_video(
    video_id: int,
    file: UploadFile | None = File(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    video = await session.get(CompStudentVideo, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if title is not None:
        video.title = title
    if description is not None:
        video.description = description
    if file:
        folder_id = video.comp_chapter_id or video.sub_chapter_id
        new_url = upload_to_do(file, f"comp/chapters/{folder_id}/student-content/videos")
        if video.file_url:
            delete_from_do(video.file_url)
        video.file_url = new_url
        video.original_filename = file.filename
    session.add(video)
    await session.commit()
    await session.refresh(video)
    return {"message": "Video updated", "data": video.dict()}


# ============================================================
# Previous Year Papers (Level-scoped)
# ============================================================

@router.post("/previous-year-paper")
async def upload_comp_pyp(
    file: UploadFile = File(...),
    level_id: int = Form(...),
    title: str = Form(...),
    year: Optional[int] = Form(None),
    num_questions: Optional[int] = Form(None),
    num_pages: Optional[int] = Form(None),
    is_premium: bool = Form(False),
    enabled: bool = Form(True),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    try:
        do_path = f"comp/levels/{level_id}/previous-year-papers"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(select(func.max(CompPreviousYearPaper.sort_order)).where(CompPreviousYearPaper.level_id == level_id))
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]

        obj = CompPreviousYearPaper(
            level_id=level_id,
            title=title,
            year=year,
            num_questions=num_questions,
            num_pages=num_pages,
            file_url=file_url,
            is_premium=is_premium,
            enabled=enabled,
            sort_order=(max_order or 0) + 1,
            original_filename=file.filename,
        )
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Previous year paper uploaded", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/previous-year-paper")
async def get_comp_pyps(level_id: Optional[int] = None, _admin=Depends(get_current_admin), session: AsyncSession = Depends(get_session)):
    query = select(CompPreviousYearPaper)
    if level_id is not None:
        query = query.where(CompPreviousYearPaper.level_id == level_id)
    query = query.order_by(*sort_ordering(CompPreviousYearPaper))
    result = await session.exec(query)
    return {"data": [r.dict() for r in result.all()]}


@router.put("/previous-year-paper/order")
async def reorder_comp_pyps(
    payload: OrderUpdate,
    level_id: int = Query(...),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    _result = await session.exec(
        select(CompPreviousYearPaper).where(CompPreviousYearPaper.level_id == level_id, CompPreviousYearPaper.id.in_(payload.ids))
    )
    papers = _result.all()
    if len(papers) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid paper ids")
    p_map = {p.id: p for p in papers}
    for index, pid in enumerate(payload.ids, start=1):
        p_map[pid].sort_order = index
        session.add(p_map[pid])
    await session.commit()
    return {"message": "Paper order updated"}


@router.delete("/previous-year-paper/{paper_id}")
async def delete_comp_pyp(paper_id: int, _admin=Depends(get_current_admin), session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompPreviousYearPaper, paper_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Paper not found")
    if obj.file_url:
        delete_from_do(obj.file_url)
    await session.delete(obj)
    await session.commit()
    return {"message": "Paper deleted"}


@router.put("/previous-year-paper/{paper_id}")
async def update_comp_pyp(
    paper_id: int,
    file: UploadFile | None = File(None),
    title: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    num_questions: Optional[int] = Form(None),
    num_pages: Optional[int] = Form(None),
    is_premium: Optional[bool] = Form(None),
    enabled: Optional[bool] = Form(None),
    _admin=Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    obj = await session.get(CompPreviousYearPaper, paper_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Paper not found")
    if title is not None:
        obj.title = title
    if year is not None:
        obj.year = year
    if num_questions is not None:
        obj.num_questions = num_questions
    if num_pages is not None:
        obj.num_pages = num_pages
    if is_premium is not None:
        obj.is_premium = is_premium
    if enabled is not None:
        obj.enabled = enabled
    if file:
        new_url = upload_to_do(file, f"comp/levels/{obj.level_id}/previous-year-papers")
        if obj.file_url:
            delete_from_do(obj.file_url)
        obj.file_url = new_url
        obj.original_filename = file.filename
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Paper updated", "data": obj.dict()}
