from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Query
from pydantic import BaseModel
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from sqlalchemy import func, case
from app.models import (
    Chapter,
    StudentTextbook,
    StudentNotes,
    StudentVideo,
    PreviousYearQuestionPaper,
    Subject,
)
from app.services.database import get_session
from app.utils.files import upload_to_do, delete_from_do

router = APIRouter()


class OrderUpdate(BaseModel):
    ids: list[int]


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


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
    """Upload a textbook directly to DigitalOcean and add to DB/vector store."""
    try:

        do_path = f"chapters/{chapter_id}/student-content/textbooks"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(
            select(func.max(StudentTextbook.sort_order)).where(
                StudentTextbook.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        textbook = StudentTextbook(
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

        return {
            "message": "Document uploaded",
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/textbook",
    description=(
        "List student textbooks grouped by chapter. "
        "If subject_id is provided it takes priority over chapter_id and returns "
        "all textbooks for all chapters in that subject. "
        "If only chapter_id is provided, returns textbooks for that chapter."
    ),
)
async def get_textbooks(
    subject_id: Optional[int] = None,
    chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """Get textbooks grouped by chapter, filtered by subject or chapter."""
    try:
        chapters = []
        if subject_id is not None:
            subject = await session.get(Subject, subject_id)
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
            _result = await session.exec(
                select(Chapter).where(Chapter.subject_id == subject_id)
            )
            chapters = _result.all()
            if not chapters:
                return {"data": []}
        elif chapter_id is not None:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            chapters = [chapter]
        else:
            _result = await session.exec(select(Chapter))
            chapters = _result.all()

        chapter_ids = [c.id for c in chapters]
        textbooks_query = select(StudentTextbook)
        if chapter_ids:
            textbooks_query = textbooks_query.where(
                StudentTextbook.chapter_id.in_(chapter_ids)
            )
        textbooks_query = textbooks_query.order_by(*sort_ordering(StudentTextbook))
        _result = await session.exec(textbooks_query)
        textbooks = _result.all()

        textbooks_by_chapter: dict[int, list[dict]] = {c.id: [] for c in chapters}
        for textbook in textbooks:
            textbooks_by_chapter.setdefault(textbook.chapter_id, []).append(
                textbook.dict()
            )

        return {
            "data": [
                {
                    "chapter_id": chapter.id,
                    "chapter_name": chapter.name,
                    "textbooks": textbooks_by_chapter.get(chapter.id, []),
                }
                for chapter in chapters
            ]
        }

    except HTTPException:
        raise
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
        select(StudentTextbook).where(
            StudentTextbook.chapter_id == chapter_id,
            StudentTextbook.id.in_(payload.ids),
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
async def delete_textbook(
    textbook_id: int, session: AsyncSession = Depends(get_session)
):
    """Delete a textbook from DB, vector store, and DigitalOcean Spaces."""
    try:
        textbook = await session.get(StudentTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        if textbook.file_url:
            delete_from_do(textbook.file_url)

        await session.delete(textbook)
        await session.commit()

        return {"message": f"Textbook '{textbook.title}' deleted successfully"}

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
    """Update a textbook's metadata or file."""
    try:
        textbook = await session.get(StudentTextbook, textbook_id)
        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        if chapter_id is not None:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            textbook.chapter_id = chapter_id

        if title is not None:
            textbook.title = title
        if description is not None:
            textbook.description = description

        if file:
            do_path = f"chapters/{textbook.chapter_id}/student-content/textbooks"
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

        return {"message": "Textbook updated", "data": textbook.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating textbook: {e}")
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
    session: AsyncSession = Depends(get_session),
):
    """
    Upload a note file directly to DigitalOcean and add to DB.
    """
    try:
        do_path = f"chapters/{chapter_id}/student-content/notes"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(
            select(func.max(StudentNotes.sort_order)).where(
                StudentNotes.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        note = StudentNotes(
            chapter_id=chapter_id,
            title=title,
            description=description,
            file_url=file_url,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)

        return {"message": "Note uploaded", "data": note.dict()}

    except Exception as e:
        logger.error(f"Error uploading note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notes")
async def get_notes(
    chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """Get all notes or filter by chapter."""
    try:
        query = select(StudentNotes)
        if chapter_id is not None:
            query = query.where(StudentNotes.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(StudentNotes))
        _result = await session.exec(query)
        notes = _result.all()
        return {"data": [n.dict() for n in notes]}

    except Exception as e:
        logger.error(f"Error fetching notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/notes/order")
async def reorder_notes(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update notes order within a chapter."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(StudentNotes).where(
            StudentNotes.chapter_id == chapter_id,
            StudentNotes.id.in_(payload.ids),
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


@router.delete("/notes/{note_id}")
async def delete_note(note_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a note from DB and DigitalOcean Spaces."""
    try:
        note = await session.get(StudentNotes, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        if note.file_url:
            try:
                delete_from_do(note.file_url)
            except Exception as e:
                logger.error(f"Error deleting note file from DO: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        await session.delete(note)
        await session.commit()

        return {"message": f"Note '{note.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/notes/{note_id}")
async def update_note(
    note_id: int,
    file: UploadFile | None = File(None),
    chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update a student note's metadata or file."""
    try:
        note = await session.get(StudentNotes, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        if chapter_id is not None:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            note.chapter_id = chapter_id

        if title is not None:
            note.title = title
        if description is not None:
            note.description = description

        if file:
            do_path = f"chapters/{note.chapter_id}/student-content/notes"
            new_file_url = upload_to_do(file, do_path)

            if note.file_url:
                delete_from_do(note.file_url)

            note.file_url = new_file_url
            note.original_filename = file.filename
            if title is None:
                note.title = file.filename

        session.add(note)
        await session.commit()
        await session.refresh(note)

        return {"message": "Note updated", "data": note.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating note: {e}")
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
    session: AsyncSession = Depends(get_session),
):
    """
    Upload a video file directly to DigitalOcean and add to DB.
    """
    try:
        do_path = f"chapters/{chapter_id}/student-content/videos"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(
            select(func.max(StudentVideo.sort_order)).where(
                StudentVideo.chapter_id == chapter_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        video = StudentVideo(
            chapter_id=chapter_id,
            title=title,
            description=description,
            file_url=file_url,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(video)
        await session.commit()
        await session.refresh(video)

        return {"message": "Video uploaded", "data": video.model_dump()}

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos")
async def get_videos(
    chapter_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """Get all videos or filter by chapter."""
    try:
        query = select(StudentVideo)
        if chapter_id is not None:
            query = query.where(StudentVideo.chapter_id == chapter_id)
        query = query.order_by(*sort_ordering(StudentVideo))
        _result = await session.exec(query)
        videos = _result.all()
        return {"data": [v.dict() for v in videos]}

    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/videos/order")
async def reorder_videos(
    payload: OrderUpdate,
    chapter_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update video order within a chapter."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(StudentVideo).where(
            StudentVideo.chapter_id == chapter_id,
            StudentVideo.id.in_(payload.ids),
        )
    )
    videos = _result.all()
    if len(videos) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid video ids for chapter")

    video_map = {v.id: v for v in videos}
    for index, video_id in enumerate(payload.ids, start=1):
        video_map[video_id].sort_order = index
        session.add(video_map[video_id])

    await session.commit()
    return {"message": "Video order updated"}


@router.delete("/videos/{video_id}")
async def delete_video(video_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a video from DB and DigitalOcean Spaces."""
    try:
        video = await session.get(StudentVideo, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if video.file_url:
            try:
                delete_from_do(video.file_url)
            except Exception as e:
                logger.error(f"Error deleting video file from DO: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        await session.delete(video)
        await session.commit()

        return {"message": f"Video '{video.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/videos/{video_id}")
async def update_video(
    video_id: int,
    file: UploadFile | None = File(None),
    chapter_id: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update a student video's metadata or file."""
    try:
        video = await session.get(StudentVideo, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if chapter_id is not None:
            chapter = await session.get(Chapter, chapter_id)
            if not chapter:
                raise HTTPException(status_code=404, detail="Chapter not found")
            video.chapter_id = chapter_id

        if title is not None:
            video.title = title
        if description is not None:
            video.description = description

        if file:
            do_path = f"chapters/{video.chapter_id}/student-content/videos"
            new_file_url = upload_to_do(file, do_path)

            if video.file_url:
                delete_from_do(video.file_url)

            video.file_url = new_file_url
            video.original_filename = file.filename
            if title is None:
                video.title = file.filename

        session.add(video)
        await session.commit()
        await session.refresh(video)

        return {"message": "Video updated", "data": video.dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating video: {e}")
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
    session: AsyncSession = Depends(get_session),
):
    """Upload a previous year question paper and store it against a subject."""
    try:
        subject = await session.get(Subject, subject_id)
        if not subject:
            raise HTTPException(status_code=400, detail="Subject not found")

        do_path = f"subjects/{subject_id}/student-content/previous-year-question-papers"
        file_url = upload_to_do(file, do_path)

        _result = await session.exec(
            select(func.max(PreviousYearQuestionPaper.sort_order)).where(
                PreviousYearQuestionPaper.subject_id == subject_id
            )
        )
        max_order = _result.first()
        if isinstance(max_order, tuple):
            max_order = max_order[0]
        next_order = (max_order or 0) + 1

        paper = PreviousYearQuestionPaper(
            subject_id=subject_id,
            title=title,
            num_pages=num_pages,
            file_url=file_url,
            is_premium=is_premium,
            enabled=enabled,
            sort_order=next_order,
            original_filename=file.filename,
        )
        session.add(paper)
        await session.commit()
        await session.refresh(paper)

        return {
            "message": "Previous year question paper uploaded",
            "data": paper.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error uploading previous year paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/previous-year-question-papers")
async def get_previous_year_question_papers(
    subject_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """List previous year question papers, optionally filtered by subject."""
    try:
        query = select(PreviousYearQuestionPaper)
        if subject_id is not None:
            query = query.where(PreviousYearQuestionPaper.subject_id == subject_id)
        query = query.order_by(*sort_ordering(PreviousYearQuestionPaper))
        _result = await session.exec(query)
        papers = _result.all()
        return {"data": [p.dict() for p in papers]}

    except Exception as e:
        logger.error(f"Error fetching previous year papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/previous-year-question-papers/order")
async def reorder_previous_year_question_papers(
    payload: OrderUpdate,
    subject_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Update previous year question paper order within a subject."""
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")

    _result = await session.exec(
        select(PreviousYearQuestionPaper).where(
            PreviousYearQuestionPaper.subject_id == subject_id,
            PreviousYearQuestionPaper.id.in_(payload.ids),
        )
    )
    papers = _result.all()
    if len(papers) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid paper ids for subject")

    paper_map = {p.id: p for p in papers}
    for index, paper_id in enumerate(payload.ids, start=1):
        paper_map[paper_id].sort_order = index
        session.add(paper_map[paper_id])

    await session.commit()
    return {"message": "Previous year paper order updated"}


@router.get("/previous-year-question-papers/{paper_id}")
async def get_previous_year_question_paper(
    paper_id: int, session: AsyncSession = Depends(get_session)
):
    """Get a single previous year question paper by ID."""
    try:
        paper = await session.get(PreviousYearQuestionPaper, paper_id)
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
    session: AsyncSession = Depends(get_session),
):
    """Update metadata, file, or subject for a previous year question paper."""
    try:
        paper = await session.get(PreviousYearQuestionPaper, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Previous year paper not found")

        if subject_id is not None:
            subject = await session.get(Subject, subject_id)
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
            paper.original_filename = file.filename

        session.add(paper)
        await session.commit()
        await session.refresh(paper)

        return {
            "message": "Previous year question paper updated",
            "data": paper.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error updating previous year paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/previous-year-question-papers/{paper_id}")
async def delete_previous_year_question_paper(
    paper_id: int, session: AsyncSession = Depends(get_session)
):
    """Delete a previous year question paper and its file."""
    try:
        paper = await session.get(PreviousYearQuestionPaper, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Previous year paper not found")

        if paper.file_url:
            try:
                delete_from_do(paper.file_url)
            except Exception as e:
                logger.error(f"Error deleting previous year paper file from DO: {e}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

        await session.delete(paper)
        await session.commit()
        return {"message": f"Previous year paper '{paper.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error deleting previous year paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
