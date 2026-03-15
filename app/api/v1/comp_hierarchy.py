from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.competitive_hierarchy import (
    ExamCategory, ExamCategoryCreate, ExamCategoryRead,
    Exam, ExamCreate, ExamRead,
    CompExamMedium, CompExamMediumCreate, CompExamMediumRead,
    Level, LevelCreate, LevelRead,
    CompSubject, CompSubjectCreate, CompSubjectRead,
    CompChapter, CompChapterCreate, CompChapterUpdate, CompChapterRead,
    SubChapter, SubChapterCreate, SubChapterUpdate, SubChapterRead,
)
from app.services.database import get_session

router = APIRouter()


class OrderUpdate(BaseModel):
    ids: list[int]


def sort_ordering(model):
    return [
        case((model.sort_order == None, 1), else_=0),
        model.sort_order,
        model.created_at,
    ]


# ============================================================
# Exam Categories
# ============================================================

@router.get("/exam-categories")
async def get_exam_categories(session: AsyncSession = Depends(get_session)):
    result = await session.exec(select(ExamCategory).order_by(*sort_ordering(ExamCategory)))
    return {"data": [r.dict() for r in result.all()]}


@router.post("/exam-categories")
async def create_exam_category(payload: ExamCategoryCreate, session: AsyncSession = Depends(get_session)):
    try:
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(ExamCategory.sort_order)))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = ExamCategory(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Exam category created", "data": obj.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/exam-categories/{id}")
async def update_exam_category(id: int, payload: ExamCategoryCreate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(ExamCategory, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Exam category not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Exam category updated", "data": obj.dict()}


@router.delete("/exam-categories/{id}")
async def delete_exam_category(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(ExamCategory, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Exam category not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Exam category deleted"}


# ============================================================
# Exams
# ============================================================

@router.get("/exams")
async def get_exams(exam_category_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(Exam)
    if exam_category_id is not None:
        query = query.where(Exam.exam_category_id == exam_category_id)
    query = query.order_by(*sort_ordering(Exam))
    result = await session.exec(query)
    exams = result.all()
    data = []
    for e in exams:
        row = e.dict()
        if e.exam_category_id:
            cat = await session.get(ExamCategory, e.exam_category_id)
            row["exam_category_name"] = cat.name if cat else None
        data.append(row)
    return {"data": data}


@router.post("/exams")
async def create_exam(payload: ExamCreate, session: AsyncSession = Depends(get_session)):
    try:
        cat = await session.get(ExamCategory, payload.exam_category_id)
        if not cat:
            raise HTTPException(status_code=404, detail="Exam category not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(Exam.sort_order)).where(Exam.exam_category_id == payload.exam_category_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = Exam(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Exam created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/exams/{id}")
async def update_exam(id: int, payload: ExamCreate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(Exam, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Exam not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Exam updated", "data": obj.dict()}


@router.delete("/exams/{id}")
async def delete_exam(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(Exam, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Exam not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Exam deleted"}


# ============================================================
# Comp Exam Mediums
# ============================================================

@router.get("/mediums")
async def get_comp_mediums(exam_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(CompExamMedium)
    if exam_id is not None:
        query = query.where(CompExamMedium.exam_id == exam_id)
    query = query.order_by(*sort_ordering(CompExamMedium))
    result = await session.exec(query)
    mediums = result.all()
    data = []
    for m in mediums:
        row = m.dict()
        exam = await session.get(Exam, m.exam_id)
        row["exam_name"] = exam.name if exam else None
        data.append(row)
    return {"data": data}


@router.post("/mediums")
async def create_comp_medium(payload: CompExamMediumCreate, session: AsyncSession = Depends(get_session)):
    try:
        exam = await session.get(Exam, payload.exam_id)
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(CompExamMedium.sort_order)).where(CompExamMedium.exam_id == payload.exam_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = CompExamMedium(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Medium created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/mediums/{id}")
async def update_comp_medium(id: int, payload: CompExamMediumCreate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompExamMedium, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Medium not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Medium updated", "data": obj.dict()}


@router.delete("/mediums/{id}")
async def delete_comp_medium(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompExamMedium, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Medium not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Medium deleted"}


# ============================================================
# Levels
# ============================================================

@router.get("/levels")
async def get_levels(medium_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(Level)
    if medium_id is not None:
        query = query.where(Level.medium_id == medium_id)
    query = query.order_by(*sort_ordering(Level))
    result = await session.exec(query)
    levels = result.all()
    data = []
    for lv in levels:
        row = lv.dict()
        medium = await session.get(CompExamMedium, lv.medium_id)
        row["medium_name"] = medium.name if medium else None
        data.append(row)
    return {"data": data}


@router.post("/levels")
async def create_level(payload: LevelCreate, session: AsyncSession = Depends(get_session)):
    try:
        medium = await session.get(CompExamMedium, payload.medium_id)
        if not medium:
            raise HTTPException(status_code=404, detail="Medium not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(Level.sort_order)).where(Level.medium_id == payload.medium_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = Level(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Level created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/levels/{id}")
async def update_level(id: int, payload: LevelCreate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(Level, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Level not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Level updated", "data": obj.dict()}


@router.delete("/levels/{id}")
async def delete_level(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(Level, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Level not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Level deleted"}


# ============================================================
# Comp Subjects
# ============================================================

@router.get("/subjects")
async def get_comp_subjects(level_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(CompSubject)
    if level_id is not None:
        query = query.where(CompSubject.level_id == level_id)
    query = query.order_by(*sort_ordering(CompSubject))
    result = await session.exec(query)
    subjects = result.all()
    data = []
    for s in subjects:
        row = s.dict()
        lv = await session.get(Level, s.level_id)
        row["level_name"] = lv.name if lv else None
        data.append(row)
    return {"data": data}


@router.post("/subjects")
async def create_comp_subject(payload: CompSubjectCreate, session: AsyncSession = Depends(get_session)):
    try:
        lv = await session.get(Level, payload.level_id)
        if not lv:
            raise HTTPException(status_code=404, detail="Level not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(CompSubject.sort_order)).where(CompSubject.level_id == payload.level_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = CompSubject(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Subject created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/subjects/{id}")
async def update_comp_subject(id: int, payload: CompSubjectCreate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompSubject, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Subject not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Subject updated", "data": obj.dict()}


@router.delete("/subjects/{id}")
async def delete_comp_subject(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompSubject, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Subject not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Subject deleted"}


# ============================================================
# Comp Chapters
# ============================================================

@router.get("/chapters")
async def get_comp_chapters(subject_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(CompChapter)
    if subject_id is not None:
        query = query.where(CompChapter.subject_id == subject_id)
    query = query.order_by(*sort_ordering(CompChapter))
    result = await session.exec(query)
    chapters = result.all()
    data = []
    for c in chapters:
        row = c.dict()
        subj = await session.get(CompSubject, c.subject_id)
        row["subject_name"] = subj.name if subj else None
        data.append(row)
    return {"data": data}


@router.get("/chapters/{id}")
async def get_comp_chapter(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompChapter, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Chapter not found")
    row = obj.dict()
    subj = await session.get(CompSubject, obj.subject_id)
    row["subject_name"] = subj.name if subj else None
    return {"data": row}


@router.post("/chapters")
async def create_comp_chapter(payload: CompChapterCreate, session: AsyncSession = Depends(get_session)):
    try:
        subj = await session.get(CompSubject, payload.subject_id)
        if not subj:
            raise HTTPException(status_code=404, detail="Subject not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(CompChapter.sort_order)).where(CompChapter.subject_id == payload.subject_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = CompChapter(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Chapter created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/chapters/{id}")
async def update_comp_chapter(id: int, payload: CompChapterUpdate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompChapter, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Chapter not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Chapter updated", "data": obj.dict()}


@router.delete("/chapters/{id}")
async def delete_comp_chapter(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(CompChapter, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Chapter not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Chapter deleted"}


# ============================================================
# Sub Chapters
# ============================================================

@router.get("/sub-chapters")
async def get_sub_chapters(chapter_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(SubChapter)
    if chapter_id is not None:
        query = query.where(SubChapter.chapter_id == chapter_id)
    query = query.order_by(*sort_ordering(SubChapter))
    result = await session.exec(query)
    sub_chapters = result.all()
    data = []
    for sc in sub_chapters:
        row = sc.dict()
        ch = await session.get(CompChapter, sc.chapter_id)
        row["chapter_name"] = ch.name if ch else None
        data.append(row)
    return {"data": data}


@router.post("/sub-chapters")
async def create_sub_chapter(payload: SubChapterCreate, session: AsyncSession = Depends(get_session)):
    try:
        ch = await session.get(CompChapter, payload.chapter_id)
        if not ch:
            raise HTTPException(status_code=404, detail="Chapter not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(SubChapter.sort_order)).where(SubChapter.chapter_id == payload.chapter_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = SubChapter(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Sub-chapter created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sub-chapters/{id}")
async def update_sub_chapter(id: int, payload: SubChapterUpdate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(SubChapter, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Sub-chapter not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Sub-chapter updated", "data": obj.dict()}


@router.delete("/sub-chapters/{id}")
async def delete_sub_chapter(id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(SubChapter, id)
    if not obj:
        raise HTTPException(status_code=404, detail="Sub-chapter not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Sub-chapter deleted"}
