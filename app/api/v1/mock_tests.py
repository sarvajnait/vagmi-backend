from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.mock_tests import (
    MockTest, MockTestCreate, MockTestUpdate, MockTestQuestion, MockTestQuestionCreate, MockTestQuestionUpdate,
)
from app.models.competitive_hierarchy import Level
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
# Mock Tests
# ============================================================

@router.get("/mock-tests")
async def get_mock_tests(level_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(MockTest)
    if level_id is not None:
        query = query.where(MockTest.level_id == level_id)
    query = query.order_by(*sort_ordering(MockTest))
    result = await session.exec(query)
    tests = result.all()
    data = []
    for t in tests:
        row = t.dict()
        _count = await session.exec(select(func.count()).where(MockTestQuestion.mock_test_id == t.id))
        q_count = _count.first()
        if isinstance(q_count, tuple):
            q_count = q_count[0]
        row["question_count"] = q_count or 0
        data.append(row)
    return {"data": data}


@router.post("/mock-tests")
async def create_mock_test(payload: MockTestCreate, session: AsyncSession = Depends(get_session)):
    try:
        level = await session.get(Level, payload.level_id)
        if not level:
            raise HTTPException(status_code=404, detail="Level not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(MockTest.sort_order)).where(MockTest.level_id == payload.level_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = MockTest(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Mock test created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/mock-tests/order")
async def reorder_mock_tests(
    payload: OrderUpdate,
    level_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")
    _result = await session.exec(
        select(MockTest).where(MockTest.level_id == level_id, MockTest.id.in_(payload.ids))
    )
    tests = _result.all()
    if len(tests) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid mock test ids")
    t_map = {t.id: t for t in tests}
    for index, tid in enumerate(payload.ids, start=1):
        t_map[tid].sort_order = index
        session.add(t_map[tid])
    await session.commit()
    return {"message": "Mock test order updated"}


@router.put("/mock-tests/{test_id}")
async def update_mock_test(test_id: int, payload: MockTestUpdate, session: AsyncSession = Depends(get_session)):
    obj = await session.get(MockTest, test_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Mock test not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Mock test updated", "data": obj.dict()}


@router.delete("/mock-tests/{test_id}")
async def delete_mock_test(test_id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(MockTest, test_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Mock test not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Mock test deleted"}


# ============================================================
# Mock Test Questions
# ============================================================

@router.get("/mock-test-questions")
async def get_mock_test_questions(
    mock_test_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(MockTestQuestion)
    if mock_test_id is not None:
        query = query.where(MockTestQuestion.mock_test_id == mock_test_id)
    query = query.order_by(*sort_ordering(MockTestQuestion))
    result = await session.exec(query)
    return {"data": [q.dict() for q in result.all()]}


@router.post("/mock-test-questions")
async def create_mock_test_question(payload: MockTestQuestionCreate, session: AsyncSession = Depends(get_session)):
    try:
        test = await session.get(MockTest, payload.mock_test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Mock test not found")
        if payload.sort_order is None:
            _r = await session.exec(select(func.max(MockTestQuestion.sort_order)).where(MockTestQuestion.mock_test_id == payload.mock_test_id))
            max_order = _r.first()
            if isinstance(max_order, tuple):
                max_order = max_order[0]
            payload = payload.copy(update={"sort_order": (max_order or 0) + 1})
        obj = MockTestQuestion(**payload.dict())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"message": "Question created", "data": obj.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/mock-test-questions/order")
async def reorder_mock_test_questions(
    payload: OrderUpdate,
    mock_test_id: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    if len(payload.ids) != len(set(payload.ids)):
        raise HTTPException(status_code=400, detail="Duplicate ids provided")
    _result = await session.exec(
        select(MockTestQuestion).where(
            MockTestQuestion.mock_test_id == mock_test_id,
            MockTestQuestion.id.in_(payload.ids),
        )
    )
    questions = _result.all()
    if len(questions) != len(payload.ids):
        raise HTTPException(status_code=400, detail="Invalid question ids")
    q_map = {q.id: q for q in questions}
    for index, qid in enumerate(payload.ids, start=1):
        q_map[qid].sort_order = index
        session.add(q_map[qid])
    await session.commit()
    return {"message": "Question order updated"}


@router.put("/mock-test-questions/{question_id}")
async def update_mock_test_question(
    question_id: int, payload: MockTestQuestionUpdate, session: AsyncSession = Depends(get_session)
):
    obj = await session.get(MockTestQuestion, question_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Question not found")
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return {"message": "Question updated", "data": obj.dict()}


@router.delete("/mock-test-questions/{question_id}")
async def delete_mock_test_question(question_id: int, session: AsyncSession = Depends(get_session)):
    obj = await session.get(MockTestQuestion, question_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Question not found")
    await session.delete(obj)
    await session.commit()
    return {"message": "Question deleted"}
