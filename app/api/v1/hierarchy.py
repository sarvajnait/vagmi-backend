from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, text
from typing import Dict, List, Optional
from app.services.database import get_session
from app.schemas import HierarchyFilter
from app.core.agents.graph import EducationPlatform
from app.models import *
from app.schemas import *
from loguru import logger

router = APIRouter()
platform = EducationPlatform()


async def fetch_hierarchy_options(
    filter_params: HierarchyFilter, session: AsyncSession
) -> Dict[str, List[Dict]]:
    """Fetch hierarchy options returning objects with id and name"""
    try:
        result = {
            "class_levels": [],
            "boards": [],
            "mediums": [],
            "subjects": [],
            "chapters": [],
        }

        # Get all class levels
        _result = await session.exec(
            select(ClassLevel).order_by(
                text(
                    "CASE WHEN name ~ '^[0-9]+$' THEN CAST(name AS INTEGER) ELSE 999 END, name"
                )
            ))
        class_levels = _result.all()
        result["class_levels"] = [{"id": cl.id, "name": cl.name} for cl in class_levels]

        # Get boards filtered by class_level_id
        if filter_params.class_level_id:
            _result = await session.exec(
                select(Board)
                .where(Board.class_level_id == filter_params.class_level_id)
                .order_by(Board.name))
            boards = _result.all()
            result["boards"] = [
                {"id": b.id, "name": b.name, "class_level_id": b.class_level_id}
                for b in boards
            ]

        # Get mediums filtered by board_id
        if filter_params.board_id:
            _result = await session.exec(
                select(Medium)
                .where(Medium.board_id == filter_params.board_id)
                .order_by(Medium.name))
            mediums = _result.all()
            result["mediums"] = [
                {"id": m.id, "name": m.name, "board_id": m.board_id} for m in mediums
            ]

        # Get subjects filtered by medium_id
        if filter_params.medium_id:
            _result = await session.exec(
                select(Subject)
                .where(Subject.medium_id == filter_params.medium_id)
                .order_by(Subject.name))
            subjects = _result.all()
            result["subjects"] = [
                {"id": s.id, "name": s.name, "medium_id": s.medium_id} for s in subjects
            ]

        # Get chapters filtered by subject_id
        if filter_params.subject_id:
            _result = await session.exec(
                select(Chapter)
                .where(Chapter.subject_id == filter_params.subject_id)
                .order_by(Chapter.chapter_number, Chapter.name))
            chapters = _result.all()
            result["chapters"] = [
                {"id": ch.id, "name": ch.name, "subject_id": ch.subject_id}
                for ch in chapters
            ]

        return result

    except Exception as e:
        logger.error(f"Error getting hierarchy options: {e}")
        return {
            "class_levels": [],
            "boards": [],
            "mediums": [],
            "subjects": [],
            "chapters": [],
        }


@router.get("/")
async def get_hierarchy_options(
    class_level_id: Optional[int] = Query(None),
    board_id: Optional[int] = Query(None),
    medium_id: Optional[int] = Query(None),
    subject_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """Get hierarchy options with IDs for document management"""
    filter_params = HierarchyFilter(
        class_level_id=class_level_id,
        board_id=board_id,
        medium_id=medium_id,
        subject_id=subject_id,
    )
    options = await fetch_hierarchy_options(filter_params, session)
    return {"data": options}
