from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select, text
from typing import Dict, List, Optional
from app.services.database import get_session
from app.schemas import HierarchyFilter
from app.core.langgraph.graph import EducationPlatform
from app.models import *
from app.schemas import *
from loguru import logger

router = APIRouter()
platform = EducationPlatform()


def fetch_hierarchy_options(
    filter_params: HierarchyFilter, session: Session
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
        class_levels = session.exec(
            select(ClassLevel).order_by(
                text(
                    "CASE WHEN name ~ '^[0-9]+$' THEN CAST(name AS INTEGER) ELSE 999 END, name"
                )
            )
        ).all()
        result["class_levels"] = [{"id": cl.id, "name": cl.name} for cl in class_levels]

        # Get boards filtered by class_level_id
        if filter_params.class_level_id:
            boards = session.exec(
                select(Board)
                .where(Board.class_level_id == filter_params.class_level_id)
                .order_by(Board.name)
            ).all()
            result["boards"] = [
                {"id": b.id, "name": b.name, "class_level_id": b.class_level_id}
                for b in boards
            ]

        # Get mediums filtered by board_id
        if filter_params.board_id:
            mediums = session.exec(
                select(Medium)
                .where(Medium.board_id == filter_params.board_id)
                .order_by(Medium.name)
            ).all()
            result["mediums"] = [
                {"id": m.id, "name": m.name, "board_id": m.board_id} for m in mediums
            ]

        # Get subjects filtered by medium_id
        if filter_params.medium_id:
            subjects = session.exec(
                select(Subject)
                .where(Subject.medium_id == filter_params.medium_id)
                .order_by(Subject.name)
            ).all()
            result["subjects"] = [
                {"id": s.id, "name": s.name, "medium_id": s.medium_id} for s in subjects
            ]

        # Get chapters filtered by subject_id
        if filter_params.subject_id:
            chapters = session.exec(
                select(Chapter)
                .where(Chapter.subject_id == filter_params.subject_id)
                .order_by(Chapter.chapter_number, Chapter.name)
            ).all()
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
    session: Session = Depends(get_session),
):
    """Get hierarchy options with IDs for document management"""
    filter_params = HierarchyFilter(
        class_level_id=class_level_id,
        board_id=board_id,
        medium_id=medium_id,
        subject_id=subject_id,
    )
    options = fetch_hierarchy_options(filter_params, session)
    return {"data": options}
