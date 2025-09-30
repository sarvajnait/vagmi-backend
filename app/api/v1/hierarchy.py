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
) -> Dict[str, List[str]]:
    try:
        result = {
            "class_level": [],
            "board": [],
            "medium": [],
            "subject": [],
            "chapter": [],
        }

        class_levels = session.exec(
            select(ClassLevel).order_by(
                text(
                    "CASE WHEN name ~ '^[0-9]+$' THEN CAST(name AS INTEGER) ELSE 999 END, name"
                )
            )
        ).all()
        result["class_level"] = [cl.name for cl in class_levels]

        if filter_params.class_level:
            boards = session.exec(
                select(Board)
                .join(ClassLevel)
                .where(ClassLevel.name == filter_params.class_level)
                .order_by(Board.name)
            ).all()
            result["board"] = [b.name for b in boards]

        if filter_params.board and filter_params.class_level:
            mediums = session.exec(
                select(Medium)
                .join(Board)
                .join(ClassLevel)
                .where(
                    ClassLevel.name == filter_params.class_level,
                    Board.name == filter_params.board,
                )
                .order_by(Medium.name)
            ).all()
            result["medium"] = [m.name for m in mediums]

        if filter_params.medium and filter_params.board and filter_params.class_level:
            subjects = session.exec(
                select(Subject)
                .join(Medium)
                .join(Board)
                .join(ClassLevel)
                .where(
                    ClassLevel.name == filter_params.class_level,
                    Board.name == filter_params.board,
                    Medium.name == filter_params.medium,
                )
                .order_by(Subject.name)
            ).all()
            result["subject"] = [s.name for s in subjects]

        if (
            filter_params.subject
            and filter_params.medium
            and filter_params.board
            and filter_params.class_level
        ):
            chapters = session.exec(
                select(Chapter)
                .join(Subject)
                .join(Medium)
                .join(Board)
                .join(ClassLevel)
                .where(
                    ClassLevel.name == filter_params.class_level,
                    Board.name == filter_params.board,
                    Medium.name == filter_params.medium,
                    Subject.name == filter_params.subject,
                )
                .order_by(Chapter.name)
            ).all()
            result["chapter"] = [ch.name for ch in chapters]

        return result

    except Exception as e:
        logger.error(f"Error getting hierarchy options: {e}")
        return {
            level: []
            for level in ["class_level", "board", "medium", "subject", "chapter"]
        }


@router.get("/")
async def get_hierarchy_options(
    class_level: Optional[str] = Query(None),
    board: Optional[str] = Query(None),
    medium: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    filter_params = HierarchyFilter(
        class_level=class_level, board=board, medium=medium, subject=subject
    )
    options = fetch_hierarchy_options(filter_params, session)
    return {"data": options}
