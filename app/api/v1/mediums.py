from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from typing import Dict, List, Optional
from loguru import logger

from app.models import Medium, Board
from app.models import MediumCreate, MediumRead
from app.services.database import get_session

router = APIRouter()


@router.get("/", response_model=Dict[str, List[MediumRead]])
async def get_mediums(
    board_id: Optional[int] = Query(None), session: AsyncSession = Depends(get_session)
):
    try:
        query = select(Medium, Board).join(Board)
        if board_id:
            query = query.where(Medium.board_id == board_id)
        _result = await session.exec(query.order_by(Board.name, Medium.name))
        results = _result.all()

        mediums = [
            MediumRead(
                id=medium.id,
                name=medium.name,
                board_id=medium.board_id,
                board_name=board.name,
            )
            for medium, board in results
        ]
        return {"data": mediums}
    except Exception as e:
        logger.error(f"Error getting mediums: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, MediumRead])
async def create_medium(medium: MediumCreate, session: AsyncSession = Depends(get_session)):
    try:
        board = await session.get(Board, medium.board_id)
        if not board:
            raise HTTPException(status_code=400, detail="Board not found")

        _result = await session.exec(
            select(Medium).where(
                Medium.name == medium.name, Medium.board_id == medium.board_id
            )
        )
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Medium already exists for this board"
            )

        db_medium = Medium.model_validate(medium)
        session.add(db_medium)
        await session.commit()
        await session.refresh(db_medium)
        return {
            "data": MediumRead(
                id=db_medium.id,
                name=db_medium.name,
                board_id=db_medium.board_id,
                board_name=board.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error creating medium: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{medium_id}")
async def delete_medium(medium_id: int, session: AsyncSession = Depends(get_session)):
    try:
        medium = await session.get(Medium, medium_id)
        if not medium:
            raise HTTPException(status_code=404, detail="Medium not found")
        await session.delete(medium)
        await session.commit()
        return {"message": "Medium deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error deleting medium: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{medium_id}", response_model=Dict[str, MediumRead])
async def update_medium(
    medium_id: int,
    medium_data: MediumCreate,
    session: AsyncSession = Depends(get_session),
):
    try:
        db_medium = await session.get(Medium, medium_id)
        if not db_medium:
            raise HTTPException(status_code=404, detail="Medium not found")

        board = await session.get(Board, medium_data.board_id)
        if not board:
            raise HTTPException(status_code=400, detail="Board not found")

        # Prevent duplicate medium names under the same board (excluding itself)
        _result = await session.exec(
            select(Medium).where(
                Medium.name == medium_data.name,
                Medium.board_id == medium_data.board_id,
                Medium.id != medium_id,
            ))
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Another medium with this name already exists for this board",
            )

        # Update fields
        db_medium.name = medium_data.name
        db_medium.board_id = medium_data.board_id

        session.add(db_medium)
        await session.commit()
        await session.refresh(db_medium)

        return {
            "data": MediumRead(
                id=db_medium.id,
                name=db_medium.name,
                board_id=db_medium.board_id,
                board_name=board.name,
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error updating medium: {e}")
        raise HTTPException(status_code=400, detail=str(e))
