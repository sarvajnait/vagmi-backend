from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from typing import Dict, List, Optional
from loguru import logger

from app.models import Board, ClassLevel
from app.models import BoardCreate, BoardRead
from app.services.database import get_session

router = APIRouter()


@router.get("/", response_model=Dict[str, List[BoardRead]])
async def get_boards(
    class_level_id: Optional[int] = Query(None), session: AsyncSession = Depends(get_session)
):
    try:
        query = select(Board, ClassLevel).join(ClassLevel)
        if class_level_id:
            query = query.where(Board.class_level_id == class_level_id)
        _result = await session.exec(query.order_by(ClassLevel.name, Board.name))
        results = _result.all()

        boards = [
            BoardRead(
                id=board.id,
                name=board.name,
                class_level_id=board.class_level_id,
                class_level_name=class_level.name,
            )
            for board, class_level in results
        ]
        return {"data": boards}
    except Exception as e:
        logger.error(f"Error getting boards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, BoardRead])
async def create_board(board: BoardCreate, session: AsyncSession = Depends(get_session)):
    try:
        class_level = await session.get(ClassLevel, board.class_level_id)
        if not class_level:
            raise HTTPException(status_code=400, detail="Class level not found")

        _result = await session.exec(
            select(Board).where(
                Board.name == board.name, Board.class_level_id == board.class_level_id
            )
        )
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Board already exists for this class level"
            )

        db_board = Board.model_validate(board)
        session.add(db_board)
        await session.commit()
        await session.refresh(db_board)
        return {
            "data": BoardRead(
                id=db_board.id,
                name=db_board.name,
                class_level_id=db_board.class_level_id,
                class_level_name=class_level.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error creating board: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{board_id}")
async def delete_board(board_id: int, session: AsyncSession = Depends(get_session)):
    try:
        board = await session.get(Board, board_id)
        if not board:
            raise HTTPException(status_code=404, detail="Board not found")
        await session.delete(board)
        await session.commit()
        return {"message": "Board deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error deleting board: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{board_id}", response_model=Dict[str, BoardRead])
async def update_board(
    board_id: int,
    board_data: BoardCreate,
    session: AsyncSession = Depends(get_session),
):
    try:
        db_board = await session.get(Board, board_id)
        if not db_board:
            raise HTTPException(status_code=404, detail="Board not found")

        class_level = await session.get(ClassLevel, board_data.class_level_id)
        if not class_level:
            raise HTTPException(status_code=400, detail="Class level not found")

        # Prevent duplicate board name under the same class level
        _result = await session.exec(
            select(Board).where(
                Board.name == board_data.name,
                Board.class_level_id == board_data.class_level_id,
                Board.id != board_id,
            ))
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Another board with this name already exists for this class level",
            )

        # Update fields
        db_board.name = board_data.name
        db_board.class_level_id = board_data.class_level_id

        session.add(db_board)
        await session.commit()
        await session.refresh(db_board)

        return {
            "data": BoardRead(
                id=db_board.id,
                name=db_board.name,
                class_level_id=db_board.class_level_id,
                class_level_name=class_level.name,
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error updating board: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
