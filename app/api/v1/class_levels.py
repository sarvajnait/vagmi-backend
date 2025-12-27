from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, text
from typing import Dict, List
from loguru import logger

from app.models import ClassLevel
from app.models import ClassLevelCreate, ClassLevelRead
from app.services.database import get_session

router = APIRouter()


@router.get("/", response_model=Dict[str, List[ClassLevelRead]])
async def get_class_levels(session: AsyncSession = Depends(get_session)):
    try:
        _result = await session.exec(
            select(ClassLevel).order_by(
                text(
                    "CASE WHEN name ~ '^[0-9]+$' THEN CAST(name AS INTEGER) ELSE 999 END, name"
                )
            ))
        class_levels = _result.all()
        return {"data": class_levels}
    except Exception as e:
        logger.error(f"Error getting class levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, ClassLevelRead])
async def create_class_level(
    class_level: ClassLevelCreate, session: AsyncSession = Depends(get_session)
):
    try:
        _result = await session.exec(
            select(ClassLevel).where(ClassLevel.name == class_level.name))
        existing = _result.first()
        if existing:
            raise HTTPException(status_code=400, detail="Class level already exists")

        db_class_level = ClassLevel.model_validate(class_level)
        session.add(db_class_level)
        await session.commit()
        await session.refresh(db_class_level)
        return {"data": db_class_level}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error creating class level: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{class_level_id}")
async def delete_class_level(
    class_level_id: int, session: AsyncSession = Depends(get_session)
):
    try:
        class_level = await session.get(ClassLevel, class_level_id)
        if not class_level:
            raise HTTPException(status_code=404, detail="Class level not found")

        await session.delete(class_level)
        await session.commit()
        return {"message": "Class level deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error deleting class level: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{class_level_id}", response_model=Dict[str, ClassLevelRead])
async def update_class_level(
    class_level_id: int,
    class_level_data: ClassLevelCreate,
    session: AsyncSession = Depends(get_session),
):
    try:
        db_class_level = await session.get(ClassLevel, class_level_id)
        if not db_class_level:
            raise HTTPException(status_code=404, detail="Class level not found")

        # Prevent duplicate names
        _result = await session.exec(
            select(ClassLevel).where(
                ClassLevel.name == class_level_data.name,
                ClassLevel.id != class_level_id,
            ))
        existing = _result.first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Another class level with this name already exists",
            )

        # Update name
        db_class_level.name = class_level_data.name

        session.add(db_class_level)
        await session.commit()
        await session.refresh(db_class_level)

        return {"data": db_class_level}

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error updating class level: {e}")
        raise HTTPException(status_code=400, detail=str(e))
