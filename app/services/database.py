from sqlmodel import SQLModel, create_engine, Session
from contextlib import contextmanager
from app.core.config import settings
from loguru import logger

# Database URL
DATABASE_URL = settings.POSTGRES_URL

# Create engine
engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_tables() -> None:
    """Create all tables based on SQLModel metadata."""

    # SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables created successfully")


def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
