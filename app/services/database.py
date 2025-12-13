from sqlmodel import SQLModel, create_engine, Session, text
from contextlib import contextmanager
from app.core.config import settings
from loguru import logger

# Create engine
engine = create_engine(
    settings.POSTGRES_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)


def force_drop_all_tables(schema: str = "public"):
    """Drops all tables in the specified schema with CASCADE."""
    with engine.begin() as conn:
        # Fetch all table names in the schema
        result = conn.execute(
            text(
                f"""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = :schema
            """
            ),
            {"schema": schema},
        )
        tables = [row[0] for row in result]

        if not tables:
            print("No tables found to drop.")
            return

        # Drop each table using CASCADE, quoting reserved keywords
        for table in tables:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
            print(f"Dropped table: {table}")

        print("All tables dropped successfully.")


def create_db_and_tables() -> None:
    """Create all tables based on SQLModel metadata."""

    # SQLModel.metadata.drop_all(engine)
    # force_drop_all_tables()
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables created successfully")


def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
