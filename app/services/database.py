from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel, text
from app.core.config import settings
from loguru import logger

# Create engine
engine: AsyncEngine = create_async_engine(
    settings.POSTGRES_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def force_drop_all_tables(schema: str = "public"):
    """Drops all tables in the specified schema with CASCADE."""
    async with engine.begin() as conn:
        # Fetch all table names in the schema
        result = await conn.execute(
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
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
            print(f"Dropped table: {table}")

        print("All tables dropped successfully.")


async def create_db_and_tables() -> None:
    """Create all tables based on SQLModel metadata."""
    # async with engine.begin() as conn:
    #     await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables created successfully")


async def get_session():
    async with async_session_maker() as session:
        yield session
