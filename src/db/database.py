"""
Database engine and session management for async SQLAlchemy
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase
from src.config.constants import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_POOL_SIZE,
    DB_MAX_OVERFLOW,
    DB_POOL_TIMEOUT,
    DB_ECHO,
    DB_EXPIRE_ON_COMMIT,
    DOCKER_DB_HOST,
)
from src.utils.exceptions import ConnectionException
from src.utils.logger import LoggerManager


logger = LoggerManager.get_logger("database")


class Base(DeclarativeBase):
    """Base class for all ORM models"""

    pass


def get_database_url() -> str:
    """
    Get database URL from environment or constants
    Format: postgresql+asyncpg://user:password@host:port/database
    """
    # Priority: Environment variables > Constants
    is_docker = os.getenv("DOCKER_ENV") or os.getenv("DOCKER_COMPOSE")
    db_host = os.getenv("DB_HOST", DOCKER_DB_HOST if is_docker else DB_HOST)
    db_port = os.getenv("DB_PORT", str(DB_PORT))
    db_name = os.getenv("DB_NAME", DB_NAME)
    db_user = os.getenv("DB_USER", DB_USER)
    db_password = os.getenv("DB_PASSWORD", DB_PASSWORD)

    return f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def create_engine() -> AsyncEngine:
    """
    Create async database engine with connection pool
    """
    database_url = get_database_url()

    engine = create_async_engine(
        database_url,
        echo=DB_ECHO,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_timeout=DB_POOL_TIMEOUT,
        pool_pre_ping=True,  # Verify connections before using
    )

    logger.info(f"Database engine created for: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create async session factory
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=DB_EXPIRE_ON_COMMIT,
        autoflush=False,
        autocommit=False,
    )


# Global engine and session factory (lazy initialization)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the database engine (singleton)
    """
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the session factory (singleton)
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory(get_engine())
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection for FastAPI endpoints
    Yields an async session and ensures proper cleanup

    Usage in FastAPI:
        @app.get("/stocks/")
        async def get_stocks(session: AsyncSession = Depends(get_session)):
            ...
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database (create tables if not exist)
    Call this on application startup
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_schema)
    logger.info("Database tables created successfully")


def _migrate_schema(sync_conn):
    """
    Run schema migrations for existing tables.
    Adds missing columns without dropping data.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(sync_conn)

    if "news" in inspector.get_table_names():
        columns = {col["name"] for col in inspector.get_columns("news")}

        if "content" not in columns:
            sync_conn.execute(text("ALTER TABLE news ADD COLUMN content TEXT"))
            logger.info("Migration: Added 'content' column to news table")

        if "sentiment" not in columns:
            sync_conn.execute(text("ALTER TABLE news ADD COLUMN sentiment INTEGER"))
            logger.info("Migration: Added 'sentiment' column to news table")


async def close_db():
    """
    Close database connections
    Call this on application shutdown
    """
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db_context():
    """
    Context manager for database operations outside of FastAPI

    Usage:
        async with get_db_context() as session:
            repo = StockRepository(session)
            stocks = await repo.list()
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database context error: {e}")
            raise
