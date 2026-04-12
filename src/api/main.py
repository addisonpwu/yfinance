"""
FastAPI Application Entry Point
Stock Analysis Database API
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.db.database import init_db, close_db
from src.utils.logger import LoggerManager
from src.api.routes import stocks, news, ai_analyses, broker_ratings


logger = LoggerManager.get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler
    Initialize database on startup, close on shutdown
    """
    logger.info("Starting up Stock Analysis Database API...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    logger.info("Shutting down...")
    await close_db()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    """
    app = FastAPI(
        title="Stock Analysis Database API",
        description="API for managing stocks and news data",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(stocks.router)
    app.include_router(news.router)
    app.include_router(ai_analyses.router)
    app.include_router(broker_ratings.router)

    return app


app = create_app()


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint

    Returns:
        dict: Health status
    """
    return {"status": "ok", "service": "stock-analysis-db-api"}


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint

    Returns:
        dict: Welcome message and API info
    """
    return {
        "message": "Stock Analysis Database API",
        "docs": "/docs",
        "health": "/health",
    }
