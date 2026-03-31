"""
News API Routes
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import get_session
from src.repositories.news_repo import NewsRepository
from src.repositories.stock_repo import StockRepository
from src.api.schemas.news import (
    NewsCreate,
    NewsResponse,
    NewsListResponse,
)
from src.utils.exceptions import (
    NewsNotFoundException,
    DuplicateRecordException,
    StockNotFoundException,
)
from src.utils.logger import LoggerManager


logger = LoggerManager.get_logger("api.news")
router = APIRouter(prefix="/api/v1/news", tags=["news"])


async def get_news_repo(session: AsyncSession = Depends(get_session)) -> NewsRepository:
    return NewsRepository(session)


async def get_stock_repo(
    session: AsyncSession = Depends(get_session),
) -> StockRepository:
    return StockRepository(session)


@router.post(
    "/",
    response_model=NewsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new news entry",
    description="Create a new news entry linked to a stock",
)
async def create_news(
    news_create: NewsCreate,
    news_repo: NewsRepository = Depends(get_news_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    Create a new news entry

    - **stock_symbol**: Stock symbol (e.g., AAPL, 0700.HK)
    - **title**: News title
    - **publish_time**: Publication datetime
    - **url**: News URL (must be unique)
    """
    try:
        stock = await stock_repo.get_by_symbol_or_raise(news_create.stock_symbol)
    except StockNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock not found: {news_create.stock_symbol}",
        )

    try:
        news = await news_repo.create_news(news_create, stock_id=stock.id)
        logger.info(f"Created news: {news.id} for stock: {stock.symbol}")
        return NewsResponse(
            id=news.id,
            stock_id=news.stock_id,
            stock_symbol=stock.symbol,
            title=news.title,
            publish_time=news.publish_time,
            url=news.url,
            created_at=news.created_at,
        )
    except DuplicateRecordException as e:
        logger.warning(f"Duplicate news creation attempt: {news_create.url}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.get(
    "/",
    response_model=NewsListResponse,
    summary="List news",
    description="List all news with pagination and optional filters",
)
async def list_news(
    skip: int = 0,
    limit: int = 100,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    stock_symbol: Optional[str] = None,
    news_repo: NewsRepository = Depends(get_news_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    List news with optional filters

    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    - **start_time**: Filter news published after this time (optional)
    - **end_time**: Filter news published before this time (optional)
    - **stock_symbol**: Filter by stock symbol (optional)
    """
    stock_id = None
    if stock_symbol:
        try:
            stock = await stock_repo.get_by_symbol_or_raise(stock_symbol)
            stock_id = stock.id
        except StockNotFoundException:
            return NewsListResponse(items=[], total=0, skip=skip, limit=limit)

    news_list = await news_repo.list(
        skip=skip,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        stock_id=stock_id,
    )
    total = await news_repo.count()

    items = []
    for n in news_list:
        stock = await stock_repo.get_by_id(n.stock_id)
        items.append(
            NewsResponse(
                id=n.id,
                stock_id=n.stock_id,
                stock_symbol=stock.symbol if stock else "",
                title=n.title,
                publish_time=n.publish_time,
                url=n.url,
                created_at=n.created_at,
            )
        )

    return NewsListResponse(items=items, total=total, skip=skip, limit=limit)


@router.get(
    "/{news_id}",
    response_model=NewsResponse,
    summary="Get news by ID",
    description="Get a specific news by its ID",
)
async def get_news(
    news_id: int,
    news_repo: NewsRepository = Depends(get_news_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    Get news by ID

    - **news_id**: News ID
    """
    try:
        news = await news_repo.get_by_id_or_raise(news_id)
        stock = await stock_repo.get_by_id(news.stock_id)
        return NewsResponse(
            id=news.id,
            stock_id=news.stock_id,
            stock_symbol=stock.symbol if stock else "",
            title=news.title,
            publish_time=news.publish_time,
            url=news.url,
            created_at=news.created_at,
        )
    except NewsNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete(
    "/{news_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete news",
    description="Delete a news by ID",
)
async def delete_news(
    news_id: int,
    news_repo: NewsRepository = Depends(get_news_repo),
):
    """
    Delete news

    - **news_id**: News ID to delete
    """
    try:
        await news_repo.delete(news_id)
        logger.info(f"Deleted news: {news_id}")
    except NewsNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
