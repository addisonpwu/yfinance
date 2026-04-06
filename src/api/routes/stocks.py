"""
Stock API Routes
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import get_session
from src.repositories.stock_repo import StockRepository
from src.api.schemas.stock import (
    StockCreate,
    StockUpdate,
    StockResponse,
    StockListResponse,
)
from src.utils.exceptions import StockNotFoundException, DuplicateRecordException
from src.utils.logger import LoggerManager


logger = LoggerManager.get_logger("api.stocks")
router = APIRouter(prefix="/api/v1/stocks", tags=["stocks"])


async def get_stock_repo(
    session: AsyncSession = Depends(get_session),
) -> StockRepository:
    """Dependency injection for StockRepository"""
    return StockRepository(session)


@router.post(
    "/",
    response_model=StockResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new stock",
    description="Create a new stock entry in the database",
)
async def create_stock(
    stock_create: StockCreate,
    repo: StockRepository = Depends(get_stock_repo),
):
    """
    Create a new stock

    - **symbol**: Stock symbol (e.g., AAPL, 0700.HK)
    - **name**: Stock name
    - **market**: Market code (US or HK)
    """
    try:
        stock = await repo.create(stock_create)
        logger.info(f"Created stock: {stock.symbol}")
        return stock
    except DuplicateRecordException as e:
        logger.warning(f"Duplicate stock creation attempt: {stock_create.symbol}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.get(
    "/",
    response_model=StockListResponse,
    summary="List stocks",
    description="List all stocks with pagination and optional market filter",
)
async def list_stocks(
    market: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    repo: StockRepository = Depends(get_stock_repo),
):
    """
    List stocks with optional filters

    - **market**: Filter by market (US or HK), optional
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    """
    stocks_with_counts = await repo.list(market=market, skip=skip, limit=limit)
    total = await repo.count()

    return StockListResponse(
        items=[
            StockResponse.model_validate(
                {
                    **s.__dict__,
                    "positive_news_count": pos_count,
                    "negative_news_count": neg_count,
                }
            )
            for s, pos_count, neg_count in stocks_with_counts
        ],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{symbol}",
    response_model=StockResponse,
    summary="Get stock by symbol",
    description="Get a specific stock by its symbol",
)
async def get_stock(
    symbol: str,
    repo: StockRepository = Depends(get_stock_repo),
):
    """
    Get stock by symbol

    - **symbol**: Stock symbol (e.g., AAPL, 0700.HK)
    """
    try:
        stock = await repo.get_by_symbol_or_raise(symbol)
        return stock
    except StockNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.put(
    "/{symbol}",
    response_model=StockResponse,
    summary="Update stock",
    description="Update an existing stock",
)
async def update_stock(
    symbol: str,
    stock_update: StockUpdate,
    repo: StockRepository = Depends(get_stock_repo),
):
    """
    Update stock

    - **symbol**: Stock symbol to update
    - **name**: New stock name (optional)
    - **market**: New market code (optional)
    """
    try:
        stock = await repo.update(symbol, stock_update)
        logger.info(f"Updated stock: {symbol}")
        return stock
    except StockNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete(
    "/{symbol}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete stock",
    description="Delete a stock by symbol",
)
async def delete_stock(
    symbol: str,
    repo: StockRepository = Depends(get_stock_repo),
):
    """
    Delete stock

    - **symbol**: Stock symbol to delete
    """
    try:
        await repo.delete_by_symbol(symbol)
        logger.info(f"Deleted stock: {symbol}")
    except StockNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
