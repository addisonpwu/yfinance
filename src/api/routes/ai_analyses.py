"""
AI Analysis API Routes
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import get_session
from src.repositories.ai_analysis_repo import AIAnalysisRepository
from src.repositories.stock_repo import StockRepository
from src.api.schemas.ai_analysis import (
    AIAnalysisCreate,
    AIAnalysisResponse,
    AIAnalysisListResponse,
    AIAnalysisBulkCreate,
)
from src.api.schemas.stock import StockCreate
from src.config.constants import AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT
from src.utils.logger import LoggerManager


logger = LoggerManager.get_logger("api.ai_analyses")
router = APIRouter(prefix="/api/v1/ai-analyses", tags=["ai-analyses"])


async def get_ai_analysis_repo(
    session: AsyncSession = Depends(get_session),
) -> AIAnalysisRepository:
    """Dependency injection for AIAnalysisRepository"""
    return AIAnalysisRepository(session)


async def get_stock_repo(
    session: AsyncSession = Depends(get_session),
) -> StockRepository:
    """Dependency injection for StockRepository"""
    return StockRepository(session)


@router.post(
    "/{symbol}",
    response_model=AIAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create AI analysis",
    description="Create a new AI analysis for a stock (auto-creates stock if not exists)",
)
async def create_analysis(
    symbol: str,
    analysis_create: AIAnalysisCreate,
    analysis_repo: AIAnalysisRepository = Depends(get_ai_analysis_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    Create a new AI analysis for a stock

    - **symbol**: Stock symbol (e.g., AAPL, 0700.HK)
    - **analysis_create**: AI analysis data including provider, model, summary, confidence, etc.

    Auto-creates the stock if it doesn't exist.
    """
    try:
        # Get or create stock
        stock = await stock_repo.get_by_symbol(symbol)
        if not stock:
            stock = await stock_repo.create(
                StockCreate(
                    symbol=symbol,
                    name=symbol,
                    market="US" if not symbol.endswith(".HK") else "HK",
                )
            )
            logger.info(f"Auto-created stock: {symbol}")

        # Create analysis
        analysis = await analysis_repo.create(analysis_create, stock.id)
        logger.info(
            f"Created AI analysis for {symbol} using {analysis_create.provider}"
        )

        # Cleanup old records
        await analysis_repo.cleanup_old_records(
            stock_id=stock.id,
            provider=analysis_create.provider,
            interval=analysis_create.interval,
            max_records=AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT,
        )

        return analysis
    except Exception as e:
        logger.error(f"Error creating AI analysis for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create AI analysis: {str(e)}",
        )


@router.post(
    "/{symbol}/bulk",
    response_model=list[AIAnalysisResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Bulk create AI analyses",
    description="Create multiple AI analyses for a stock (useful for multi-provider analysis)",
)
async def bulk_create_analyses(
    symbol: str,
    bulk_create: AIAnalysisBulkCreate,
    analysis_repo: AIAnalysisRepository = Depends(get_ai_analysis_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    Bulk create AI analyses for a stock

    - **symbol**: Stock symbol (e.g., AAPL, 0700.HK)
    - **bulk_create**: List of AI analysis data from multiple providers

    Auto-creates the stock if it doesn't exist.
    """
    try:
        # Get or create stock
        stock = await stock_repo.get_by_symbol(symbol)
        if not stock:
            stock = await stock_repo.create(
                StockCreate(
                    symbol=symbol,
                    name=symbol,
                    market="US" if not symbol.endswith(".HK") else "HK",
                )
            )
            logger.info(f"Auto-created stock: {symbol}")

        # Create analyses
        analyses = await analysis_repo.bulk_create(bulk_create.analyses, stock.id)
        logger.info(f"Created {len(analyses)} AI analyses for {symbol}")

        # Cleanup old records for each provider/interval combination
        provider_intervals = set((a.provider, a.interval) for a in bulk_create.analyses)
        for provider, interval in provider_intervals:
            await analysis_repo.cleanup_old_records(
                stock_id=stock.id,
                provider=provider,
                interval=interval,
                max_records=AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT,
            )

        return analyses
    except Exception as e:
        logger.error(f"Error bulk creating AI analyses for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to bulk create AI analyses: {str(e)}",
        )


@router.get(
    "/",
    response_model=AIAnalysisListResponse,
    summary="List all AI analyses",
    description="List all AI analyses with optional filters and pagination",
)
async def list_analyses(
    symbol: Optional[str] = None,
    provider: Optional[str] = None,
    interval: Optional[str] = None,
    recommendation: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    analysis_repo: AIAnalysisRepository = Depends(get_ai_analysis_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    List all AI analyses with optional filters

    - **symbol**: Filter by stock symbol, optional
    - **provider**: Filter by AI provider (e.g., iflow, nvidia, gemini), optional
    - **interval**: Filter by interval (e.g., 1d, 1h, 1m), optional
    - **recommendation**: Filter by recommendation (e.g., BUY, SELL), optional
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    """
    try:
        stock_id = None
        if symbol:
            stock = await stock_repo.get_by_symbol(symbol)
            if stock:
                stock_id = stock.id

        analyses = await analysis_repo.list(
            stock_id=stock_id,
            provider=provider,
            interval=interval,
            recommendation=recommendation,
            skip=skip,
            limit=limit,
        )
        total = await analysis_repo.count(
            stock_id=stock_id,
            provider=provider,
            interval=interval,
        )

        return AIAnalysisListResponse(
            items=[AIAnalysisResponse.model_validate(a) for a in analyses],
            total=total,
            skip=skip,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Error listing AI analyses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list AI analyses: {str(e)}",
        )


@router.get(
    "/{symbol}",
    response_model=AIAnalysisListResponse,
    summary="Get analysis history for a stock",
    description="Get all AI analysis history for a specific stock",
)
async def get_analysis_history(
    symbol: str,
    provider: Optional[str] = None,
    interval: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    analysis_repo: AIAnalysisRepository = Depends(get_ai_analysis_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    Get AI analysis history for a stock

    - **symbol**: Stock symbol (e.g., AAPL, 0700.HK)
    - **provider**: Filter by AI provider, optional
    - **interval**: Filter by interval, optional
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    """
    try:
        stock = await stock_repo.get_by_symbol(symbol)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock not found: {symbol}",
            )

        analyses = await analysis_repo.get_history(
            stock_id=stock.id,
            provider=provider,
            interval=interval,
            skip=skip,
            limit=limit,
        )
        total = await analysis_repo.count(
            stock_id=stock.id,
            provider=provider,
            interval=interval,
        )

        return AIAnalysisListResponse(
            items=[AIAnalysisResponse.model_validate(a) for a in analyses],
            total=total,
            skip=skip,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis history for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analysis history: {str(e)}",
        )


@router.get(
    "/{symbol}/latest",
    response_model=AIAnalysisResponse,
    summary="Get latest analysis",
    description="Get the latest AI analysis for a specific stock",
)
async def get_latest_analysis(
    symbol: str,
    provider: Optional[str] = None,
    interval: Optional[str] = None,
    analysis_repo: AIAnalysisRepository = Depends(get_ai_analysis_repo),
    stock_repo: StockRepository = Depends(get_stock_repo),
):
    """
    Get the latest AI analysis for a stock

    - **symbol**: Stock symbol (e.g., AAPL, 0700.HK)
    - **provider**: Filter by AI provider, optional
    - **interval**: Filter by interval, optional
    """
    try:
        stock = await stock_repo.get_by_symbol(symbol)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock not found: {symbol}",
            )

        analysis = await analysis_repo.get_latest(
            stock_id=stock.id,
            provider=provider,
            interval=interval,
        )

        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No AI analysis found for {symbol}",
            )

        return AIAnalysisResponse.model_validate(analysis)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest analysis for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get latest analysis: {str(e)}",
        )
