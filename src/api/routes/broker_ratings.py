"""
Broker Rating API Routes
"""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import get_session
from src.repositories.broker_rating_repo import BrokerRatingRepository
from src.api.schemas.broker_rating import (
    BrokerRatingImport,
    BrokerRatingListRequest,
    BrokerRatingLatestRequest,
    BrokerRatingConsensusRequest,
    BrokerRatingBatchRequest,
    BrokerRatingResponse,
    BrokerRatingConsensus,
)

router = APIRouter(prefix="/api/v1/broker-ratings", tags=["broker-ratings"])


def get_repo(session: AsyncSession = Depends(get_session)) -> BrokerRatingRepository:
    """Dependency to get broker rating repository"""
    return BrokerRatingRepository(session)


@router.post("/import", status_code=201)
async def import_ratings(
    data: List[BrokerRatingImport],
    repo: BrokerRatingRepository = Depends(get_repo),
):
    """
    批量導入券商評級數據（JSON body）

    - 自動創建不存在的股票
    - 根據 stock_id + broker + rating_date 去重更新
    """
    count = await repo.bulk_import([d.model_dump() for d in data])
    return {"message": f"Imported {count} ratings"}


@router.post("/list", response_model=List[BrokerRatingResponse])
async def list_ratings(
    req: BrokerRatingListRequest,
    repo: BrokerRatingRepository = Depends(get_repo),
):
    """獲取某股票的評級列表（按日期降序）"""
    return await repo.get_by_stock(req.stock_id, req.limit)


@router.post("/latest")
async def get_latest(
    req: BrokerRatingLatestRequest,
    repo: BrokerRatingRepository = Depends(get_repo),
):
    """獲取某股票的最新評級（每券商一筆）"""
    return await repo.get_latest_by_stock(req.stock_id)


@router.post("/consensus", response_model=BrokerRatingConsensus)
async def get_consensus(
    req: BrokerRatingConsensusRequest,
    repo: BrokerRatingRepository = Depends(get_repo),
):
    """獲取某股票的共識評級（最多券商的評級）"""
    return await repo.get_consensus(req.stock_id)


@router.post("/batch")
async def get_batch_ratings(
    req: BrokerRatingBatchRequest,
    repo: BrokerRatingRepository = Depends(get_repo),
):
    """批量獲取多隻股票的評級"""
    return await repo.get_by_stocks(req.stock_ids, req.limit)
