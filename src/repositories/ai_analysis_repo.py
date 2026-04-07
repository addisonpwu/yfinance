"""
AI Analysis Repository Implementation
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import select, delete, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models.ai_analysis import AIAnalysis
from src.repositories.base import BaseRepository
from src.api.schemas.ai_analysis import AIAnalysisCreate
from src.config.constants import AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT


class AIAnalysisRepository(BaseRepository[AIAnalysis]):
    """
    Repository for AI Analysis model operations

    Methods:
        - get_latest(stock_id, provider, interval): Get latest analysis for stock
        - get_history(stock_id, provider, interval, skip, limit): Get analysis history
        - bulk_create(analyses_data, stock_id): Create multiple analyses
        - create(analysis_data, stock_id): Create single analysis
        - cleanup_old_records(stock_id, provider, interval, max_records): Cleanup old records
        - list(stock_id, provider, interval, recommendation, skip, limit): List analyses with filters
        - count(stock_id, provider, interval): Count analyses with filters
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, AIAnalysis)

    async def get_latest(
        self,
        stock_id: int,
        provider: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> Optional[AIAnalysis]:
        """
        Get latest analysis for a stock

        Args:
            stock_id: Stock ID
            provider: Optional provider filter
            interval: Optional interval filter

        Returns:
            Latest AIAnalysis or None
        """
        try:
            query = (
                select(AIAnalysis)
                .where(AIAnalysis.stock_id == stock_id)
                .order_by(AIAnalysis.analyzed_at.desc())
                .limit(1)
            )

            if provider:
                query = query.where(AIAnalysis.provider == provider)
            if interval:
                query = query.where(AIAnalysis.interval == interval)

            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(
                f"Error getting latest analysis for stock {stock_id}: {e}"
            )
            raise

    async def get_history(
        self,
        stock_id: int,
        provider: Optional[str] = None,
        interval: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AIAnalysis]:
        """
        Get analysis history for a stock

        Args:
            stock_id: Stock ID
            provider: Optional provider filter
            interval: Optional interval filter
            skip: Number of records to skip
            limit: Maximum number of records

        Returns:
            List of AIAnalysis records
        """
        try:
            query = (
                select(AIAnalysis)
                .where(AIAnalysis.stock_id == stock_id)
                .order_by(AIAnalysis.analyzed_at.desc())
                .offset(skip)
                .limit(limit)
            )

            if provider:
                query = query.where(AIAnalysis.provider == provider)
            if interval:
                query = query.where(AIAnalysis.interval == interval)

            result = await self.session.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(
                f"Error getting analysis history for stock {stock_id}: {e}"
            )
            raise

    async def bulk_create(
        self,
        analyses_data: List[AIAnalysisCreate],
        stock_id: int,
    ) -> List[AIAnalysis]:
        """
        Create multiple AI analysis records

        Args:
            analyses_data: List of AIAnalysisCreate schemas
            stock_id: Stock ID to associate with

        Returns:
            List of created AIAnalysis records
        """
        try:
            analyses = []
            for data in analyses_data:
                analysis = AIAnalysis(
                    stock_id=stock_id,
                    provider=data.provider,
                    model_used=data.model_used,
                    interval=data.interval,
                    summary=data.summary,
                    confidence=data.confidence,
                    recommendation=data.recommendation,
                    entry_price=data.entry_price,
                    exit_price=data.exit_price,
                    stop_loss=data.stop_loss,
                    detailed_analysis=data.detailed_analysis,
                    error=data.error,
                    analyzed_at=data.analyzed_at,
                )
                self.session.add(analysis)
                analyses.append(analysis)

            await self.session.flush()

            for analysis in analyses:
                await self.session.refresh(analysis)

            self.logger.info(
                f"Created {len(analyses)} AI analyses for stock {stock_id}"
            )
            return analyses
        except Exception as e:
            self.logger.error(f"Error bulk creating analyses for stock {stock_id}: {e}")
            raise

    async def create(
        self, analysis_data: AIAnalysisCreate, stock_id: int
    ) -> AIAnalysis:
        """
        Create single AI analysis record

        Args:
            analysis_data: AIAnalysisCreate schema
            stock_id: Stock ID to associate with

        Returns:
            Created AIAnalysis record
        """
        try:
            analysis = AIAnalysis(
                stock_id=stock_id,
                provider=analysis_data.provider,
                model_used=analysis_data.model_used,
                interval=analysis_data.interval,
                summary=analysis_data.summary,
                confidence=analysis_data.confidence,
                recommendation=analysis_data.recommendation,
                entry_price=analysis_data.entry_price,
                exit_price=analysis_data.exit_price,
                stop_loss=analysis_data.stop_loss,
                detailed_analysis=analysis_data.detailed_analysis,
                error=analysis_data.error,
                analyzed_at=analysis_data.analyzed_at,
            )
            self.session.add(analysis)
            await self.session.flush()
            await self.session.refresh(analysis)
            self.logger.info(f"Created AI analysis {analysis.id} for stock {stock_id}")
            return analysis
        except Exception as e:
            self.logger.error(f"Error creating analysis for stock {stock_id}: {e}")
            raise

    async def cleanup_old_records(
        self,
        stock_id: int,
        provider: Optional[str] = None,
        interval: Optional[str] = None,
        max_records: int = AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT,
    ) -> int:
        """
        Cleanup old records keeping only the latest N records

        Args:
            stock_id: Stock ID
            provider: Optional provider filter
            interval: Optional interval filter
            max_records: Maximum number of records to keep

        Returns:
            Number of deleted records
        """
        try:
            # Build filter conditions
            filters = [AIAnalysis.stock_id == stock_id]
            if provider:
                filters.append(AIAnalysis.provider == provider)
            if interval:
                filters.append(AIAnalysis.interval == interval)

            # Subquery to get IDs of records to keep (latest N by analyzed_at)
            keep_subquery = (
                select(AIAnalysis.id)
                .where(and_(*filters))
                .order_by(AIAnalysis.analyzed_at.desc())
                .limit(max_records)
                .scalar_subquery()
            )

            # Delete records not in the keep list
            delete_stmt = (
                delete(AIAnalysis)
                .where(and_(*filters))
                .where(AIAnalysis.id.notin_(keep_subquery))
            )

            result = await self.session.execute(delete_stmt)
            await self.session.flush()

            deleted_count = result.rowcount
            self.logger.info(
                f"Cleaned up {deleted_count} old analyses for stock {stock_id} "
                f"(provider={provider}, interval={interval}, keeping {max_records})"
            )
            return deleted_count
        except Exception as e:
            self.logger.error(
                f"Error cleaning up old analyses for stock {stock_id}: {e}"
            )
            raise

    async def list(
        self,
        stock_id: Optional[int] = None,
        provider: Optional[str] = None,
        interval: Optional[str] = None,
        recommendation: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AIAnalysis]:
        """
        List analyses with optional filters

        Args:
            stock_id: Optional stock ID filter
            provider: Optional provider filter
            interval: Optional interval filter
            recommendation: Optional recommendation filter
            skip: Number of records to skip
            limit: Maximum number of records

        Returns:
            List of AIAnalysis records
        """
        try:
            query = (
                select(AIAnalysis)
                .order_by(AIAnalysis.analyzed_at.desc())
                .offset(skip)
                .limit(limit)
            )

            if stock_id:
                query = query.where(AIAnalysis.stock_id == stock_id)
            if provider:
                query = query.where(AIAnalysis.provider == provider)
            if interval:
                query = query.where(AIAnalysis.interval == interval)
            if recommendation:
                query = query.where(AIAnalysis.recommendation == recommendation)

            result = await self.session.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error listing analyses: {e}")
            raise

    async def count(
        self,
        stock_id: Optional[int] = None,
        provider: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> int:
        """
        Count analyses with optional filters

        Args:
            stock_id: Optional stock ID filter
            provider: Optional provider filter
            interval: Optional interval filter

        Returns:
            Total count
        """
        try:
            query = select(func.count()).select_from(AIAnalysis)

            if stock_id:
                query = query.where(AIAnalysis.stock_id == stock_id)
            if provider:
                query = query.where(AIAnalysis.provider == provider)
            if interval:
                query = query.where(AIAnalysis.interval == interval)

            result = await self.session.execute(query)
            return result.scalar_one()
        except Exception as e:
            self.logger.error(f"Error counting analyses: {e}")
            raise
