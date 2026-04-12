"""
Broker Rating Repository
"""

from datetime import date
from typing import Optional, List, Dict, Any
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models.broker_rating import BrokerRating
from src.repositories.base import BaseRepository


class BrokerRatingRepository(BaseRepository[BrokerRating]):
    """
    Repository for BrokerRating model operations

    Methods:
        - upsert_by_stock_broker(): Upsert rating by stock_id + broker + rating_date
        - get_by_stock(stock_id, limit): Get ratings for a stock
        - get_latest_by_stock(stock_id): Get latest rating per broker
        - get_consensus(stock_id): Get consensus rating
        - bulk_import(ratings_data): Batch import with auto-create stock
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, BrokerRating)

    async def upsert_by_stock_broker(
        self,
        stock_id: int,
        broker: str,
        rating: str,
        rating_date: date,
        last_rating: Optional[str] = None,
        reason: Optional[str] = None,
        rank: Optional[int] = None,
    ) -> BrokerRating:
        """
        Upsert rating by stock_id + broker + rating_date

        If a record with the same combination exists, update it.
        Otherwise, create a new record.
        """
        existing = await self.session.execute(
            select(BrokerRating).where(
                BrokerRating.stock_id == stock_id,
                BrokerRating.broker == broker,
                BrokerRating.rating_date == rating_date,
            )
        )
        existing_record = existing.scalar_one_or_none()

        if existing_record:
            # Update existing record
            existing_record.rating = rating
            existing_record.last_rating = last_rating
            existing_record.reason = reason
            existing_record.rank = rank
            await self.session.flush()
            await self.session.refresh(existing_record)
            return existing_record
        else:
            # Create new record
            record = BrokerRating(
                stock_id=stock_id,
                broker=broker,
                rating=rating,
                last_rating=last_rating,
                reason=reason,
                rank=rank,
                rating_date=rating_date,
            )
            self.session.add(record)
            await self.session.flush()
            await self.session.refresh(record)
            return record

    async def get_by_stock(
        self,
        stock_id: int,
        limit: int = 50,
    ) -> List[BrokerRating]:
        """Get ratings for a stock, ordered by date DESC"""
        result = await self.session.execute(
            select(BrokerRating)
            .where(BrokerRating.stock_id == stock_id)
            .order_by(BrokerRating.rating_date.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_latest_by_stock(
        self,
        stock_id: int,
    ) -> Dict[str, BrokerRating]:
        """Get latest rating per broker for a stock"""
        subquery = (
            select(
                BrokerRating.broker,
                func.max(BrokerRating.rating_date).label("max_date")
            )
            .where(BrokerRating.stock_id == stock_id)
            .group_by(BrokerRating.broker)
            .subquery()
        )
        result = await self.session.execute(
            select(BrokerRating)
            .join(
                subquery,
                (BrokerRating.broker == subquery.c.broker) &
                (BrokerRating.rating_date == subquery.c.max_date)
            )
            .where(BrokerRating.stock_id == stock_id)
        )
        records = result.scalars().all()
        return {r.broker: r for r in records}

    async def get_consensus(
        self,
        stock_id: int,
    ) -> Dict[str, Any]:
        """Get consensus rating (most common rating)"""
        result = await self.session.execute(
            select(
                BrokerRating.rating,
                func.count().label("count")
            )
            .where(BrokerRating.stock_id == stock_id)
            .group_by(BrokerRating.rating)
            .order_by(func.count().desc())
        )
        rows = result.all()
        if not rows:
            return {"rating": None, "count": 0, "total_brokers": 0}
        return {
            "rating": rows[0].rating,
            "count": rows[0].count,
            "total_brokers": len(rows),
            "distribution": {r.rating: r.count for r in rows},
        }

    async def bulk_import(
        self,
        ratings_data: List[Dict[str, Any]],
    ) -> int:
        """
        Import ratings, auto-create stock if needed

        Returns the number of imported ratings.
        """
        from src.db.models.stock import Stock

        count = 0
        for data in ratings_data:
            code = data["code"]
            # Normalize symbol to XXXX.HK format
            if code.upper().endswith(".HK"):
                symbol = code.upper()
            else:
                symbol = f"{code}.HK"
            name = data.get("name", symbol)

            # Get or create stock
            stock_result = await self.session.execute(
                select(Stock).where(Stock.symbol == symbol)
            )
            stock = stock_result.scalar_one_or_none()

            if not stock:
                stock = Stock(
                    symbol=symbol,
                    name=name,
                    market="HK",
                )
                self.session.add(stock)
                await self.session.flush()
                await self.session.refresh(stock)

            # Upsert rating
            await self.upsert_by_stock_broker(
                stock_id=stock.id,
                broker=data["broker"],
                rating=data["rating"],
                rating_date=data["date"],
                last_rating=data.get("lastRating"),
                reason=data.get("reason"),
                rank=data.get("rank"),
            )
            count += 1

        self.logger.info(f"Bulk imported {count} broker ratings")
        return count
