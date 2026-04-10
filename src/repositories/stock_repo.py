"""
Stock Repository Implementation
"""

from typing import Optional, List, Tuple
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.db.models.stock import Stock
from src.db.models.news import News
from src.repositories.base import BaseRepository
from src.utils.exceptions import StockNotFoundException, DuplicateRecordException
from src.api.schemas.stock import StockCreate, StockUpdate


class StockRepository(BaseRepository[Stock]):
    """
    Repository for Stock model operations

    Methods:
        - get_by_symbol(symbol): Get stock by symbol
        - list(market, skip, limit): List stocks with optional market filter
        - create(stock): Create new stock
        - update(symbol, stock): Update existing stock
        - delete_by_symbol(symbol): Delete stock by symbol
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, Stock)

    async def get_by_symbol(self, symbol: str) -> Optional[Stock]:
        """
        Get stock by symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL', '0700.HK')

        Returns:
            Stock instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(Stock).where(Stock.symbol == symbol)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting stock by symbol {symbol}: {e}")
            raise

    async def get_by_symbol_or_raise(self, symbol: str) -> Stock:
        """
        Get stock by symbol or raise exception

        Args:
            symbol: Stock symbol

        Returns:
            Stock instance

        Raises:
            StockNotFoundException: If stock not found
        """
        stock = await self.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundException(f"Stock with symbol '{symbol}' not found")
        return stock

    async def list(
        self,
        market: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
    ) -> List[Tuple[Stock, int, int]]:
        """
        List stocks with optional market filter, news counts, and sorting

        Args:
            market: Filter by market (US or HK), optional
            skip: Number of records to skip
            limit: Maximum number of records to return
            sort_by: Sort field (positive_news, negative_news, created_at), optional
            sort_order: Sort order (asc or desc), default is desc

        Returns:
            List of (Stock, positive_news_count, negative_news_count) tuples
        """
        try:
            positive_count = (
                func.count(News.id).filter(News.sentiment == 1).label("positive_count")
            )
            negative_count = (
                func.count(News.id).filter(News.sentiment == 0).label("negative_count")
            )
            query = (
                select(Stock, positive_count, negative_count)
                .outerjoin(News, Stock.id == News.stock_id)
                .group_by(Stock.id)
            )

            if market:
                query = query.where(Stock.market == market)

            # Apply sorting
            if sort_by == "positive_news":
                order_expr = positive_count
            elif sort_by == "negative_news":
                order_expr = negative_count
            elif sort_by == "created_at":
                order_expr = Stock.created_at
            else:
                # Default sorting by symbol
                order_expr = Stock.symbol

            if sort_order == "asc":
                query = query.order_by(order_expr.asc())
            else:
                query = query.order_by(order_expr.desc())

            query = query.offset(skip).limit(limit)

            result = await self.session.execute(query)
            return [(row[0], row[1], row[2]) for row in result.all()]
        except Exception as e:
            self.logger.error(f"Error listing stocks: {e}")
            raise

    async def create(self, stock_create: StockCreate) -> Stock:
        """
        Create a new stock

        Args:
            stock_create: StockCreate schema

        Returns:
            Created Stock instance

        Raises:
            DuplicateRecordException: If stock with symbol already exists
        """
        # Check for duplicate
        existing = await self.get_by_symbol(stock_create.symbol)
        if existing:
            raise DuplicateRecordException(
                f"Stock with symbol '{stock_create.symbol}' already exists"
            )

        stock = Stock(
            symbol=stock_create.symbol,
            name=stock_create.name,
            market=stock_create.market,
        )

        return await super().create(stock)

    async def update(self, symbol: str, stock_update: StockUpdate) -> Stock:
        """
        Update an existing stock

        Args:
            symbol: Stock symbol to update
            stock_update: StockUpdate schema with fields to update

        Returns:
            Updated Stock instance

        Raises:
            StockNotFoundException: If stock not found
        """
        stock = await self.get_by_symbol_or_raise(symbol)

        update_data = stock_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(stock, field, value)

        await self.session.flush()
        await self.session.refresh(stock)
        self.logger.info(f"Updated stock {symbol}")
        return stock

    async def delete_by_symbol(self, symbol: str) -> bool:
        """
        Delete stock by symbol

        Args:
            symbol: Stock symbol to delete

        Returns:
            True if deleted

        Raises:
            StockNotFoundException: If stock not found
        """
        stock = await self.get_by_symbol_or_raise(symbol)
        await self.session.delete(stock)
        await self.session.flush()
        self.logger.info(f"Deleted stock {symbol}")
        return True
