"""
Stock ORM Model
"""

from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.database import Base
from src.config.constants import (
    STOCK_SYMBOL_MAX_LENGTH,
    STOCK_NAME_MAX_LENGTH,
    STOCK_MARKET_MAX_LENGTH,
)

if TYPE_CHECKING:
    from src.db.models.news import News


class Stock(Base):
    """
    Stock ORM Model

    Table: stocks
    Fields: id, symbol, name, market, created_at, updated_at
    Relationships: One-to-Many with News (a stock hasmany news)
    """

    __tablename__ = "stocks"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Fields
    symbol: Mapped[str] = mapped_column(
        String(STOCK_SYMBOL_MAX_LENGTH),
        unique=True,
        nullable=False,
        index=True,
        comment="股票代码 (e.g., AAPL, 0700.HK)",
    )
    name: Mapped[str] = mapped_column(
        String(STOCK_NAME_MAX_LENGTH), nullable=False, comment="股票名称"
    )
    market: Mapped[str] = mapped_column(
        String(STOCK_MARKET_MAX_LENGTH),
        nullable=False,
        index=True,
        comment="市场 (US/HK)",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间",
    )

    # Relationships
    # Stock -> News (one-to-many)
    news: Mapped[list["News"]] = relationship(
        "News",
        back_populates="stock",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Stock(id={self.id}, symbol='{self.symbol}', name='{self.name}', market='{self.market}')>"

    def __str__(self) -> str:
        return f"{self.symbol} ({self.name}) - {self.market}"
