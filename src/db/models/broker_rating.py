from datetime import date, datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, Text, Integer, Date, DateTime, ForeignKey, func, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.database import Base

if TYPE_CHECKING:
    from src.db.models.stock import Stock


class BrokerRating(Base):
    __tablename__ = "broker_ratings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False, index=True)
    broker: Mapped[str] = mapped_column(String(100), nullable=False)
    rating: Mapped[str] = mapped_column(String(50), nullable=False)
    last_rating: Mapped[str | None] = mapped_column(String(50), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="broker_ratings", lazy="selectin")

    __table_args__ = (
        UniqueConstraint("stock_id", "broker", "rating_date", name="uq_stock_broker_date"),
        Index("idx_broker_ratings_stock", "stock_id"),
        Index("idx_broker_ratings_broker", "broker"),
        Index("idx_broker_ratings_date", "rating_date", postgresql_ops={"rating_date": "DESC"}),
    )

    def __repr__(self) -> str:
        return f"<BrokerRating(id={self.id}, stock_id={self.stock_id}, broker='{self.broker}', rating='{self.rating}')>"
