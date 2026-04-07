from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict
from sqlalchemy import String, Text, DateTime, Integer, Float, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.database import Base
from src.config.constants import (
    AI_ANALYSIS_SUMMARY_MAX_LENGTH,
    AI_ANALYSIS_PROVIDER_MAX_LENGTH,
    AI_ANALYSIS_MODEL_MAX_LENGTH,
    AI_ANALYSIS_INTERVAL_MAX_LENGTH,
)

if TYPE_CHECKING:
    from src.db.models.stock import Stock


class AIAnalysis(Base):
    __tablename__ = "ai_analyses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    stock_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    provider: Mapped[str] = mapped_column(
        String(AI_ANALYSIS_PROVIDER_MAX_LENGTH),
        nullable=False,
        index=True,
    )
    model_used: Mapped[str] = mapped_column(
        String(AI_ANALYSIS_MODEL_MAX_LENGTH),
        nullable=False,
    )

    interval: Mapped[str] = mapped_column(
        String(AI_ANALYSIS_INTERVAL_MAX_LENGTH),
        nullable=False,
        index=True,
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    recommendation: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )

    entry_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    detailed_analysis: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)

    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    stock: Mapped["Stock"] = relationship(
        "Stock",
        back_populates="ai_analyses",
        lazy="selectin",
    )

    __table_args__ = (
        Index("idx_ai_analyses_stock_analyzed", "stock_id", "analyzed_at"),
        Index("idx_ai_analyses_provider_interval", "provider", "interval"),
    )

    def __repr__(self) -> str:
        return f"<AIAnalysis(id={self.id}, stock_id={self.stock_id}, provider='{self.provider}')>"
