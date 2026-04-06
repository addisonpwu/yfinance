"""
News ORM Model
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from sqlalchemy import String, Text, DateTime, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.database import Base
from src.config.constants import NEWS_URL_MAX_LENGTH

if TYPE_CHECKING:
    from src.db.models.stock import Stock


class News(Base):
    """
    News ORM Model

    Table: news
    Fields: id, stock_id, title, content, sentiment, publish_time, url, created_at
    Relationships: Many-to-One with Stock (a news belongs to onestock)
    """

    __tablename__ = "news"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Foreign key to Stock
    stock_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="关联股票ID",
    )

    # Fields
    title: Mapped[str] = mapped_column(Text, nullable=False, comment="新闻标题")
    content: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="新闻内容/摘要"
    )
    sentiment: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="情感倾向: 1=正面, 0=负面"
    )
    publish_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True, comment="发布时间"
    )
    url: Mapped[str] = mapped_column(
        String(NEWS_URL_MAX_LENGTH),
        unique=True,
        nullable=False,
        index=True,
        comment="新闻链接",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
    )

    # Relationships
    # News -> Stock (many-to-one)
    stock: Mapped["Stock"] = relationship(
        "Stock",
        back_populates="news",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<News(id={self.id}, stock_id={self.stock_id}, title='{self.title[:30]}...')>"

    def __str__(self) -> str:
        return f"[{self.publish_time}] {self.title}"
