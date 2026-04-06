from datetime import datetime
from typing import Optional, List
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models.news import News
from src.repositories.base import BaseRepository
from src.utils.exceptions import NewsNotFoundException, DuplicateRecordException
from src.api.schemas.news import NewsCreate


class NewsRepository(BaseRepository[News]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, News)

    async def get_by_url(self, url: str) -> Optional[News]:
        result = await self.session.execute(select(News).where(News.url == url))
        return result.scalar_one_or_none()

    async def get_by_id_or_raise(self, news_id: int) -> News:
        news = await self.get_by_id(news_id)
        if news is None:
            raise NewsNotFoundException(f"News with id {news_id} not found")
        return news

    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        descending: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        stock_id: Optional[int] = None,
    ) -> List[News]:
        query = select(News).order_by(News.publish_time.desc())

        conditions = []
        if start_time:
            conditions.append(News.publish_time >= start_time)
        if end_time:
            conditions.append(News.publish_time <= end_time)
        if stock_id is not None:
            conditions.append(News.stock_id == stock_id)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.offset(skip).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create_news(self, news_create: NewsCreate, stock_id: int) -> News:
        existing = await self.get_by_url(news_create.url)
        if existing:
            raise DuplicateRecordException(
                f"News with URL '{news_create.url}' already exists"
            )

        news = News(
            stock_id=stock_id,
            title=news_create.title,
            content=news_create.content,
            sentiment=news_create.sentiment,
            publish_time=news_create.publish_time,
            url=news_create.url,
        )

        return await super().create(news)
