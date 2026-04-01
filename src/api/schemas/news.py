"""
News Pydantic Schemas for API Request/Response validation
"""

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.config.constants import NEWS_URL_MAX_LENGTH


class NewsBase(BaseModel):
    """Base schema for News"""

    stock_symbol: str = Field(..., max_length=20, description="股票代码")
    title: str = Field(..., description="新闻标题")
    publish_time: datetime = Field(..., description="发布时间")
    url: str = Field(..., max_length=NEWS_URL_MAX_LENGTH, description="新闻链接")

    model_config = ConfigDict(from_attributes=True)


class NewsCreate(NewsBase):
    """Schema for creating a news entry"""

    @field_validator("stock_symbol")
    @classmethod
    def validate_stock_symbol(cls, v: str, info) -> str:
        pattern = r"^\d{4}\.HK$"
        if not re.match(pattern, v):
            raise ValueError(
                "股票代码格式错误: 必须为4位数字+.HK (例如: 0700.HK, 1234.HK)"
            )
        return v


class NewsUpdate(BaseModel):
    """Schema for updating a news entry"""

    stock_symbol: Optional[str] = Field(None, max_length=20, description="股票代码")
    title: Optional[str] = Field(None, description="新闻标题")
    publish_time: Optional[datetime] = Field(None, description="发布时间")


class NewsResponse(BaseModel):
    """Schema for news response"""

    id: int
    stock_id: int
    stock_symbol: str
    title: str
    publish_time: datetime
    url: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class NewsListResponse(BaseModel):
    """Schema for news list response"""

    items: list[NewsResponse]
    total: int
    skip: int
    limit: int
