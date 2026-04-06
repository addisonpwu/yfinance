"""
Stock Pydantic Schemas for API Request/Response validation
"""

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.config.constants import (
    STOCK_SYMBOL_MAX_LENGTH,
    STOCK_NAME_MAX_LENGTH,
    STOCK_MARKET_MAX_LENGTH,
)


class StockBase(BaseModel):
    """Base schema for Stock"""

    symbol: str = Field(..., max_length=STOCK_SYMBOL_MAX_LENGTH, description="股票代码")
    name: str = Field(..., max_length=STOCK_NAME_MAX_LENGTH, description="股票名称")
    market: str = Field(
        ..., max_length=STOCK_MARKET_MAX_LENGTH, description="市场 (US/HK)"
    )

    model_config = ConfigDict(from_attributes=True)


class StockCreate(StockBase):
    """Schema for creating a new stock"""

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str, info) -> str:
        pattern = r"^\d{4}\.HK$"
        if not re.match(pattern, v):
            raise ValueError(
                "股票代码格式错误: 必须为4位数字+.HK (例如: 0700.HK, 1234.HK)"
            )
        return v


class StockUpdate(BaseModel):
    """Schema for updating a stock"""

    name: Optional[str] = Field(
        None, max_length=STOCK_NAME_MAX_LENGTH, description="股票名称"
    )
    market: Optional[str] = Field(
        None, max_length=STOCK_MARKET_MAX_LENGTH, description="市场"
    )


class StockResponse(StockBase):
    """Schema for stock response"""

    id: int
    created_at: datetime
    updated_at: datetime
    positive_news_count: int = 0
    negative_news_count: int = 0


class StockListResponse(BaseModel):
    """Schema for stock list response"""

    items: list[StockResponse]
    total: int
    skip: int
    limit: int
