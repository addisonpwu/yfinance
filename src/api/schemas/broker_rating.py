"""
Broker Rating API Schemas
"""

import re
from datetime import date
from typing import Optional, List
from pydantic import BaseModel, field_validator, ValidationError


# 港股代碼格式：4 位數字或 XXXX.HK 格式
HK_STOCK_CODE_PATTERN = re.compile(r'^\d{4}(\.HK)?$')


class BrokerRatingImport(BaseModel):
    """Schema for importing broker rating data"""
    rank: Optional[int] = None
    broker: str
    code: str
    date: date
    lastRating: Optional[str] = None
    name: str
    rating: str
    reason: Optional[str] = ""

    @field_validator('code')
    @classmethod
    def validate_hk_stock_code(cls, v: str) -> str:
        """Validate HK stock code format (4 digits or XXXX.HK)"""
        if not HK_STOCK_CODE_PATTERN.match(v):
            raise ValueError(
                f"Invalid HK stock code format: '{v}'. "
                f"Must be 4 digits (e.g., '01873') or XXXX.HK format (e.g., '01873.HK')"
            )
        return v


class BrokerRatingListRequest(BaseModel):
    """Schema for listing broker ratings"""
    stock_id: int
    limit: int = 50


class BrokerRatingLatestRequest(BaseModel):
    """Schema for getting latest broker ratings"""
    stock_id: int


class BrokerRatingConsensusRequest(BaseModel):
    """Schema for getting consensus rating"""
    stock_id: int


class BrokerRatingBatchRequest(BaseModel):
    """Schema for batch fetching broker ratings"""
    stock_ids: List[int]
    limit: int = 5


class BrokerRatingResponse(BaseModel):
    """Schema for broker rating response"""
    id: int
    stock_id: int
    broker: str
    rating: str
    last_rating: Optional[str]
    reason: Optional[str]
    rank: Optional[int]
    rating_date: date
    created_at: str

    model_config = {"from_attributes": True}


class BrokerRatingConsensus(BaseModel):
    """Schema for consensus rating"""
    rating: Optional[str]
    count: int
    total_brokers: int
    distribution: dict
