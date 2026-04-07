from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from src.config.constants import AI_ANALYSIS_SUMMARY_MAX_LENGTH, Recommendation


class AIAnalysisBase(BaseModel):
    provider: str
    model_used: str
    interval: str
    summary: str = Field(..., max_length=AI_ANALYSIS_SUMMARY_MAX_LENGTH)
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendation: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    detailed_analysis: Optional[Dict] = None
    error: Optional[str] = None

    @field_validator("recommendation")
    @classmethod
    def validate_recommendation(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        valid_values = [r.value for r in Recommendation]
        if v not in valid_values:
            raise ValueError(f"Invalid recommendation. Must be one of: {valid_values}")
        return v


class AIAnalysisCreate(AIAnalysisBase):
    analyzed_at: datetime


class AIAnalysisUpdate(BaseModel):
    summary: Optional[str] = Field(None, max_length=AI_ANALYSIS_SUMMARY_MAX_LENGTH)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    recommendation: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    detailed_analysis: Optional[Dict] = None
    error: Optional[str] = None

    @field_validator("recommendation")
    @classmethod
    def validate_recommendation(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        valid_values = [r.value for r in Recommendation]
        if v not in valid_values:
            raise ValueError(f"Invalid recommendation. Must be one of: {valid_values}")
        return v


class AIAnalysisResponse(AIAnalysisBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stock_id: int
    analyzed_at: datetime
    created_at: datetime


class AIAnalysisListResponse(BaseModel):
    items: List[AIAnalysisResponse]
    total: int
    skip: int
    limit: int


class AIAnalysisTriggerRequest(BaseModel):
    interval: str = Field(default="1d")
    force: bool = Field(default=False)


class AIAnalysisBulkCreate(BaseModel):
    analyses: List[AIAnalysisCreate]
