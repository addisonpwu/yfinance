"""
Database Models - Import all models to register them with Base.metadata

This module MUST be imported before calling init_db() to ensure
all tables are created.
"""

from src.db.models.stock import Stock
from src.db.models.news import News
from src.db.models.ai_analysis import AIAnalysis

__all__ = ["Stock", "News", "AIAnalysis"]
