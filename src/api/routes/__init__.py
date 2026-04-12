"""API Routes"""

from src.api.routes.stocks import router as stocks_router
from src.api.routes.news import router as news_router
from src.api.routes.ai_analyses import router as ai_analyses_router
from src.api.routes.broker_ratings import router as broker_ratings_router

__all__ = ["stocks_router", "news_router", "ai_analyses_router", "broker_ratings_router"]
