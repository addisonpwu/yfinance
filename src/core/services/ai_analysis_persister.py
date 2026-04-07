"""
AI Analysis Persistence Service

Handles saving AI analysis results to the database after analysis completes.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from src.db.database import get_engine, get_session_factory, Base
from src.db.models.ai_analysis import AIAnalysis
from src.db.models.stock import Stock
from src.repositories.ai_analysis_repo import AIAnalysisRepository
from src.repositories.stock_repo import StockRepository
from src.api.schemas.stock import StockCreate
from src.api.schemas.ai_analysis import AIAnalysisCreate
from src.config.constants import (
    AI_ANALYSIS_AUTO_SAVE_DEFAULT,
    AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT,
)
from src.utils.logger import LoggerManager

logger = LoggerManager.get_logger("services.ai_analysis_persister")


class AIAnalysisPersister:
    """
    Persists AI analysis results to the database.

    Manages its own async event loop so it can be called from sync code
    (e.g. StockAnalyzer.analyze() which runs in a ThreadPoolExecutor thread).
    """

    def __init__(
        self,
        auto_save: bool = AI_ANALYSIS_AUTO_SAVE_DEFAULT,
        max_records: int = AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT,
    ):
        self.auto_save = auto_save
        self.max_records = max_records

    def persist_multi_provider_results(
        self,
        symbol: str,
        interval: str,
        ai_results: List[Dict],
    ) -> bool:
        """
        Persist multi-provider AI analysis results.

        Args:
            symbol: Stock symbol
            interval: Data interval (1d/1h/1m)
            ai_results: List of dicts with keys:
                provider, model_used, summary, confidence, detailed_analysis

        Returns:
            True if persisted successfully, False otherwise
        """
        if not self.auto_save or not ai_results:
            return False

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._persist_multi_provider_results_async(
                        symbol, interval, ai_results
                    )
                )
            finally:
                loop.close()
            return True
        except Exception as e:
            logger.error(f"Error persisting AI analyses for {symbol}: {e}")
            return False

    async def _persist_multi_provider_results_async(
        self,
        symbol: str,
        interval: str,
        ai_results: List[Dict],
    ):
        engine = get_engine()
        session_factory = get_session_factory()

        async with session_factory() as session:
            try:
                stock_repo = StockRepository(session)
                ai_repo = AIAnalysisRepository(session)

                stock = await stock_repo.get_by_symbol(symbol)
                if not stock:
                    stock = await stock_repo.create(
                        StockCreate(
                            symbol=symbol,
                            name=symbol,
                            market="US" if not symbol.endswith(".HK") else "HK",
                        )
                    )
                    logger.info(f"Auto-created stock: {symbol}")

                analyses_data = []
                for result in ai_results:
                    analyses_data.append(
                        AIAnalysisCreate(
                            provider=result["provider"],
                            model_used=result["model_used"],
                            interval=interval,
                            summary=result["summary"],
                            confidence=result["confidence"],
                            recommendation=self._extract_recommendation(
                                result.get("detailed_analysis")
                            ),
                            entry_price=self._extract_price(
                                result.get("detailed_analysis"), "entry"
                            ),
                            exit_price=self._extract_price(
                                result.get("detailed_analysis"), "exit"
                            ),
                            stop_loss=self._extract_price(
                                result.get("detailed_analysis"), "stop_loss"
                            ),
                            detailed_analysis=result.get("detailed_analysis"),
                            analyzed_at=datetime.now(),
                        )
                    )

                await ai_repo.bulk_create(analyses_data, stock.id)
                await session.commit()

                provider_intervals = set((r["provider"], interval) for r in ai_results)
                for provider, intv in provider_intervals:
                    await ai_repo.cleanup_old_records(
                        stock_id=stock.id,
                        provider=provider,
                        interval=intv,
                        max_records=self.max_records,
                    )

                logger.info(f"Persisted {len(analyses_data)} AI analyses for {symbol}")
            except Exception:
                await session.rollback()
                raise

    def _extract_recommendation(self, detailed: Optional[Dict]) -> Optional[str]:
        if not detailed:
            return None
        return detailed.get("recommendation")

    def _extract_price(
        self, detailed: Optional[Dict], price_type: str
    ) -> Optional[float]:
        if not detailed:
            return None
        price_keys = {
            "entry": ["entry_price", "entryPrice", "buy_price"],
            "exit": ["exit_price", "exitPrice", "sell_price"],
            "stop_loss": ["stop_loss", "stopLoss", "stop_price"],
        }
        for key in price_keys.get(price_type, []):
            val = detailed.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None
