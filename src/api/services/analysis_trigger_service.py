"""
Analysis Trigger Service

Handles async triggering of NVIDIA multi-model stock analysis,
task state management, and result persistence.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import pandas as pd

from src.config.settings import config_manager
from src.config.constants import AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT
from src.data.loaders.yahoo_loader import YahooFinanceRepository, calculate_technical_indicators
from src.ai.analyzer.nvidia_analyzer import NvidiaAIAnalyzer
from src.db.database import get_engine, get_session_factory
from src.db.models.stock import Stock
from src.repositories.stock_repo import StockRepository
from src.repositories.ai_analysis_repo import AIAnalysisRepository
from src.api.schemas.ai_analysis import AIAnalysisCreate
from src.utils.logger import LoggerManager

logger = LoggerManager.get_logger("services.analysis_trigger")


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskState:
    """Represents the state of a single analysis task."""

    def __init__(self, task_id: str, symbol: str, market: str, interval: str, models: List[str]):
        self.task_id = task_id
        self.symbol = symbol
        self.market = market
        self.interval = interval
        self.models = models
        self.status = TaskStatus.QUEUED
        self.current_model: Optional[str] = None
        self.current_step: str = ""
        self.completed_models: List[str] = []
        self.failed_models: List[str] = []
        self.results: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "symbol": self.symbol,
            "market": self.market,
            "interval": self.interval,
            "status": self.status.value,
            "current_model": self.current_model,
            "current_step": self.current_step,
            "progress": {
                "current_model_index": len(self.completed_models) + (1 if self.current_model and self.current_model not in self.completed_models else 0),
                "total_models": len(self.models),
                "current_model_name": self.current_model or "",
                "completed_models": self.completed_models[:],
                "failed_models": self.failed_models[:],
            },
            "results": self.results[:],
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AnalysisTriggerService:
    """
    Manages async NVIDIA multi-model analysis tasks.

    Uses an in-memory task registry (single-process deployment).
    Each task runs as an asyncio.Task in the background.
    """

    def __init__(self):
        self._tasks: Dict[str, TaskState] = {}
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID."""
        state = self._tasks.get(task_id)
        if state:
            return state.to_dict()
        return None

    def get_active_tasks(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active tasks, optionally filtered by symbol."""
        tasks = []
        for state in self._tasks.values():
            if symbol and state.symbol != symbol:
                continue
            tasks.append(state.to_dict())
        return tasks

    async def trigger_analysis(
        self,
        symbol: str,
        market: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Trigger a new NVIDIA multi-model analysis task.

        Args:
            symbol: Stock symbol (e.g., '3988.HK')
            market: Market code (e.g., 'HK', 'US')
            interval: Data interval ('1d', '1h', '1m')
            force_refresh: Force refresh cached data

        Returns:
            Dict with task_id and initial status.

        Raises:
            ValueError: If a task for this symbol is already running.
        """
        # Check for existing active task for this symbol
        for state in self._tasks.values():
            if state.symbol == symbol and state.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                return {
                    "task_id": state.task_id,
                    "status": state.status.value,
                    "message": f"Task already running for {symbol}",
                    "existing": True,
                }

        # Get available NVIDIA models from config
        config = config_manager.get_config()
        models = config.ai.providers.nvidia.available_models
        if not models:
            raise ValueError("No NVIDIA models configured. Check ai.providers.nvidia.available_models in config.json")

        task_id = str(uuid.uuid4())[:8]
        task_state = TaskState(task_id=task_id, symbol=symbol, market=market, interval=interval, models=models)
        task_state.started_at = datetime.now()
        self._tasks[task_id] = task_state

        # Launch background task
        asyncio.create_task(
            self._run_analysis_task(task_state, force_refresh),
            name=f"analysis_{symbol}_{task_id}",
        )

        logger.info(f"Triggered analysis task {task_id} for {symbol} (models: {models})")
        return {
            "task_id": task_id,
            "status": task_state.status.value,
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "models": models,
            "existing": False,
        }

    async def _run_analysis_task(self, task_state: TaskState, force_refresh: bool):
        """Execute the full multi-model analysis pipeline."""
        task_state.status = TaskStatus.RUNNING
        symbol = task_state.symbol
        market = task_state.market
        interval = task_state.interval

        try:
            # Step 1: Fetch data
            logger.info(f"[{task_state.task_id}] Fetching data for {symbol}")
            task_state.current_step = "Fetching data..."

            data_repo = YahooFinanceRepository()
            hist = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: data_repo.get_historical_data(symbol, market, interval=interval, force_refresh=force_refresh),
            )
            info = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: data_repo.get_financial_info(symbol),
            )

            if hist is None or hist.empty or len(hist) < 2 or not info:
                raise ValueError("Insufficient data for analysis")

            # Step 2: Calculate technical indicators
            task_state.current_step = "Calculating technical indicators..."
            config = config_manager.get_config()
            hist = calculate_technical_indicators(hist, config)

            # Step 3: Analyze with each model sequentially
            stock_data = {
                "symbol": symbol,
                "strategies": ["跳过策略"],
                "info": info,
                "market": market,
            }

            for model_name in task_state.models:
                task_state.current_model = model_name
                task_state.current_step = f"Analyzing with {model_name}..."
                logger.info(f"[{task_state.task_id}] Analyzing {symbol} with {model_name}")

                try:
                    result = await self._analyze_single_model(stock_data, hist, interval, model_name)
                    if result:
                        task_state.completed_models.append(model_name)
                        task_state.results.append(result)
                        logger.info(f"[{task_state.task_id}] {model_name} completed (confidence: {result.get('confidence', 0):.0%})")

                        # Persist immediately after each model completes
                        await self._persist_result(symbol, interval, result)
                    else:
                        task_state.failed_models.append(model_name)
                        logger.warning(f"[{task_state.task_id}] {model_name} returned no result")
                except Exception as e:
                    task_state.failed_models.append(model_name)
                    logger.error(f"[{task_state.task_id}] {model_name} failed: {e}")

                task_state.current_model = None

            # Determine final status
            if task_state.completed_models:
                task_state.status = TaskStatus.COMPLETED
                task_state.completed_at = datetime.now()
                logger.info(f"[{task_state.task_id}] Completed: {len(task_state.completed_models)}/{len(task_state.models)} models succeeded")
            else:
                task_state.status = TaskStatus.FAILED
                task_state.error = "All model analyses failed"
                task_state.completed_at = datetime.now()
                logger.error(f"[{task_state.task_id}] All models failed for {symbol}")

        except Exception as e:
            task_state.status = TaskStatus.FAILED
            task_state.error = str(e)
            task_state.completed_at = datetime.now()
            logger.error(f"[{task_state.task_id}] Task failed for {symbol}: {e}")

    async def _analyze_single_model(
        self,
        stock_data: Dict[str, Any],
        hist: pd.DataFrame,
        interval: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Run 3-step analysis for a single NVIDIA model."""
        analyzer = NvidiaAIAnalyzer(enable_cache=False)

        # Run the 3-step analysis in a thread executor (it's sync internally)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: analyzer._step_by_step_analysis(stock_data, hist, model_name, use_multi_timeframe=True),
        )

        if not result:
            return None

        direction = "中性"
        if result.detailed_analysis:
            direction = result.detailed_analysis.get("direction", "中性")

        return {
            "provider": "nvidia",
            "model_used": result.model_used or model_name,
            "summary": result.summary,
            "confidence": result.confidence,
            "detailed_analysis": result.detailed_analysis,
            "direction": direction,
        }

    async def _persist_result(self, symbol: str, interval: str, result: Dict[str, Any]):
        """Persist a single model's analysis result to the database."""
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                stock_repo = StockRepository(session)
                ai_repo = AIAnalysisRepository(session)

                # Get or create stock
                stock = await stock_repo.get_by_symbol(symbol)
                if not stock:
                    stock = await stock_repo.create(
                        StockCreateForPersist(
                            symbol=symbol,
                            name=symbol,
                            market="US" if not symbol.endswith(".HK") else "HK",
                        )
                    )

                analysis_create = AIAnalysisCreate(
                    provider=result["provider"],
                    model_used=result["model_used"],
                    interval=interval,
                    summary=result["summary"][:4000],  # Truncate if too long
                    confidence=result["confidence"],
                    recommendation=result.get("detailed_analysis", {}).get("recommendation") if result.get("detailed_analysis") else None,
                    detailed_analysis=result.get("detailed_analysis"),
                    analyzed_at=datetime.now(),
                )

                await ai_repo.create(analysis_create, stock.id)
                await session.commit()

                # Cleanup old records
                await ai_repo.cleanup_old_records(
                    stock_id=stock.id,
                    provider=result["provider"],
                    interval=interval,
                    max_records=AI_ANALYSIS_RETENTION_MAX_RECORDS_DEFAULT,
                )

                logger.info(f"Persisted NVIDIA analysis for {symbol} ({result['model_used']})")
        except Exception as e:
            logger.error(f"Failed to persist result for {symbol}: {e}")
            raise


# Minimal schema-like class for stock creation (avoid circular import)
from pydantic import BaseModel


class StockCreateForPersist(BaseModel):
    symbol: str
    name: str
    market: str


# Global singleton instance
_analysis_trigger_service: Optional[AnalysisTriggerService] = None


def get_analysis_trigger_service() -> AnalysisTriggerService:
    """Get or create the AnalysisTriggerService singleton."""
    global _analysis_trigger_service
    if _analysis_trigger_service is None:
        _analysis_trigger_service = AnalysisTriggerService()
    return _analysis_trigger_service
