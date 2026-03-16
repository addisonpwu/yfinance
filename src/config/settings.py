"""
配置管理模块

所有默认值从 constants.py 导入，确保单一来源原则
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import os

# 从常量模块导入所有默认值
from src.config.constants import (
    # API 配置
    API_BASE_DELAY, API_MAX_DELAY, API_MIN_DELAY, API_RETRY_ATTEMPTS, API_MAX_WORKERS,
    # 数据配置
    DATA_MAX_CACHE_DAYS, DATA_FLOAT_DTYPE,
    # 分析配置
    ANALYSIS_MIN_VOLUME_THRESHOLD, ANALYSIS_MIN_DATA_POINTS,
    # 技术指标
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD_DEV, ATR_PERIOD,
    MA_PERIODS, CMO_PERIOD, WILLIAMS_R_PERIOD, STOCHASTIC_PERIOD, 
    STOCHASTIC_SMOOTH_PERIOD, VOLUME_Z_SCORE_PERIOD,
    # 新增技术指标 (2026-03-04)
    ADX_PERIOD, CMF_PERIOD, VWAP_ANCHOR, STOCH_RSI_PERIOD,
    STOCH_RSI_SMOOTH_K, STOCH_RSI_SMOOTH_D,
    # 新闻配置
    NEWS_TIMEOUT, NEWS_MAX_ITEMS, NEWS_DAYS_BACK, NEWS_CACHE_TTL_HOURS,
    # AI 配置
    AI_API_TIMEOUT, AI_MAX_DATA_POINTS, DEFAULT_AI_PROVIDERS,
    # 股票列表配置
    STOCK_LIST_JSON_PATH, STOCK_LIST_ENABLED,
)


# 速度模式预设配置
# 注意：enable_cache 由配置文件单独控制，不随速度模式改变
SPEED_MODE_PRESETS = {
    "fast": {
        "base_delay": 0.2,
        "max_delay": 1.0,
        "min_delay": 0.1,
        "max_workers": 8,
        "ai_timeout": 20,
        "description": "快速模式：高并行、低延迟，适合测试"
    },
    "balanced": {
        "base_delay": API_BASE_DELAY,
        "max_delay": API_MAX_DELAY,
        "min_delay": API_MIN_DELAY,
        "max_workers": API_MAX_WORKERS,
        "ai_timeout": AI_API_TIMEOUT,
        "description": "平衡模式：速度与稳定性兼顾，推荐日常使用"
    },
    "safe": {
        "base_delay": 1.0,
        "max_delay": 5.0,
        "min_delay": 0.5,
        "max_workers": 1,
        "ai_timeout": 45,
        "description": "安全模式：低并行、高延迟，避免 API 限流"
    }
}


@dataclass
class APIConfig:
    base_delay: float = API_BASE_DELAY
    max_delay: float = API_MAX_DELAY
    min_delay: float = API_MIN_DELAY
    retry_attempts: int = API_RETRY_ATTEMPTS
    max_workers: int = API_MAX_WORKERS


@dataclass
class DataDownloadPeriodConfig:
    m1: str = "7d"      # 1分钟线
    h1: str = "730d"    # 1小时线
    d1: str = "max"     # 1日线


@dataclass
class DataConfig:
    max_cache_days: int = DATA_MAX_CACHE_DAYS
    float_dtype: str = DATA_FLOAT_DTYPE
    data_download_period: DataDownloadPeriodConfig = None
    enable_cache: bool = True
    enable_finviz: bool = True

    def __post_init__(self):
        if self.data_download_period is None:
            self.data_download_period = DataDownloadPeriodConfig()


@dataclass
class AnalysisConfig:
    enable_realtime_output: bool = True
    enable_data_preprocessing: bool = True
    min_volume_threshold: int = ANALYSIS_MIN_VOLUME_THRESHOLD
    min_data_points_threshold: int = ANALYSIS_MIN_DATA_POINTS


@dataclass
class TechnicalIndicatorsConfig:
    rsi_period: int = RSI_PERIOD
    macd_fast: int = MACD_FAST
    macd_slow: int = MACD_SLOW
    macd_signal: int = MACD_SIGNAL
    bb_period: int = BB_PERIOD
    bb_std_dev: float = BB_STD_DEV
    atr_period: int = ATR_PERIOD
    ma_periods: List[int] = None
    cmo_period: int = CMO_PERIOD
    williams_r_period: int = WILLIAMS_R_PERIOD
    stochastic_period: int = STOCHASTIC_PERIOD
    stochastic_smooth_period: int = STOCHASTIC_SMOOTH_PERIOD
    volume_z_score_period: int = VOLUME_Z_SCORE_PERIOD
    # 新增技术指标配置 (2026-03-04)
    adx_period: int = ADX_PERIOD
    cmf_period: int = CMF_PERIOD
    vwap_anchor: str = VWAP_ANCHOR
    stoch_rsi_period: int = STOCH_RSI_PERIOD
    stoch_rsi_smooth_k: int = STOCH_RSI_SMOOTH_K
    stoch_rsi_smooth_d: int = STOCH_RSI_SMOOTH_D

    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = MA_PERIODS


@dataclass
class VCPSpecificConfig:
    ma_periods: List[int] = None
    volatility_windows: List[int] = None
    volume_avg_period: int = 50
    pp_lookback_period: int = 10
    pp_max_bias_ratio: float = 0.08
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [50, 150, 200]
        if self.volatility_windows is None:
            self.volatility_windows = [50, 20, 10]


@dataclass
class BollingerSqueezeSpecificConfig:
    bb_period: int = BB_PERIOD
    squeeze_lookback: int = 100
    squeeze_percentile: float = 0.10
    prolonged_squeeze_period: int = 5
    long_trend_period: int = 200
    ma_slope_period: int = 5
    volume_period: int = 50


@dataclass
class StrategiesConfig:
    vcp_pocket_pivot: VCPSpecificConfig = None
    bollinger_squeeze: BollingerSqueezeSpecificConfig = None
    
    def __post_init__(self):
        if self.vcp_pocket_pivot is None:
            self.vcp_pocket_pivot = VCPSpecificConfig()
        if self.bollinger_squeeze is None:
            self.bollinger_squeeze = BollingerSqueezeSpecificConfig()


@dataclass
class NewsGoogleConfig:
    """Google News 配置"""
    default_language: str = "zh-TW"
    default_region: str = "TW"


@dataclass
class NewsConfig:
    timeout: int = NEWS_TIMEOUT
    max_news_items: int = NEWS_MAX_ITEMS
    days_back: int = NEWS_DAYS_BACK
    cache_ttl_hours: int = NEWS_CACHE_TTL_HOURS
    provider: str = "both"  # "google" | "yahoo" | "both"
    google: NewsGoogleConfig = None
    
    def __post_init__(self):
        if self.google is None:
            self.google = NewsGoogleConfig()


@dataclass
class StockListConfig:
    """股票列表配置"""
    json_path: str = STOCK_LIST_JSON_PATH
    enabled: bool = STOCK_LIST_ENABLED


@dataclass
class AIProviderConfig:
    """单个 AI 提供商配置"""
    default_model: str = ""
    available_models: List[str] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = []


@dataclass
class AIProvidersConfig:
    """AI 提供商配置"""
    iflow: AIProviderConfig = None
    nvidia: AIProviderConfig = None
    gemini: AIProviderConfig = None
    
    def __post_init__(self):
        if self.iflow is None:
            self.iflow = AIProviderConfig(
                default_model=DEFAULT_AI_PROVIDERS["iflow"]["default_model"],
                available_models=DEFAULT_AI_PROVIDERS["iflow"]["available_models"]
            )
        if self.nvidia is None:
            self.nvidia = AIProviderConfig(
                default_model=DEFAULT_AI_PROVIDERS["nvidia"]["default_model"],
                available_models=DEFAULT_AI_PROVIDERS["nvidia"]["available_models"]
            )
        if self.gemini is None:
            self.gemini = AIProviderConfig(
                default_model=DEFAULT_AI_PROVIDERS["gemini"]["default_model"],
                available_models=DEFAULT_AI_PROVIDERS["gemini"]["available_models"]
            )


@dataclass
class AIConfig:
    api_timeout: int = AI_API_TIMEOUT
    model: str = DEFAULT_AI_PROVIDERS["iflow"]["default_model"]
    max_data_points: int = AI_MAX_DATA_POINTS
    providers: AIProvidersConfig = None
    
    def __post_init__(self):
        if self.providers is None:
            self.providers = AIProvidersConfig()


@dataclass
class AppConfig:
    api: APIConfig
    data: DataConfig
    analysis: AnalysisConfig
    technical_indicators: TechnicalIndicatorsConfig = None
    strategies: StrategiesConfig = None
    news: NewsConfig = None
    ai: AIConfig = None
    stock_list: StockListConfig = None
    speed_mode: str = "balanced"
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = TechnicalIndicatorsConfig()
        if self.strategies is None:
            self.strategies = StrategiesConfig()
        if self.news is None:
            self.news = NewsConfig()
        if self.ai is None:
            self.ai = AIConfig()
        if self.stock_list is None:
            self.stock_list = StockListConfig()
    
    def apply_speed_mode(self, speed_mode: str = None):
        """应用速度模式预设"""
        mode = speed_mode or self.speed_mode
        
        if mode not in SPEED_MODE_PRESETS:
            print(f"警告: 未知速度模式 '{mode}'，使用 'balanced'")
            mode = "balanced"
        
        preset = SPEED_MODE_PRESETS[mode]
        
        self.api.base_delay = preset["base_delay"]
        self.api.max_delay = preset["max_delay"]
        self.api.min_delay = preset["min_delay"]
        self.api.max_workers = preset["max_workers"]
        # enable_cache 由配置文件控制，不随速度模式改变
        self.ai.api_timeout = preset["ai_timeout"]
        self.speed_mode = mode
        
        print(f"已应用速度模式: {mode} ({preset['description']})")
        return self


class ConfigManager:
    _instance = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: str = "config.json") -> AppConfig:
        if self._config is not None:
            return self._config
            
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = json.load(f)
        
        speed_mode = raw_config.get('speed_mode', 'balanced')
        api_config = raw_config.get('api', {})
        data_config = raw_config.get('data', {})
        analysis_config = raw_config.get('analysis', {})
        tech_ind_config = raw_config.get('technical_indicators', {})
        strategies_config = raw_config.get('strategies', {})
        news_config = raw_config.get('news', {})
        ai_config = raw_config.get('ai', {})
        stock_list_config = raw_config.get('stock_list', {})
        
        data_download_period_config = data_config.get('data_download_period', {})
        vcp_config = strategies_config.get('vcp_pocket_pivot', {})
        bollinger_squeeze_config = strategies_config.get('bollinger_squeeze', {})
        
        providers_config = ai_config.get('providers', {})
        iflow_config = providers_config.get('iflow', {})
        nvidia_config = providers_config.get('nvidia', {})
        gemini_config = providers_config.get('gemini', {})
        
        self._config = AppConfig(
            api=APIConfig(
                base_delay=api_config.get('base_delay', API_BASE_DELAY),
                max_delay=api_config.get('max_delay', API_MAX_DELAY),
                min_delay=api_config.get('min_delay', API_MIN_DELAY),
                retry_attempts=api_config.get('retry_attempts', API_RETRY_ATTEMPTS),
                max_workers=api_config.get('max_workers', API_MAX_WORKERS)
            ),
            data=DataConfig(
                max_cache_days=data_config.get('max_cache_days', DATA_MAX_CACHE_DAYS),
                float_dtype=data_config.get('float_dtype', DATA_FLOAT_DTYPE),
                data_download_period=DataDownloadPeriodConfig(
                    m1=data_download_period_config.get('1m', '7d'),
                    h1=data_download_period_config.get('1h', '730d'),
                    d1=data_download_period_config.get('1d', 'max')
                ),
                enable_cache=data_config.get('enable_cache', True),
                enable_finviz=data_config.get('enable_finviz', True)
            ),
            analysis=AnalysisConfig(
                enable_realtime_output=analysis_config.get('enable_realtime_output', True),
                enable_data_preprocessing=analysis_config.get('enable_data_preprocessing', True),
                min_volume_threshold=analysis_config.get('min_volume_threshold', ANALYSIS_MIN_VOLUME_THRESHOLD),
                min_data_points_threshold=analysis_config.get('min_data_points_threshold', ANALYSIS_MIN_DATA_POINTS)
            ),
            technical_indicators=TechnicalIndicatorsConfig(
                rsi_period=tech_ind_config.get('rsi_period', RSI_PERIOD),
                macd_fast=tech_ind_config.get('macd_fast', MACD_FAST),
                macd_slow=tech_ind_config.get('macd_slow', MACD_SLOW),
                macd_signal=tech_ind_config.get('macd_signal', MACD_SIGNAL),
                bb_period=tech_ind_config.get('bb_period', BB_PERIOD),
                bb_std_dev=tech_ind_config.get('bb_std_dev', BB_STD_DEV),
                atr_period=tech_ind_config.get('atr_period', ATR_PERIOD),
                ma_periods=tech_ind_config.get('ma_periods', MA_PERIODS),
                cmo_period=tech_ind_config.get('cmo_period', CMO_PERIOD),
                williams_r_period=tech_ind_config.get('williams_r_period', WILLIAMS_R_PERIOD),
                stochastic_period=tech_ind_config.get('stochastic_period', STOCHASTIC_PERIOD),
                stochastic_smooth_period=tech_ind_config.get('stochastic_smooth_period', STOCHASTIC_SMOOTH_PERIOD),
                volume_z_score_period=tech_ind_config.get('volume_z_score_period', VOLUME_Z_SCORE_PERIOD)
            ),
            strategies=StrategiesConfig(
                vcp_pocket_pivot=VCPSpecificConfig(
                    ma_periods=vcp_config.get('ma_periods', [50, 150, 200]),
                    volatility_windows=vcp_config.get('volatility_windows', [50, 20, 10]),
                    volume_avg_period=vcp_config.get('volume_avg_period', 50),
                    pp_lookback_period=vcp_config.get('pp_lookback_period', 10),
                    pp_max_bias_ratio=vcp_config.get('pp_max_bias_ratio', 0.08)
                ),
                bollinger_squeeze=BollingerSqueezeSpecificConfig(
                    bb_period=bollinger_squeeze_config.get('bb_period', BB_PERIOD),
                    squeeze_lookback=bollinger_squeeze_config.get('squeeze_lookback', 100),
                    squeeze_percentile=bollinger_squeeze_config.get('squeeze_percentile', 0.10),
                    prolonged_squeeze_period=bollinger_squeeze_config.get('prolonged_squeeze_period', 5),
                    long_trend_period=bollinger_squeeze_config.get('long_trend_period', 200),
                    ma_slope_period=bollinger_squeeze_config.get('ma_slope_period', 5),
                    volume_period=bollinger_squeeze_config.get('volume_period', 50)
                )
            ),
            news=NewsConfig(
                timeout=news_config.get('timeout', NEWS_TIMEOUT),
                max_news_items=news_config.get('max_news_items', NEWS_MAX_ITEMS),
                days_back=news_config.get('days_back', NEWS_DAYS_BACK),
                cache_ttl_hours=news_config.get('cache_ttl_hours', NEWS_CACHE_TTL_HOURS),
                provider=news_config.get('provider', 'both'),
                google=NewsGoogleConfig(
                    default_language=news_config.get('google', {}).get('default_language', 'zh-TW'),
                    default_region=news_config.get('google', {}).get('default_region', 'TW')
                )
            ),
            ai=AIConfig(
                api_timeout=ai_config.get('api_timeout', AI_API_TIMEOUT),
                model=ai_config.get('model', DEFAULT_AI_PROVIDERS["iflow"]["default_model"]),
                max_data_points=ai_config.get('max_data_points', AI_MAX_DATA_POINTS),
                providers=AIProvidersConfig(
                    iflow=AIProviderConfig(
                        default_model=iflow_config.get('default_model', DEFAULT_AI_PROVIDERS["iflow"]["default_model"]),
                        available_models=iflow_config.get('available_models', DEFAULT_AI_PROVIDERS["iflow"]["available_models"])
                    ),
                    nvidia=AIProviderConfig(
                        default_model=nvidia_config.get('default_model', DEFAULT_AI_PROVIDERS["nvidia"]["default_model"]),
                        available_models=nvidia_config.get('available_models', DEFAULT_AI_PROVIDERS["nvidia"]["available_models"])
                    ),
                    gemini=AIProviderConfig(
                        default_model=gemini_config.get('default_model', DEFAULT_AI_PROVIDERS["gemini"]["default_model"]),
                        available_models=gemini_config.get('available_models', DEFAULT_AI_PROVIDERS["gemini"]["available_models"])
                    )
                )
            ),
            stock_list=StockListConfig(
                json_path=stock_list_config.get('json_path', STOCK_LIST_JSON_PATH),
                enabled=stock_list_config.get('enabled', STOCK_LIST_ENABLED)
            ),
            speed_mode=speed_mode
        )
        
        if speed_mode in SPEED_MODE_PRESETS:
            self._config.apply_speed_mode(speed_mode)
        
        return self._config
    
    def _create_default_config(self, config_path: str):
        """创建默认配置文件"""
        default_config = {
            "speed_mode": "balanced",
            "api": {
                "base_delay": API_BASE_DELAY,
                "max_delay": API_MAX_DELAY,
                "min_delay": API_MIN_DELAY,
                "retry_attempts": API_RETRY_ATTEMPTS,
                "max_workers": API_MAX_WORKERS
            },
            "data": {
                "max_cache_days": DATA_MAX_CACHE_DAYS,
                "float_dtype": DATA_FLOAT_DTYPE,
                "data_download_period": {"1m": "7d", "1h": "730d", "1d": "max"},
                "enable_cache": True,
                "enable_finviz": True
            },
            "analysis": {
                "enable_realtime_output": True,
                "enable_data_preprocessing": True,
                "min_volume_threshold": ANALYSIS_MIN_VOLUME_THRESHOLD,
                "min_data_points_threshold": ANALYSIS_MIN_DATA_POINTS
            },
            "technical_indicators": {
                "rsi_period": RSI_PERIOD,
                "macd_fast": MACD_FAST,
                "macd_slow": MACD_SLOW,
                "macd_signal": MACD_SIGNAL,
                "bb_period": BB_PERIOD,
                "bb_std_dev": BB_STD_DEV,
                "atr_period": ATR_PERIOD,
                "ma_periods": MA_PERIODS,
                "cmo_period": CMO_PERIOD,
                "williams_r_period": WILLIAMS_R_PERIOD,
                "stochastic_period": STOCHASTIC_PERIOD,
                "stochastic_smooth_period": STOCHASTIC_SMOOTH_PERIOD,
                "volume_z_score_period": VOLUME_Z_SCORE_PERIOD
            },
            "strategies": {
                "momentum_breakout": {
                    "price_breakout_threshold": 1.01,
                    "volume_burst_multiplier": 2.0,
                    "momentum_5d_threshold": 0.03,
                    "momentum_20d_threshold": 0.05,
                    "min_data_points": 21,
                    "confidence": 0.8
                },
                "accumulation_acceleration": {
                    "accumulation_period": 30,
                    "volatility_threshold": 0.15,
                    "volume_trend_period": 30,
                    "breakout_threshold": 1.015,
                    "volume_ratio_threshold": 2.5,
                    "rsi_prev_min": 40,
                    "rsi_prev_max": 60,
                    "rsi_current_threshold": 65,
                    "min_data_points": 30,
                    "confidence": 0.8
                },
                "volatility_squeeze": {
                    "bb_period": 20,
                    "bb_std_dev": 2,
                    "squeeze_lookback": 100,
                    "squeeze_percentile": 0.1,
                    "breakout_threshold": 0.02,
                    "volume_multiplier": 1.5,
                    "volume_period": 50,
                    "min_data_points": 100,
                    "confidence": 0.8
                },
                "signal_scorer": {
                    "weights": {
                        "trend_following": 0.25,
                        "momentum_breakout": 0.20,
                        "volume_confirmation": 0.15,
                        "market_correction": 0.20,
                        "sector_strength": 0.20
                    },
                    "pass_threshold": 0.7,
                    "min_data_points": 50
                },
                "market_regime": {
                    "min_data_points": 50,
                    "health_score_threshold": 0.6,
                    "trend_strength_threshold": 0.3,
                    "low_volatility_threshold": 0.15,
                    "high_volatility_threshold": 0.35,
                    "extreme_volatility_threshold": 0.4,
                    "risk_free_rate": 0.02
                },
                "vcp_pocket_pivot": {
                    "ma_periods": [50, 150, 200],
                    "volatility_windows": [50, 20, 10],
                    "volume_avg_period": 50,
                    "pp_lookback_period": 10,
                    "pp_max_bias_ratio": 0.08
                },
                "bollinger_squeeze": {
                    "bb_period": BB_PERIOD,
                    "squeeze_lookback": 100,
                    "squeeze_percentile": 0.10,
                    "prolonged_squeeze_period": 5,
                    "long_trend_period": 200,
                    "ma_slope_period": 5,
                    "volume_period": 50
                }
            },
            "news": {
                "timeout": NEWS_TIMEOUT,
                "max_news_items": NEWS_MAX_ITEMS,
                "days_back": NEWS_DAYS_BACK,
                "cache_ttl_hours": NEWS_CACHE_TTL_HOURS
            },
            "ai": {
                "api_timeout": AI_API_TIMEOUT,
                "model": DEFAULT_AI_PROVIDERS["iflow"]["default_model"],
                "max_data_points": AI_MAX_DATA_POINTS,
                "providers": DEFAULT_AI_PROVIDERS
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    def get_config(self) -> AppConfig:
        if self._config is None:
            self.load_config()
        return self._config
    
    def apply_speed_mode(self, speed_mode: str) -> AppConfig:
        if self._config is None:
            self.load_config()
        self._config.apply_speed_mode(speed_mode)
        return self._config


# 全局配置实例
config_manager = ConfigManager()
