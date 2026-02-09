from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import os

@dataclass
class APIConfig:
    base_delay: float = 0.5
    max_delay: float = 2.0
    min_delay: float = 0.1
    retry_attempts: int = 3
    max_workers: int = 4

@dataclass
class DataDownloadPeriodConfig:
    m1: str = "7d"      # 1分钟线
    h1: str = "730d"    # 1小时线
    d1: str = "max"     # 1日线

@dataclass
class DataConfig:
    max_cache_days: int = 7
    float_dtype: str = "float32"
    data_download_period: DataDownloadPeriodConfig = None
    enable_cache: bool = True  # 是否启用缓存
    enable_finviz: bool = True  # 是否启用 Finviz 数据获取

    def __post_init__(self):
        if self.data_download_period is None:
            self.data_download_period = DataDownloadPeriodConfig()

@dataclass
class AnalysisConfig:
    enable_realtime_output: bool = True
    enable_data_preprocessing: bool = True
    min_volume_threshold: int = 100000
    min_data_points_threshold: int = 20

@dataclass
class TechnicalIndicatorsConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std_dev: float = 2
    atr_period: int = 14
    ma_periods: List[int] = None
    # 新增指标配置
    cmo_period: int = 14
    williams_r_period: int = 14
    stochastic_period: int = 14
    stochastic_smooth_period: int = 3
    volume_z_score_period: int = 20

    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 200]

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
    bb_period: int = 20
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
class NewsConfig:
    timeout: int = 60000
    max_news_items: int = 5

@dataclass
class AIConfig:
    api_timeout: int = 30
    model: str = "deepseek-v3.2"
    max_data_points: int = 100

@dataclass
class AppConfig:
    api: APIConfig
    data: DataConfig
    analysis: AnalysisConfig
    technical_indicators: TechnicalIndicatorsConfig = None
    strategies: StrategiesConfig = None
    news: NewsConfig = None
    ai: AIConfig = None
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = TechnicalIndicatorsConfig()
        if self.strategies is None:
            self.strategies = StrategiesConfig()
        if self.news is None:
            self.news = NewsConfig()
        if self.ai is None:
            self.ai = AIConfig()

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
            
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            # 如果不存在，创建默认配置
            self._create_default_config(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = json.load(f)
        
        # 使用默认值填充缺失的配置项
        api_config = raw_config.get('api', {})
        data_config = raw_config.get('data', {})
        analysis_config = raw_config.get('analysis', {})
        tech_ind_config = raw_config.get('technical_indicators', {})
        strategies_config = raw_config.get('strategies', {})
        news_config = raw_config.get('news', {})
        ai_config = raw_config.get('ai', {})
        
        # 处理嵌套配置
        data_download_period_config = data_config.get('data_download_period', {})
        
        # 处理策略特定配置
        vcp_config = strategies_config.get('vcp_pocket_pivot', {})
        bollinger_squeeze_config = strategies_config.get('bollinger_squeeze', {})
        
        self._config = AppConfig(
            api=APIConfig(
                base_delay=api_config.get('base_delay', 0.5),
                max_delay=api_config.get('max_delay', 2.0),
                min_delay=api_config.get('min_delay', 0.1),
                retry_attempts=api_config.get('retry_attempts', 3),
                max_workers=api_config.get('max_workers', 4)
            ),
            data=DataConfig(
                max_cache_days=data_config.get('max_cache_days', 7),
                float_dtype=data_config.get('float_dtype', 'float32'),
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
                min_volume_threshold=analysis_config.get('min_volume_threshold', 100000),
                min_data_points_threshold=analysis_config.get('min_data_points_threshold', 20)
            ),
            technical_indicators=TechnicalIndicatorsConfig(
                rsi_period=tech_ind_config.get('rsi_period', 14),
                macd_fast=tech_ind_config.get('macd_fast', 12),
                macd_slow=tech_ind_config.get('macd_slow', 26),
                macd_signal=tech_ind_config.get('macd_signal', 9),
                bb_period=tech_ind_config.get('bb_period', 20),
                bb_std_dev=tech_ind_config.get('bb_std_dev', 2),
                atr_period=tech_ind_config.get('atr_period', 14),
                ma_periods=tech_ind_config.get('ma_periods', [5, 10, 20, 50, 200]),
                cmo_period=tech_ind_config.get('cmo_period', 14),
                williams_r_period=tech_ind_config.get('williams_r_period', 14),
                stochastic_period=tech_ind_config.get('stochastic_period', 14),
                stochastic_smooth_period=tech_ind_config.get('stochastic_smooth_period', 3),
                volume_z_score_period=tech_ind_config.get('volume_z_score_period', 20)
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
                    bb_period=bollinger_squeeze_config.get('bb_period', 20),
                    squeeze_lookback=bollinger_squeeze_config.get('squeeze_lookback', 100),
                    squeeze_percentile=bollinger_squeeze_config.get('squeeze_percentile', 0.10),
                    prolonged_squeeze_period=bollinger_squeeze_config.get('prolonged_squeeze_period', 5),
                    long_trend_period=bollinger_squeeze_config.get('long_trend_period', 200),
                    ma_slope_period=bollinger_squeeze_config.get('ma_slope_period', 5),
                    volume_period=bollinger_squeeze_config.get('volume_period', 50)
                )
            ),
            news=NewsConfig(
                timeout=news_config.get('timeout', 60000),
                max_news_items=news_config.get('max_news_items', 5)
            ),
            ai=AIConfig(
                api_timeout=ai_config.get('api_timeout', 30),
                model=ai_config.get('model', 'deepseek-v3.2'),
                max_data_points=ai_config.get('max_data_points', 100)
            )
        )
        return self._config
    
    def _create_default_config(self, config_path: str):
        """创建默认配置文件"""
        default_config = {
            "api": {
                "base_delay": 0.5,
                "max_delay": 2.0,
                "min_delay": 0.1,
                "retry_attempts": 3,
                "max_workers": 4
            },
            "data": {
                "max_cache_days": 7,
                "float_dtype": "float32",
                "data_download_period": {
                    "1m": "7d",
                    "1h": "730d",
                    "1d": "max"
                },
                "enable_cache": True,
                "enable_finviz": True
            },
            "analysis": {
                "enable_realtime_output": True,
                "enable_data_preprocessing": True,
                "min_volume_threshold": 100000,
                "min_data_points_threshold": 20
            },
            "technical_indicators": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std_dev": 2,
                "atr_period": 14,
                "ma_periods": [5, 10, 20, 50, 200]
            },
            "strategies": {
                "vcp_pocket_pivot": {
                    "ma_periods": [50, 150, 200],
                    "volatility_windows": [50, 20, 10],
                    "volume_avg_period": 50,
                    "pp_lookback_period": 10,
                    "pp_max_bias_ratio": 0.08
                },
                "bollinger_squeeze": {
                    "bb_period": 20,
                    "squeeze_lookback": 100,
                    "squeeze_percentile": 0.10,
                    "prolonged_squeeze_period": 5,
                    "long_trend_period": 200,
                    "ma_slope_period": 5,
                    "volume_period": 50
                }
            },
            "news": {
                "timeout": 60000,
                "max_news_items": 5
            },
            "ai": {
                "api_timeout": 30,
                "model": "deepseek-v3.2",
                "max_data_points": 100
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    def get_config(self) -> AppConfig:
        if self._config is None:
            self.load_config()
        return self._config

# 全局配置实例
config_manager = ConfigManager()