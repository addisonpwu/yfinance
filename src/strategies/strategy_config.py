"""
策略配置数据类

定义各策略的可配置参数，支持从配置文件加载
所有默认值从 constants.py 导入
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

# 从常量模块导入策略默认值
from src.config.constants import (
    # 动量爆发策略
    MOMENTUM_PRICE_BREAKOUT_THRESHOLD, MOMENTUM_VOLUME_BURST_MULTIPLIER,
    MOMENTUM_5D_THRESHOLD, MOMENTUM_20D_THRESHOLD, MOMENTUM_MIN_DATA_POINTS,
    # 吸筹加速策略
    ACCUMULATION_PERIOD, ACCUMULATION_VOLATILITY_THRESHOLD,
    ACCUMULATION_BREAKOUT_THRESHOLD, ACCUMULATION_VOLUME_RATIO_THRESHOLD,
    ACCUMULATION_RSI_PREV_MIN, ACCUMULATION_RSI_PREV_MAX,
    ACCUMULATION_RSI_CURRENT_THRESHOLD, ACCUMULATION_CONFIDENCE,
    # 波动率压缩策略
    VOLATILITY_BB_PERIOD, VOLATILITY_SQUEEZE_LOOKBACK, VOLATILITY_SQUEEZE_PERCENTILE,
    VOLATILITY_BREAKOUT_THRESHOLD, VOLATILITY_VOLUME_MULTIPLIER,
    VOLATILITY_VOLUME_PERIOD, VOLATILITY_CONFIDENCE,
    # 信号评分器
    SCORER_TREND_WEIGHT, SCORER_MOMENTUM_WEIGHT, SCORER_VOLUME_WEIGHT,
    SCORER_MARKET_WEIGHT, SCORER_SECTOR_WEIGHT, SCORER_PASS_THRESHOLD,
    SCORER_MIN_DATA_POINTS,
    # 技术指标
    RSI_PERIOD, BB_PERIOD, BB_STD_DEV,
    # VIX 阈值
    VIX_LOW, VIX_NORMAL, VIX_HIGH, VIX_PANIC,
)


@dataclass
class MomentumBreakoutConfig:
    """动量爆发策略配置"""
    price_breakout_threshold: float = MOMENTUM_PRICE_BREAKOUT_THRESHOLD
    volume_burst_multiplier: float = MOMENTUM_VOLUME_BURST_MULTIPLIER
    momentum_5d_threshold: float = MOMENTUM_5D_THRESHOLD
    momentum_20d_threshold: float = MOMENTUM_20D_THRESHOLD
    min_data_points: int = MOMENTUM_MIN_DATA_POINTS
    
    def validate(self) -> bool:
        return (
            self.price_breakout_threshold > 1.0 and
            self.volume_burst_multiplier > 0 and
            self.momentum_5d_threshold >= 0 and
            self.momentum_20d_threshold >= 0 and
            self.min_data_points >= 10
        )


@dataclass
class AccumulationAccelerationConfig:
    """主力吸筹加速策略配置"""
    accumulation_volatility_threshold: float = ACCUMULATION_VOLATILITY_THRESHOLD
    accumulation_period: int = ACCUMULATION_PERIOD
    acceleration_price_multiplier: float = ACCUMULATION_BREAKOUT_THRESHOLD
    acceleration_volume_ratio: float = ACCUMULATION_VOLUME_RATIO_THRESHOLD
    rsi_low_range_min: float = ACCUMULATION_RSI_PREV_MIN
    rsi_low_range_max: float = ACCUMULATION_RSI_PREV_MAX
    rsi_breakout_threshold: float = ACCUMULATION_RSI_CURRENT_THRESHOLD
    rsi_period: int = RSI_PERIOD
    min_data_points: int = ACCUMULATION_PERIOD
    
    def validate(self) -> bool:
        return (
            0 < self.accumulation_volatility_threshold < 1 and
            self.acceleration_price_multiplier > 1 and
            self.acceleration_volume_ratio > 0 and
            0 <= self.rsi_low_range_min < self.rsi_low_range_max <= 100 and
            self.rsi_breakout_threshold > 0 and
            self.min_data_points >= 20
        )


@dataclass
class VolatilitySqueezeConfig:
    """波动率压缩策略配置"""
    bb_period: int = VOLATILITY_BB_PERIOD
    bb_std_dev: float = BB_STD_DEV
    squeeze_lookback: int = VOLATILITY_SQUEEZE_LOOKBACK
    squeeze_percentile: float = VOLATILITY_SQUEEZE_PERCENTILE
    breakout_change_threshold: float = VOLATILITY_BREAKOUT_THRESHOLD
    volume_multiplier: float = VOLATILITY_VOLUME_MULTIPLIER
    volume_avg_period: int = VOLATILITY_VOLUME_PERIOD
    min_data_points: int = VOLATILITY_SQUEEZE_LOOKBACK
    
    def validate(self) -> bool:
        return (
            self.bb_period > 0 and
            self.bb_std_dev > 0 and
            self.squeeze_lookback > 0 and
            0 < self.squeeze_percentile < 1 and
            self.breakout_change_threshold > 0 and
            self.volume_multiplier > 0 and
            self.min_data_points >= 50
        )


@dataclass
class SignalScorerConfig:
    """信号评分器配置"""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend_following": SCORER_TREND_WEIGHT,
        "momentum_breakout": SCORER_MOMENTUM_WEIGHT,
        "volume_confirmation": SCORER_VOLUME_WEIGHT,
        "market_correction": SCORER_MARKET_WEIGHT,
        "sector_strength": SCORER_SECTOR_WEIGHT
    })
    pass_threshold: float = SCORER_PASS_THRESHOLD
    min_data_points: int = SCORER_MIN_DATA_POINTS
    
    def validate(self) -> bool:
        total_weight = sum(self.weights.values())
        return (
            abs(total_weight - 1.0) < 0.01 and
            0 < self.pass_threshold < 1 and
            self.min_data_points >= 20
        )
    
    def normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}


@dataclass
class MarketRegimeConfig:
    """市场环境识别策略配置"""
    # 技术指标参数
    trend_strength_threshold: float = 0.3
    low_volatility_threshold: float = 0.15
    high_volatility_threshold: float = 0.35
    extreme_volatility_threshold: float = 0.4
    health_score_threshold: float = 0.6
    risk_free_rate: float = 0.02
    min_data_points: int = 50
    
    # 宏观指标参数
    use_macro_indicators: bool = True
    macro_weight: float = 0.3
    
    # VIX 阈值 (从常量导入)
    vix_low_threshold: float = VIX_LOW
    vix_normal_threshold: float = VIX_NORMAL
    vix_high_threshold: float = VIX_HIGH
    vix_panic_threshold: float = VIX_PANIC
    
    # 美债收益率阈值
    tnx_low_threshold: float = 2.0
    tnx_high_threshold: float = 4.5
    
    # 收益率曲线倒挂阈值
    curve_inversion_threshold: float = 0.0
    
    # 宏观风险评分阈值
    macro_risk_low: float = 30.0
    macro_risk_high: float = 70.0
    
    def validate(self) -> bool:
        return (
            0 < self.trend_strength_threshold < 1 and
            0 < self.low_volatility_threshold < self.high_volatility_threshold and
            self.high_volatility_threshold < self.extreme_volatility_threshold and
            0 < self.health_score_threshold < 1 and
            self.min_data_points >= 20 and
            0 <= self.macro_weight <= 1 and
            self.vix_low_threshold < self.vix_normal_threshold < self.vix_high_threshold < self.vix_panic_threshold
        )


class StrategyConfigManager:
    """策略配置管理器，支持从配置文件加载"""
    
    _instance = None
    _configs: Dict = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_from_file(self, config_path: str = "config.json") -> Dict:
        """从配置文件加载策略配置"""
        if self._configs is not None:
            return self._configs
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            strategies_config = raw_config.get('strategies', {})
            
            self._configs = {
                'momentum_breakout': self._parse_momentum_config(strategies_config),
                'accumulation_acceleration': self._parse_accumulation_config(strategies_config),
                'volatility_squeeze': self._parse_volatility_config(strategies_config),
                'signal_scorer': self._parse_signal_scorer_config(strategies_config),
                'market_regime': self._parse_market_regime_config(strategies_config),
            }
            
            return self._configs
            
        except FileNotFoundError:
            self._configs = {
                'momentum_breakout': MomentumBreakoutConfig(),
                'accumulation_acceleration': AccumulationAccelerationConfig(),
                'volatility_squeeze': VolatilitySqueezeConfig(),
                'signal_scorer': SignalScorerConfig(),
                'market_regime': MarketRegimeConfig(),
            }
            return self._configs
    
    def _parse_momentum_config(self, strategies_config: Dict) -> MomentumBreakoutConfig:
        """解析动量策略配置"""
        cfg = strategies_config.get('momentum_breakout', {})
        return MomentumBreakoutConfig(
            price_breakout_threshold=cfg.get('price_breakout_threshold', MOMENTUM_PRICE_BREAKOUT_THRESHOLD),
            volume_burst_multiplier=cfg.get('volume_burst_multiplier', MOMENTUM_VOLUME_BURST_MULTIPLIER),
            momentum_5d_threshold=cfg.get('momentum_5d_threshold', MOMENTUM_5D_THRESHOLD),
            momentum_20d_threshold=cfg.get('momentum_20d_threshold', MOMENTUM_20D_THRESHOLD),
            min_data_points=cfg.get('min_data_points', MOMENTUM_MIN_DATA_POINTS),
        )
    
    def _parse_accumulation_config(self, strategies_config: Dict) -> AccumulationAccelerationConfig:
        """解析吸筹策略配置"""
        cfg = strategies_config.get('accumulation_acceleration', {})
        return AccumulationAccelerationConfig(
            accumulation_volatility_threshold=cfg.get('volatility_threshold', ACCUMULATION_VOLATILITY_THRESHOLD),
            accumulation_period=cfg.get('accumulation_period', ACCUMULATION_PERIOD),
            acceleration_price_multiplier=cfg.get('breakout_threshold', ACCUMULATION_BREAKOUT_THRESHOLD),
            acceleration_volume_ratio=cfg.get('volume_ratio_threshold', ACCUMULATION_VOLUME_RATIO_THRESHOLD),
            rsi_low_range_min=cfg.get('rsi_prev_min', ACCUMULATION_RSI_PREV_MIN),
            rsi_low_range_max=cfg.get('rsi_prev_max', ACCUMULATION_RSI_PREV_MAX),
            rsi_breakout_threshold=cfg.get('rsi_current_threshold', ACCUMULATION_RSI_CURRENT_THRESHOLD),
            rsi_period=cfg.get('rsi_period', RSI_PERIOD),
            min_data_points=cfg.get('min_data_points', ACCUMULATION_PERIOD),
        )
    
    def _parse_volatility_config(self, strategies_config: Dict) -> VolatilitySqueezeConfig:
        """解析波动率策略配置"""
        cfg = strategies_config.get('volatility_squeeze', {})
        return VolatilitySqueezeConfig(
            bb_period=cfg.get('bb_period', VOLATILITY_BB_PERIOD),
            bb_std_dev=cfg.get('bb_std_dev', BB_STD_DEV),
            squeeze_lookback=cfg.get('squeeze_lookback', VOLATILITY_SQUEEZE_LOOKBACK),
            squeeze_percentile=cfg.get('squeeze_percentile', VOLATILITY_SQUEEZE_PERCENTILE),
            breakout_change_threshold=cfg.get('breakout_threshold', VOLATILITY_BREAKOUT_THRESHOLD),
            volume_multiplier=cfg.get('volume_multiplier', VOLATILITY_VOLUME_MULTIPLIER),
            volume_avg_period=cfg.get('volume_period', VOLATILITY_VOLUME_PERIOD),
            min_data_points=cfg.get('min_data_points', VOLATILITY_SQUEEZE_LOOKBACK),
        )
    
    def _parse_signal_scorer_config(self, strategies_config: Dict) -> SignalScorerConfig:
        """解析信号评分器配置"""
        cfg = strategies_config.get('signal_scorer', {})
        default_weights = {
            "trend_following": SCORER_TREND_WEIGHT,
            "momentum_breakout": SCORER_MOMENTUM_WEIGHT,
            "volume_confirmation": SCORER_VOLUME_WEIGHT,
            "market_correction": SCORER_MARKET_WEIGHT,
            "sector_strength": SCORER_SECTOR_WEIGHT
        }
        weights = cfg.get('weights', default_weights)
        return SignalScorerConfig(
            weights=weights,
            pass_threshold=cfg.get('pass_threshold', SCORER_PASS_THRESHOLD),
            min_data_points=cfg.get('min_data_points', SCORER_MIN_DATA_POINTS),
        )
    
    def _parse_market_regime_config(self, strategies_config: Dict) -> MarketRegimeConfig:
        """解析市场环境策略配置"""
        cfg = strategies_config.get('market_regime', {})
        return MarketRegimeConfig(
            trend_strength_threshold=cfg.get('trend_strength_threshold', 0.3),
            low_volatility_threshold=cfg.get('low_volatility_threshold', 0.15),
            high_volatility_threshold=cfg.get('high_volatility_threshold', 0.35),
            extreme_volatility_threshold=cfg.get('extreme_volatility_threshold', 0.4),
            health_score_threshold=cfg.get('health_score_threshold', 0.6),
            risk_free_rate=cfg.get('risk_free_rate', 0.02),
            min_data_points=cfg.get('min_data_points', 50),
        )
    
    def get_config(self, strategy_name: str):
        """获取指定策略的配置"""
        if self._configs is None:
            self.load_from_file()
        return self._configs.get(strategy_name)


# 全局实例
strategy_config_manager = StrategyConfigManager()