"""
策略配置数据类

定义各策略的可配置参数，支持从配置文件加载
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class MomentumBreakoutConfig:
    """动量爆发策略配置"""
    # 价格突破阈值：当前价 > 20日最高价 × 该值
    price_breakout_threshold: float = 1.01
    # 量能爆发倍数：当日成交量 > 20日均量 × 该值
    volume_burst_multiplier: float = 2.0
    # 5日涨幅阈值
    momentum_5d_threshold: float = 0.03
    # 20日涨幅阈值
    momentum_20d_threshold: float = 0.05
    # 最小数据点数
    min_data_points: int = 21
    
    def validate(self) -> bool:
        """验证配置参数"""
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
    # 吸筹期波动幅度阈值：< 该值
    accumulation_volatility_threshold: float = 0.15
    # 吸筹期天数
    accumulation_period: int = 30
    # 加速信号价格突破倍数
    acceleration_price_multiplier: float = 1.015
    # 加速信号量比阈值
    acceleration_volume_ratio: float = 2.5
    # RSI 低位区间下限
    rsi_low_range_min: float = 40.0
    # RSI 低位区间上限
    rsi_low_range_max: float = 60.0
    # RSI 突破阈值
    rsi_breakout_threshold: float = 65.0
    # RSI 周期
    rsi_period: int = 14
    # 最小数据点数
    min_data_points: int = 30
    
    def validate(self) -> bool:
        """验证配置参数"""
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
    # 布林带周期
    bb_period: int = 20
    # 布林带标准差倍数
    bb_std_dev: float = 2.0
    # 挤压检测回看周期
    squeeze_lookback: int = 100
    # 挤压百分位阈值
    squeeze_percentile: float = 0.10
    # 突破确认涨幅阈值
    breakout_change_threshold: float = 0.02
    # 量能配合倍数
    volume_multiplier: float = 1.5
    # 成交量平均周期
    volume_avg_period: int = 50
    # 最小数据点数
    min_data_points: int = 100
    
    def validate(self) -> bool:
        """验证配置参数"""
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
    # 各信号权重
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend_following": 0.25,
        "momentum_breakout": 0.20,
        "volume_confirmation": 0.15,
        "market_correction": 0.20,
        "sector_strength": 0.20
    })
    # 通过阈值
    pass_threshold: float = 0.7
    # 最小数据点数
    min_data_points: int = 50
    
    def validate(self) -> bool:
        """验证配置参数"""
        total_weight = sum(self.weights.values())
        return (
            abs(total_weight - 1.0) < 0.01 and  # 权重和接近1
            0 < self.pass_threshold < 1 and
            self.min_data_points >= 20
        )
    
    def normalize_weights(self) -> None:
        """归一化权重"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}


@dataclass
class MarketRegimeConfig:
    """
    市场环境识别策略配置
    
    整合技术指标和宏观指标
    """
    # === 技术指标参数 ===
    # 趋势强度阈值
    trend_strength_threshold: float = 0.3
    # 低波动率阈值
    low_volatility_threshold: float = 0.15
    # 高波动率阈值
    high_volatility_threshold: float = 0.35
    # 极高波动率阈值
    extreme_volatility_threshold: float = 0.4
    # 健康得分阈值
    health_score_threshold: float = 0.6
    # 无风险利率（年化）
    risk_free_rate: float = 0.02
    # 最小数据点数
    min_data_points: int = 50
    
    # === 宏观指标参数 ===
    # 是否启用宏观指标分析
    use_macro_indicators: bool = True
    # 宏观指标权重 (0-1，与技术指标权重互补)
    macro_weight: float = 0.3
    
    # VIX 阈值
    vix_low_threshold: float = 15.0      # VIX < 15: 低波动
    vix_normal_threshold: float = 20.0   # VIX 15-20: 正常
    vix_high_threshold: float = 30.0     # VIX 20-30: 高波动
    vix_panic_threshold: float = 40.0    # VIX > 40: 恐慌
    
    # 美债收益率阈值
    tnx_low_threshold: float = 2.0       # < 2%: 宽松
    tnx_high_threshold: float = 4.5      # > 4.5%: 紧缩
    
    # 收益率曲线倒挂阈值
    curve_inversion_threshold: float = 0.0  # 10Y-2Y < 0 表示倒挂
    
    # 宏观风险评分阈值
    macro_risk_low: float = 30.0         # 低风险
    macro_risk_high: float = 70.0        # 高风险
    
    def validate(self) -> bool:
        """验证配置参数"""
        return (
            0 < self.trend_strength_threshold < 1 and
            0 < self.low_volatility_threshold < self.high_volatility_threshold and
            self.high_volatility_threshold < self.extreme_volatility_threshold and
            0 < self.health_score_threshold < 1 and
            self.min_data_points >= 20 and
            0 <= self.macro_weight <= 1 and
            self.vix_low_threshold < self.vix_normal_threshold < self.vix_high_threshold < self.vix_panic_threshold
        )


# 策略配置管理器
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
            # 使用默认配置
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
            price_breakout_threshold=cfg.get('price_breakout_threshold', 1.01),
            volume_burst_multiplier=cfg.get('volume_burst_multiplier', 2.0),
            momentum_5d_threshold=cfg.get('momentum_5d_threshold', 0.03),
            momentum_20d_threshold=cfg.get('momentum_20d_threshold', 0.05),
            min_data_points=cfg.get('min_data_points', 21),
        )
    
    def _parse_accumulation_config(self, strategies_config: Dict) -> AccumulationAccelerationConfig:
        """解析吸筹策略配置"""
        cfg = strategies_config.get('accumulation_acceleration', {})
        return AccumulationAccelerationConfig(
            accumulation_volatility_threshold=cfg.get('accumulation_volatility_threshold', 0.15),
            accumulation_period=cfg.get('accumulation_period', 30),
            acceleration_price_multiplier=cfg.get('acceleration_price_multiplier', 1.015),
            acceleration_volume_ratio=cfg.get('acceleration_volume_ratio', 2.5),
            rsi_low_range_min=cfg.get('rsi_low_range_min', 40.0),
            rsi_low_range_max=cfg.get('rsi_low_range_max', 60.0),
            rsi_breakout_threshold=cfg.get('rsi_breakout_threshold', 65.0),
            rsi_period=cfg.get('rsi_period', 14),
            min_data_points=cfg.get('min_data_points', 30),
        )
    
    def _parse_volatility_config(self, strategies_config: Dict) -> VolatilitySqueezeConfig:
        """解析波动率策略配置"""
        cfg = strategies_config.get('volatility_squeeze', {})
        return VolatilitySqueezeConfig(
            bb_period=cfg.get('bb_period', 20),
            bb_std_dev=cfg.get('bb_std_dev', 2.0),
            squeeze_lookback=cfg.get('squeeze_lookback', 100),
            squeeze_percentile=cfg.get('squeeze_percentile', 0.10),
            breakout_change_threshold=cfg.get('breakout_change_threshold', 0.02),
            volume_multiplier=cfg.get('volume_multiplier', 1.5),
            volume_avg_period=cfg.get('volume_avg_period', 50),
            min_data_points=cfg.get('min_data_points', 100),
        )
    
    def _parse_signal_scorer_config(self, strategies_config: Dict) -> SignalScorerConfig:
        """解析信号评分器配置"""
        cfg = strategies_config.get('signal_scorer', {})
        default_weights = {
            "trend_following": 0.25,
            "momentum_breakout": 0.20,
            "volume_confirmation": 0.15,
            "market_correction": 0.20,
            "sector_strength": 0.20
        }
        weights = cfg.get('weights', default_weights)
        return SignalScorerConfig(
            weights=weights,
            pass_threshold=cfg.get('pass_threshold', 0.7),
            min_data_points=cfg.get('min_data_points', 50),
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
