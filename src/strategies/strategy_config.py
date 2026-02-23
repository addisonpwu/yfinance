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
    # 技术指标
    RSI_PERIOD, BB_PERIOD, BB_STD_DEV,
    # 启动前兆策略 v2.0 (增强版)
    BREAKOUT_MA_CONVERGENCE_THRESHOLD, BREAKOUT_MA_CONVERGENCE_PERIOD,
    BREAKOUT_PRICE_NEAR_HIGH_THRESHOLD, BREAKOUT_CONSOLIDATION_PERIOD,
    BREAKOUT_VOLUME_CONTRACTION_THRESHOLD, BREAKOUT_VOLUME_EXPANSION_MIN,
    BREAKOUT_VOLUME_EXPANSION_MAX, BREAKOUT_RSI_NEUTRAL_LOW, BREAKOUT_RSI_NEUTRAL_HIGH,
    BREAKOUT_MACD_THRESHOLD, BREAKOUT_BB_SQUEEZE_THRESHOLD,
    BREAKOUT_MIN_SCORE, BREAKOUT_STRONG_SIGNAL_SCORE, BREAKOUT_MIN_DATA_POINTS,
    # TTM Squeeze 参数
    BREAKOUT_TTM_BB_PERIOD, BREAKOUT_TTM_BB_STD_DEV,
    BREAKOUT_TTM_KC_MULTIPLIER, BREAKOUT_TTM_ATR_PERIOD,
    # 形态识别参数
    BREAKOUT_PATTERN_PERIOD, BREAKOUT_CHANNEL_CONVERGENCE_THRESHOLD,
    BREAKOUT_TRIANGLE_THRESHOLD, BREAKOUT_FLAG_THRESHOLD,
    # 相对强度参数
    BREAKOUT_RS_STRONG_THRESHOLD, BREAKOUT_RS_WEAK_THRESHOLD, BREAKOUT_RS_LOOKBACK,
    # 评分权重
    BREAKOUT_MA_WEIGHT, BREAKOUT_PATTERN_WEIGHT, BREAKOUT_VOLUME_WEIGHT,
    BREAKOUT_VOLATILITY_WEIGHT, BREAKOUT_TECHNICAL_WEIGHT, BREAKOUT_RS_WEIGHT,
    # 主力建仓策略 v2.0 (增强版)
    SMART_MONEY_ACCUMULATION_PERIOD, SMART_MONEY_PRICE_STABILITY_THRESHOLD,
    SMART_MONEY_RELATIVE_STRENGTH_THRESHOLD, SMART_MONEY_VOLUME_PATTERN_THRESHOLD,
    SMART_MONEY_INTERMITTENT_VOLUME_RATIO, SMART_MONEY_TURNOVER_RATE_LOW,
    SMART_MONEY_TURNOVER_RATE_HIGH, SMART_MONEY_MIN_SCORE, SMART_MONEY_MIN_DATA_POINTS,
    # Beta 分析参数
    SMART_MONEY_BETA_LOW_THRESHOLD, SMART_MONEY_BETA_HIGH_THRESHOLD,
    SMART_MONEY_BETA_TREND_THRESHOLD,
    # 建仓尾声参数
    SMART_MONEY_VOL_RATIO_THRESHOLD, SMART_MONEY_PRICE_POSITION_THRESHOLD,
    SMART_MONEY_GROUND_VOLUME_RATIO,
    # CMF 参数
    SMART_MONEY_CMF_PERIOD, SMART_MONEY_CMF_STRONG_THRESHOLD,
    SMART_MONEY_CMF_WEAK_THRESHOLD,
    # 挖坑形态参数
    SMART_MONEY_PIT_DEPTH_THRESHOLD, SMART_MONEY_PIT_VOL_RATIO_MIN,
    SMART_MONEY_PIT_VOL_RATIO_MAX,
    # 评分权重
    SMART_MONEY_RESILIENCE_WEIGHT, SMART_MONEY_ENDING_WEIGHT,
    SMART_MONEY_VOLUME_WEIGHT, SMART_MONEY_PATTERN_WEIGHT, SMART_MONEY_FLOW_WEIGHT,
)


@dataclass
class BreakoutSetupConfig:
    """启动前兆策略配置 v2.0 (增强版)"""
    # 基础参数
    ma_convergence_threshold: float = BREAKOUT_MA_CONVERGENCE_THRESHOLD
    ma_convergence_period: int = BREAKOUT_MA_CONVERGENCE_PERIOD
    price_near_high_threshold: float = BREAKOUT_PRICE_NEAR_HIGH_THRESHOLD
    consolidation_period: int = BREAKOUT_CONSOLIDATION_PERIOD
    volume_contraction_threshold: float = BREAKOUT_VOLUME_CONTRACTION_THRESHOLD
    volume_expansion_min: float = BREAKOUT_VOLUME_EXPANSION_MIN
    volume_expansion_max: float = BREAKOUT_VOLUME_EXPANSION_MAX
    rsi_neutral_low: float = BREAKOUT_RSI_NEUTRAL_LOW
    rsi_neutral_high: float = BREAKOUT_RSI_NEUTRAL_HIGH
    macd_threshold: float = BREAKOUT_MACD_THRESHOLD
    bb_squeeze_threshold: float = BREAKOUT_BB_SQUEEZE_THRESHOLD
    min_score: float = BREAKOUT_MIN_SCORE
    strong_signal_score: float = BREAKOUT_STRONG_SIGNAL_SCORE
    min_data_points: int = BREAKOUT_MIN_DATA_POINTS
    
    # TTM Squeeze 参数
    ttm_bb_period: int = BREAKOUT_TTM_BB_PERIOD
    ttm_bb_std_dev: float = BREAKOUT_TTM_BB_STD_DEV
    ttm_kc_multiplier: float = BREAKOUT_TTM_KC_MULTIPLIER
    ttm_atr_period: int = BREAKOUT_TTM_ATR_PERIOD
    
    # 形态识别参数
    pattern_period: int = BREAKOUT_PATTERN_PERIOD
    channel_convergence_threshold: float = BREAKOUT_CHANNEL_CONVERGENCE_THRESHOLD
    triangle_threshold: float = BREAKOUT_TRIANGLE_THRESHOLD
    flag_threshold: float = BREAKOUT_FLAG_THRESHOLD
    
    # 相对强度参数
    rs_strong_threshold: float = BREAKOUT_RS_STRONG_THRESHOLD
    rs_weak_threshold: float = BREAKOUT_RS_WEAK_THRESHOLD
    rs_lookback: int = BREAKOUT_RS_LOOKBACK
    
    # 评分权重
    ma_weight: float = BREAKOUT_MA_WEIGHT
    pattern_weight: float = BREAKOUT_PATTERN_WEIGHT
    volume_weight: float = BREAKOUT_VOLUME_WEIGHT
    volatility_weight: float = BREAKOUT_VOLATILITY_WEIGHT
    technical_weight: float = BREAKOUT_TECHNICAL_WEIGHT
    rs_weight: float = BREAKOUT_RS_WEIGHT
    
    def validate(self) -> bool:
        return (
            0 < self.ma_convergence_threshold < 0.1 and
            0 < self.price_near_high_threshold <= 1 and
            0 < self.volume_contraction_threshold < 1 and
            0 < self.min_score < 100 and
            self.ttm_bb_period > 0 and
            self.ttm_atr_period > 0 and
            self.pattern_period > 0
        )


@dataclass
class SmartMoneyAccumulationConfig:
    """主力建仓策略配置 v2.0 (增强版)"""
    # 基础参数
    accumulation_period: int = SMART_MONEY_ACCUMULATION_PERIOD
    price_stability_threshold: float = SMART_MONEY_PRICE_STABILITY_THRESHOLD
    relative_strength_threshold: float = SMART_MONEY_RELATIVE_STRENGTH_THRESHOLD
    volume_pattern_threshold: float = SMART_MONEY_VOLUME_PATTERN_THRESHOLD
    intermittent_volume_ratio: float = SMART_MONEY_INTERMITTENT_VOLUME_RATIO
    turnover_rate_low: float = SMART_MONEY_TURNOVER_RATE_LOW
    turnover_rate_high: float = SMART_MONEY_TURNOVER_RATE_HIGH
    min_score: float = SMART_MONEY_MIN_SCORE
    min_data_points: int = SMART_MONEY_MIN_DATA_POINTS
    
    # Beta 分析参数
    beta_low_threshold: float = SMART_MONEY_BETA_LOW_THRESHOLD
    beta_high_threshold: float = SMART_MONEY_BETA_HIGH_THRESHOLD
    beta_trend_threshold: float = SMART_MONEY_BETA_TREND_THRESHOLD
    
    # 建仓尾声参数
    vol_ratio_threshold: float = SMART_MONEY_VOL_RATIO_THRESHOLD
    price_position_threshold: float = SMART_MONEY_PRICE_POSITION_THRESHOLD
    ground_volume_ratio: float = SMART_MONEY_GROUND_VOLUME_RATIO
    
    # CMF 参数
    cmf_period: int = SMART_MONEY_CMF_PERIOD
    cmf_strong_threshold: float = SMART_MONEY_CMF_STRONG_THRESHOLD
    cmf_weak_threshold: float = SMART_MONEY_CMF_WEAK_THRESHOLD
    
    # 挖坑形态参数
    pit_depth_threshold: float = SMART_MONEY_PIT_DEPTH_THRESHOLD
    pit_vol_ratio_min: float = SMART_MONEY_PIT_VOL_RATIO_MIN
    pit_vol_ratio_max: float = SMART_MONEY_PIT_VOL_RATIO_MAX
    
    # 评分权重
    resilience_weight: float = SMART_MONEY_RESILIENCE_WEIGHT
    ending_weight: float = SMART_MONEY_ENDING_WEIGHT
    volume_weight: float = SMART_MONEY_VOLUME_WEIGHT
    pattern_weight: float = SMART_MONEY_PATTERN_WEIGHT
    flow_weight: float = SMART_MONEY_FLOW_WEIGHT
    
    def validate(self) -> bool:
        return (
            self.accumulation_period >= 20 and
            0 < self.price_stability_threshold < 0.2 and
            0 < self.min_score < 100 and
            self.beta_low_threshold > 0 and
            self.cmf_period > 0
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
                'breakout_setup': self._parse_breakout_setup_config(strategies_config),
                'smart_money_accumulation': self._parse_smart_money_config(strategies_config),
            }
            
            return self._configs
            
        except FileNotFoundError:
            self._configs = {
                'breakout_setup': BreakoutSetupConfig(),
                'smart_money_accumulation': SmartMoneyAccumulationConfig(),
            }
            return self._configs
    
    def _parse_breakout_setup_config(self, strategies_config: Dict) -> BreakoutSetupConfig:
        """解析启动前兆策略配置"""
        cfg = strategies_config.get('breakout_setup', {})
        return BreakoutSetupConfig(
            ma_convergence_threshold=cfg.get('ma_convergence_threshold', BREAKOUT_MA_CONVERGENCE_THRESHOLD),
            ma_convergence_period=cfg.get('ma_convergence_period', BREAKOUT_MA_CONVERGENCE_PERIOD),
            price_near_high_threshold=cfg.get('price_near_high_threshold', BREAKOUT_PRICE_NEAR_HIGH_THRESHOLD),
            consolidation_period=cfg.get('consolidation_period', BREAKOUT_CONSOLIDATION_PERIOD),
            volume_contraction_threshold=cfg.get('volume_contraction_threshold', BREAKOUT_VOLUME_CONTRACTION_THRESHOLD),
            volume_expansion_min=cfg.get('volume_expansion_min', BREAKOUT_VOLUME_EXPANSION_MIN),
            volume_expansion_max=cfg.get('volume_expansion_max', BREAKOUT_VOLUME_EXPANSION_MAX),
            rsi_neutral_low=cfg.get('rsi_neutral_low', BREAKOUT_RSI_NEUTRAL_LOW),
            rsi_neutral_high=cfg.get('rsi_neutral_high', BREAKOUT_RSI_NEUTRAL_HIGH),
            macd_threshold=cfg.get('macd_threshold', BREAKOUT_MACD_THRESHOLD),
            bb_squeeze_threshold=cfg.get('bb_squeeze_threshold', BREAKOUT_BB_SQUEEZE_THRESHOLD),
            min_score=cfg.get('min_score', BREAKOUT_MIN_SCORE),
            strong_signal_score=cfg.get('strong_signal_score', BREAKOUT_STRONG_SIGNAL_SCORE),
            min_data_points=cfg.get('min_data_points', BREAKOUT_MIN_DATA_POINTS),
        )
    
    def _parse_smart_money_config(self, strategies_config: Dict) -> SmartMoneyAccumulationConfig:
        """解析主力建仓策略配置"""
        cfg = strategies_config.get('smart_money_accumulation', {})
        return SmartMoneyAccumulationConfig(
            accumulation_period=cfg.get('accumulation_period', SMART_MONEY_ACCUMULATION_PERIOD),
            price_stability_threshold=cfg.get('price_stability_threshold', SMART_MONEY_PRICE_STABILITY_THRESHOLD),
            relative_strength_threshold=cfg.get('relative_strength_threshold', SMART_MONEY_RELATIVE_STRENGTH_THRESHOLD),
            volume_pattern_threshold=cfg.get('volume_pattern_threshold', SMART_MONEY_VOLUME_PATTERN_THRESHOLD),
            intermittent_volume_ratio=cfg.get('intermittent_volume_ratio', SMART_MONEY_INTERMITTENT_VOLUME_RATIO),
            turnover_rate_low=cfg.get('turnover_rate_low', SMART_MONEY_TURNOVER_RATE_LOW),
            turnover_rate_high=cfg.get('turnover_rate_high', SMART_MONEY_TURNOVER_RATE_HIGH),
            min_score=cfg.get('min_score', SMART_MONEY_MIN_SCORE),
            min_data_points=cfg.get('min_data_points', SMART_MONEY_MIN_DATA_POINTS),
        )
    
    def get_config(self, strategy_name: str):
        """获取指定策略的配置"""
        if self._configs is None:
            self.load_from_file()
        return self._configs.get(strategy_name)


# 全局实例
strategy_config_manager = StrategyConfigManager()
