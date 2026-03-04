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
    # 启动捕捉策略参数 (整合版)
    LAUNCH_CAPTURE_PERIOD, LAUNCH_MIN_SCORE, LAUNCH_STRONG_SIGNAL_SCORE, LAUNCH_MIN_DATA_POINTS,
    LAUNCH_MA_CONVERGENCE_THRESHOLD, LAUNCH_MA_SPREAD_EXTREME, LAUNCH_MA_SPREAD_STRONG,
    LAUNCH_MA_SPREAD_MODERATE, LAUNCH_MA_SPREAD_WEAK,
    LAUNCH_PATTERN_PERIOD, LAUNCH_CHANNEL_CONVERGENCE, LAUNCH_TRIANGLE_THRESHOLD,
    LAUNCH_PIT_DEPTH_MIN, LAUNCH_PIT_DEPTH_MAX,
    LAUNCH_TTM_BB_PERIOD, LAUNCH_TTM_BB_STD_DEV, LAUNCH_TTM_KC_MULTIPLIER, LAUNCH_TTM_ATR_PERIOD,
    LAUNCH_CMF_PERIOD, LAUNCH_CMF_STRONG_THRESHOLD, LAUNCH_CMF_WEAK_THRESHOLD,
    LAUNCH_VOL_RATIO_THRESHOLD, LAUNCH_PRICE_POSITION_THRESHOLD, LAUNCH_GROUND_VOLUME_RATIO,
    LAUNCH_BETA_LOW_THRESHOLD, LAUNCH_BETA_HIGH_THRESHOLD, LAUNCH_RS_LOOKBACK, LAUNCH_RS_STRONG,
    LAUNCH_RSI_NEUTRAL_LOW, LAUNCH_RSI_NEUTRAL_HIGH, LAUNCH_MACD_THRESHOLD,
    LAUNCH_VOLUME_EXPANSION_MIN, LAUNCH_VOLUME_EXPANSION_MAX, LAUNCH_VOLUME_CONTRACTION,
    LAUNCH_MA_WEIGHT, LAUNCH_PATTERN_WEIGHT, LAUNCH_VOLATILITY_WEIGHT, LAUNCH_MONEY_FLOW_WEIGHT,
    LAUNCH_RESILIENCE_WEIGHT, LAUNCH_TECHNICAL_WEIGHT, LAUNCH_VOLUME_WEIGHT,
    # OBV 底背离 + BOLL 超卖策略参数
    OBV_BOLL_LLV_PERIOD, OBV_BOLL_OBV_LOOKBACK, OBV_BOLL_VOLUME_RATIO_MIN,
    OBV_BOLL_VOLUME_RATIO_MAX, OBV_BOLL_MA_LONG_PERIOD, OBV_BOLL_MIN_DATA_POINTS,
    OBV_BOLL_CONFIDENCE,
)


@dataclass
class LaunchCaptureConfig:
    """
    启动捕捉策略配置 (整合版)

    整合启动前兆策略和主力建仓策略的核心优势，
    专门捕捉股票即将启动的综合策略。
    """
    # 基础参数
    period: int = LAUNCH_CAPTURE_PERIOD
    min_score: float = LAUNCH_MIN_SCORE
    strong_signal_score: float = LAUNCH_STRONG_SIGNAL_SCORE
    min_data_points: int = LAUNCH_MIN_DATA_POINTS

    # 均线粘合参数
    ma_convergence_threshold: float = LAUNCH_MA_CONVERGENCE_THRESHOLD
    ma_spread_extreme: float = LAUNCH_MA_SPREAD_EXTREME
    ma_spread_strong: float = LAUNCH_MA_SPREAD_STRONG
    ma_spread_moderate: float = LAUNCH_MA_SPREAD_MODERATE
    ma_spread_weak: float = LAUNCH_MA_SPREAD_WEAK

    # 形态识别参数
    pattern_period: int = LAUNCH_PATTERN_PERIOD
    channel_convergence: float = LAUNCH_CHANNEL_CONVERGENCE
    triangle_threshold: float = LAUNCH_TRIANGLE_THRESHOLD
    pit_depth_min: float = LAUNCH_PIT_DEPTH_MIN
    pit_depth_max: float = LAUNCH_PIT_DEPTH_MAX

    # TTM Squeeze参数
    ttm_bb_period: int = LAUNCH_TTM_BB_PERIOD
    ttm_bb_std_dev: float = LAUNCH_TTM_BB_STD_DEV
    ttm_kc_multiplier: float = LAUNCH_TTM_KC_MULTIPLIER
    ttm_atr_period: int = LAUNCH_TTM_ATR_PERIOD

    # 资金信号参数
    cmf_period: int = LAUNCH_CMF_PERIOD
    cmf_strong_threshold: float = LAUNCH_CMF_STRONG_THRESHOLD
    cmf_weak_threshold: float = LAUNCH_CMF_WEAK_THRESHOLD
    vol_ratio_threshold: float = LAUNCH_VOL_RATIO_THRESHOLD
    price_position_threshold: float = LAUNCH_PRICE_POSITION_THRESHOLD
    ground_volume_ratio: float = LAUNCH_GROUND_VOLUME_RATIO

    # 抗跌特征参数
    beta_low_threshold: float = LAUNCH_BETA_LOW_THRESHOLD
    beta_high_threshold: float = LAUNCH_BETA_HIGH_THRESHOLD
    rs_lookback: int = LAUNCH_RS_LOOKBACK
    rs_strong: float = LAUNCH_RS_STRONG

    # 技术指标参数
    rsi_neutral_low: float = LAUNCH_RSI_NEUTRAL_LOW
    rsi_neutral_high: float = LAUNCH_RSI_NEUTRAL_HIGH
    macd_threshold: float = LAUNCH_MACD_THRESHOLD

    # 成交量参数
    volume_expansion_min: float = LAUNCH_VOLUME_EXPANSION_MIN
    volume_expansion_max: float = LAUNCH_VOLUME_EXPANSION_MAX
    volume_contraction: float = LAUNCH_VOLUME_CONTRACTION

    # 评分权重
    ma_weight: float = LAUNCH_MA_WEIGHT
    pattern_weight: float = LAUNCH_PATTERN_WEIGHT
    volatility_weight: float = LAUNCH_VOLATILITY_WEIGHT
    money_flow_weight: float = LAUNCH_MONEY_FLOW_WEIGHT
    resilience_weight: float = LAUNCH_RESILIENCE_WEIGHT
    technical_weight: float = LAUNCH_TECHNICAL_WEIGHT
    volume_weight: float = LAUNCH_VOLUME_WEIGHT

    def validate(self) -> bool:
        """验证配置有效性"""
        return (
            self.period >= 20 and
            0 < self.ma_convergence_threshold < 0.1 and
            0 < self.min_score < 100 and
            self.cmf_period > 0 and
            self.ttm_bb_period > 0 and
            sum([
                self.ma_weight, self.pattern_weight, self.volatility_weight,
                self.money_flow_weight, self.resilience_weight,
                self.technical_weight, self.volume_weight
            ]) > 0.99  # 权重总和应接近1
        )


@dataclass
class OBVBollConfig:
    """OBV 底背离 + BOLL 超卖策略配置"""
    llv_period: int = OBV_BOLL_LLV_PERIOD
    obv_lookback: int = OBV_BOLL_OBV_LOOKBACK
    volume_ratio_min: float = OBV_BOLL_VOLUME_RATIO_MIN
    volume_ratio_max: float = OBV_BOLL_VOLUME_RATIO_MAX
    ma_long_period: int = OBV_BOLL_MA_LONG_PERIOD
    min_data_points: int = OBV_BOLL_MIN_DATA_POINTS
    confidence: float = OBV_BOLL_CONFIDENCE

    def validate(self) -> bool:
        """验证配置有效性"""
        return (
            self.llv_period > 0 and
            self.obv_lookback > 0 and
            0 < self.volume_ratio_min < self.volume_ratio_max and
            self.ma_long_period > 0 and
            self.min_data_points >= self.ma_long_period + self.obv_lookback and
            0 < self.confidence <= 1.0
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
                'launch_capture': self._parse_launch_capture_config(strategies_config),
                'obv_boll_divergence': self._parse_obv_boll_config(strategies_config),
            }

            return self._configs

        except FileNotFoundError:
            self._configs = {
                'launch_capture': LaunchCaptureConfig(),
            }
            return self._configs

    def _parse_launch_capture_config(self, strategies_config: Dict) -> LaunchCaptureConfig:
        """解析启动捕捉策略配置"""
        cfg = strategies_config.get('launch_capture', {})
        return LaunchCaptureConfig(
            period=cfg.get('period', LAUNCH_CAPTURE_PERIOD),
            min_score=cfg.get('min_score', LAUNCH_MIN_SCORE),
            strong_signal_score=cfg.get('strong_signal_score', LAUNCH_STRONG_SIGNAL_SCORE),
            min_data_points=cfg.get('min_data_points', LAUNCH_MIN_DATA_POINTS),
            ma_convergence_threshold=cfg.get('ma_convergence_threshold', LAUNCH_MA_CONVERGENCE_THRESHOLD),
            pattern_period=cfg.get('pattern_period', LAUNCH_PATTERN_PERIOD),
            cmf_period=cfg.get('cmf_period', LAUNCH_CMF_PERIOD),
            beta_low_threshold=cfg.get('beta_low_threshold', LAUNCH_BETA_LOW_THRESHOLD),
        )

    def _parse_obv_boll_config(self, strategies_config: Dict) -> OBVBollConfig:
        """解析 OBV 底背离 + BOLL 超卖策略配置"""
        cfg = strategies_config.get('obv_boll_divergence', {})
        return OBVBollConfig(
            llv_period=cfg.get('llv_period', OBV_BOLL_LLV_PERIOD),
            obv_lookback=cfg.get('obv_lookback', OBV_BOLL_OBV_LOOKBACK),
            volume_ratio_min=cfg.get('volume_ratio_min', OBV_BOLL_VOLUME_RATIO_MIN),
            volume_ratio_max=cfg.get('volume_ratio_max', OBV_BOLL_VOLUME_RATIO_MAX),
            ma_long_period=cfg.get('ma_long_period', OBV_BOLL_MA_LONG_PERIOD),
            min_data_points=cfg.get('min_data_points', OBV_BOLL_MIN_DATA_POINTS),
            confidence=cfg.get('confidence', OBV_BOLL_CONFIDENCE),
        )

    def get_config(self, strategy_name: str):
        """获取指定策略的配置"""
        if self._configs is None:
            self.load_from_file()
        return self._configs.get(strategy_name)


# 全局实例
strategy_config_manager = StrategyConfigManager()