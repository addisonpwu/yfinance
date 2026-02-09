"""
动态仓位管理模块

提供多种仓位计算方法:
- 等额仓位
- Kelly 公式
- 波动率调整仓位
- 风险平价仓位
- 信心加权仓位

功能:
- 根据信号信心调整仓位
- 考虑波动率风险
- 控制单股和行业集中度
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionConfig:
    """仓位配置"""
    max_position: float = 0.15        # 单只股票最大仓位 (15%)
    max_sector: float = 0.30          # 单个行业最大仓位 (30%)
    max_total: float = 0.80           # 总仓位上限 (80%)
    min_position: float = 0.02        # 最小仓位 (2%)
    kelly_fraction: float = 0.5       # Kelly 分数 (减半)
    risk_per_trade: float = 0.02      # 每笔交易风险 (2%)
    volatility_lookback: int = 20     # 波动率回看期


class PositionSizer:
    """
    动态仓位计算器

    根据多种因素计算最优仓位:
    - 策略信心度
    - 波动率
    - 相关性风险
    - 账户规模
    """

    def __init__(self, config: Optional[PositionConfig] = None):
        """
        初始化仓位计算器

        Args:
            config: 仓位配置
        """
        self.config = config or PositionConfig()
        self.current_positions: Dict[str, float] = {}  # 当前持仓
        self.sector_exposure: Dict[str, float] = {}    # 行业暴露

    def calculate_position(self,
                           confidence: float,
                           price: float,
                           volatility: float = None,
                           account_balance: float = 100000,
                           symbol: str = "",
                           sector: str = "",
                           existing_positions: Dict[str, float] = None,
                           existing_sectors: Dict[str, float] = None) -> Dict:
        """
        计算仓位

        Args:
            confidence: 信号信心度 (0 ~ 1)
            price: 当前价格
            volatility: 波动率 (年化)，如果为None则使用默认值
            account_balance: 账户余额
            symbol: 股票代码
            sector: 行业
            existing_positions: 现有持仓 (可选)
            existing_sectors: 现有行业暴露 (可选)

        Returns:
            {
                "position_size": 仓位金额,
                "position_pct": 仓位比例,
                "shares": 股数,
                "adjusted_confidence": 调整后信心度,
                "risk_info": {...}
            }
        """
        # 更新现有持仓
        if existing_positions:
            self.current_positions = existing_positions.copy()
        if existing_sectors:
            self.sector_exposure = existing_sectors.copy()

        # 1. 基础信心调整
        adjusted_confidence = self._adjust_confidence(confidence, volatility)

        # 2. 波动率调整
        vol_adjustment = self._calculate_volatility_adjustment(volatility)

        # 3. 计算风险调整后的仓位
        risk_adjusted_size = self._calculate_risk_adjusted_size(
            adjusted_confidence, vol_adjustment
        )

        # 4. 账户规模限制
        max_size = account_balance * self.config.max_position
        position_value = min(max_size * risk_adjusted_size, account_balance * self.config.max_position)

        # 5. 行业集中度限制
        if sector:
            sector_limit = account_balance * self.config.max_sector
            current_sector = self.sector_exposure.get(sector, 0)
            available_sector = sector_limit - current_sector
            position_value = min(position_value, available_sector)

        # 6. 总仓位限制
        total_exposure = sum(self.current_positions.values())
        total_limit = account_balance * self.config.max_total
        position_value = min(position_value, total_limit - total_exposure)

        # 7. 最小仓位检查
        min_size = account_balance * self.config.min_position
        if position_value < min_size:
            position_value = 0

        # 8. 计算股数
        shares = int(position_value / price) if price > 0 else 0
        position_pct = position_value / account_balance if account_balance > 0 else 0

        # 更新持仓记录
        if shares > 0 and symbol:
            self.current_positions[symbol] = shares * price
        if sector:
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_value

        return {
            "position_size": round(position_value, 2),
            "position_pct": round(position_pct, 4),
            "shares": shares,
            "adjusted_confidence": round(adjusted_confidence, 3),
            "volatility_adjustment": round(vol_adjustment, 3),
            "risk_info": {
                "confidence": confidence,
                "volatility": volatility,
                "max_position": self.config.max_position,
                "current_sector_exposure": self.sector_exposure.get(sector, 0),
                "total_exposure_after": total_exposure + position_pct
            }
        }

    def _adjust_confidence(self, confidence: float, volatility: float = None) -> float:
        """
        调整信心度

        Args:
            confidence: 原始信心度
            volatility: 波动率

        Returns:
            float: 调整后的信心度
        """
        # 低信心信号不交易
        if confidence < 0.3:
            return 0.0

        # 调整
        adjusted = confidence

        # 高波动率降低信心
        if volatility and volatility > 0.4:
            adjusted *= 0.8
        elif volatility and volatility > 0.3:
            adjusted *= 0.9

        return min(adjusted, 1.0)

    def _calculate_volatility_adjustment(self, volatility: float = None) -> float:
        """
        计算波动率调整因子

        Args:
            volatility: 年化波动率

        Returns:
            float: 调整因子 (0 ~ 1)
        """
        if volatility is None:
            return 1.0

        # 低波动 = 高仓位，高波动 = 低仓位
        if volatility < 0.15:
            return 1.2  # 增强
        elif volatility < 0.25:
            return 1.0  # 正常
        elif volatility < 0.40:
            return 0.7  # 减少
        else:
            return 0.5  # 大幅减少

        return 1.0

    def _calculate_risk_adjusted_size(self, confidence: float,
                                       vol_adjustment: float) -> float:
        """
        计算风险调整后的仓位比例

        Args:
            confidence: 调整后的信心度
            vol_adjustment: 波动率调整因子

        Returns:
            float: 仓位比例 (0 ~ 1)
        """
        # 基础仓位 = 信心度 × 波动率调整
        base_size = confidence * vol_adjustment

        # 应用每笔交易风险限制
        risk_adjusted = min(base_size, self.config.risk_per_trade / (vol_adjustment + 0.1))

        return min(risk_adjusted, 1.0)

    def kelly_position(self,
                       win_rate: float,
                       avg_win: float,
                       avg_loss: float,
                       account_balance: float = 100000) -> float:
        """
        Kelly 公式计算仓位

        Kelly % = W - (1-W) / (AvgWin / |AvgLoss|)

        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损 (正数)
            account_balance: 账户余额

        Returns:
            float: Kelly 仓位比例
        """
        # Kelly 公式
        if avg_loss <= 0:
            return 0.0

        wl_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / wl_ratio)

        # 减半 Kelly 以降低风险
        kelly = kelly * self.config.kelly_fraction

        # 限制在合理范围
        kelly = max(0, min(kelly, 0.5))

        return kelly * account_balance

    def calculate_kelly_size(self,
                              returns: pd.Series,
                              account_balance: float = 100000) -> Tuple[float, Dict]:
        """
        从历史收益率计算 Kelly 仓位

        Args:
            returns: 收益率序列
            account_balance: 账户余额

        Returns:
            (kelly_pct, analysis): Kelly 比例和分析
        """
        if len(returns) < 20:
            return 0.1, {"reason": "数据不足"}

        # 计算胜率
        win_rate = (returns > 0).mean()

        # 计算盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.001
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0.001

        # Kelly 公式
        wl_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / wl_ratio)

        # 减半
        kelly = kelly * self.config.kelly_fraction

        # 限制
        kelly = max(0, min(kelly, 0.25))

        analysis = {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "wl_ratio": wl_ratio,
            "kelly_full": win_rate - ((1 - win_rate) / wl_ratio),
            "kelly_fraction": self.config.kelly_fraction,
            "kelly_recommended": kelly,
            "risk_warning": "高 Kelly 风险" if kelly > 0.2 else "风险适中" if kelly > 0.1 else "低 Kelly 风险"
        }

        return kelly, analysis

    def volatility_parity_position(self,
                                   volatilities: Dict[str, float],
                                   account_balance: float = 100000,
                                   total_target_risk: float = 0.15) -> Dict[str, float]:
        """
        风险平价仓位分配

        根据波动率倒数分配仓位，使各资产风险贡献相等

        Args:
            volatilities: 各资产波动率字典 {symbol: vol}
            account_balance: 账户余额
            total_target_risk: 目标总风险

        Returns:
            Dict: 各资产仓位比例
        """
        if not volatilities:
            return {}

        # 转换为数组
        symbols = list(volatilities.keys())
        vols = np.array([volatilities[s] for s in symbols], dtype=float)

        # 避免除零
        vols = np.maximum(vols, 0.01)

        # 波动率倒数权重
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()

        # 应用总风险限制
        avg_vol = vols.mean()
        risk_adjusted = weights * (total_target_risk / avg_vol)

        # 归一化
        risk_adjusted = risk_adjusted / risk_adjusted.sum()

        return {s: round(risk_adjusted[i], 4) for i, s in enumerate(symbols)}

    def equal_weight_position(self,
                              symbols: List[str],
                              account_balance: float = 100000,
                              max_position: float = None) -> Dict[str, float]:
        """
        等额仓位分配

        Args:
            symbols: 股票列表
            account_balance: 账户余额
            max_position: 最大单仓位

        Returns:
            Dict: 各资产仓位比例
        """
        max_pct = max_position or self.config.max_position
        n = len(symbols)
        if n == 0:
            return {}

        # 计算等额
        equal_pct = 1.0 / n

        # 检查是否超过最大限制
        if equal_pct > max_pct:
            # 重新计算
            equal_pct = max_pct

        return {s: equal_pct for s in symbols}

    def confidence_weighted_position(self,
                                     confidences: Dict[str, float],
                                     account_balance: float = 100000) -> Dict[str, float]:
        """
        信心加权仓位分配

        Args:
            confidences: 各资产信心度 {symbol: confidence}
            account_balance: 账户余额

        Returns:
            Dict: 各资产仓位比例
        """
        if not confidences:
            return {}

        # 归一化信心度
        total_conf = sum(confidences.values())
        if total_conf == 0:
            return {s: 0.0 for s in confidences}

        raw_weights = {s: c / total_conf for s, c in confidences.items()}

        # 应用最大仓位限制
        max_pct = self.config.max_position
        adjusted_weights = {}

        for s, w in raw_weights.items():
            if w > max_pct:
                adjusted_weights[s] = max_pct
            else:
                adjusted_weights[s] = w

        # 重新归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {s: w / total * min(total, self.config.max_total)
                               for s, w in adjusted_weights.items()}

        return adjusted_weights

    def get_risk_summary(self, positions: Dict[str, float],
                          account_balance: float = 100000) -> Dict:
        """
        获取风险摘要

        Args:
            positions: 当前持仓 {symbol: value}
            account_balance: 账户余额

        Returns:
            Dict: 风险摘要
        """
        total_value = sum(positions.values())
        total_pct = total_value / account_balance if account_balance > 0 else 0

        # 集中度指标
        if positions:
            largest_position = max(positions.values())
            concentration = largest_position / total_value if total_value > 0 else 0
        else:
            concentration = 0

        return {
            "total_exposure_pct": round(total_pct, 4),
            "total_exposure_value": round(total_value, 2),
            "num_positions": len(positions),
            "largest_position_pct": round(concentration, 4),
            "cash_remaining": round(account_balance - total_value, 2),
            "cash_pct": round(1 - total_pct, 4),
            "risk_level": self._assess_risk_level(total_pct, concentration, len(positions))
        }

    def _assess_risk_level(self, total_pct: float, concentration: float,
                           num_positions: int) -> str:
        """评估风险等级"""
        score = 0

        # 仓位评分
        if total_pct < 0.5:
            score += 2
        elif total_pct < 0.7:
            score += 1
        else:
            score += 0

        # 集中度评分
        if concentration < 0.2:
            score += 2
        elif concentration < 0.4:
            score += 1
        else:
            score += 0

        # 分散度评分
        if num_positions >= 10:
            score += 2
        elif num_positions >= 5:
            score += 1
        else:
            score += 0

        if score >= 6:
            return "低"
        elif score >= 4:
            return "中"
        else:
            return "高"

    def reset(self) -> None:
        """重置计算器状态"""
        self.current_positions = {}
        self.sector_exposure = {}


# 便捷函数
def calc_position(confidence: float, price: float, volatility: float = None,
                  account_balance: float = 100000, **kwargs) -> Dict:
    """
    便捷函数：快速计算仓位

    Args:
        confidence: 信号信心度
        price: 当前价格
        volatility: 波动率
        account_balance: 账户余额
        **kwargs: 其他参数

    Returns:
        Dict: 仓位信息
    """
    sizer = PositionSizer()
    return sizer.calculate_position(
        confidence=confidence,
        price=price,
        volatility=volatility,
        account_balance=account_balance,
        **kwargs
    )


def kelly_sizer(returns: pd.Series, account_balance: float = 100000) -> Tuple[float, Dict]:
    """
    便捷函数：Kelly 仓位计算

    Args:
        returns: 收益率序列
        account_balance: 账户余额

    Returns:
        (kelly_pct, analysis)
    """
    sizer = PositionSizer()
    return sizer.calculate_kelly_size(returns, account_balance)


def risk_summary(positions: Dict[str, float],
                  account_balance: float = 100000) -> Dict:
    """
    便捷函数：获取风险摘要

    Args:
        positions: 持仓字典
        account_balance: 账户余额

    Returns:
        Dict: 风险摘要
    """
    sizer = PositionSizer()
    return sizer.get_risk_summary(positions, account_balance)
