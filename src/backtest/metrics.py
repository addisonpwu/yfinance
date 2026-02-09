"""
回测指标计算模块

计算各种回测绩效指标:
- 收益率指标 (总收益、年化收益)
- 风险指标 (波动率、最大回撤)
- 风险调整收益 (夏普比率、卡玛比率、索提诺比率)
- 交易统计 (胜率、盈亏比、交易次数)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class BacktestMetrics:
    """
    回测指标计算器
    """

    def __init__(self, equity_curve: pd.Series, returns: pd.Series = None,
                 risk_free_rate: float = 0.02):
        """
        初始化指标计算器

        Args:
            equity_curve: 资金曲线 (净值序列)
            returns: 收益率序列，如果为None则自动计算
            risk_free_rate: 年化无风险利率
        """
        self.equity_curve = equity_curve
        self.returns = returns if returns is not None else equity_curve.pct_change(fill_method=None).dropna()
        self.risk_free_rate = risk_free_rate

        # 预计算基础指标
        self._compute_basics()

    def _compute_basics(self) -> None:
        """预计算基础指标"""
        self.total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        self.annualized_return = self._calculate_annualized_return()
        self.annualized_volatility = self._calculate_volatility()
        self.max_drawdown = self._calculate_max_drawdown()
        self.cumulative_returns = self._calculate_cumulative_returns()

    def calculate_all(self) -> Dict:
        """
        计算所有指标

        Returns:
            Dict: 包含所有指标的字典
        """
        return {
            # 收益率指标
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "monthly_returns": self._calculate_monthly_returns(),

            # 风险指标
            "annualized_volatility": self.annualized_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self._calculate_max_drawdown_duration(),
            "drawdown_series": self._get_drawdown_series(),

            # 风险调整收益
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "sortino_ratio": self._calculate_sortino_ratio(),
            "calmar_ratio": self._calculate_calmar_ratio(),
            "information_ratio": self._calculate_information_ratio(),

            # 交易统计
            "win_rate": self._calculate_win_rate(),
            "profit_factor": self._calculate_profit_factor(),
            "avg_win": self._calculate_avg_win(),
            "avg_loss": self._calculate_avg_loss(),
            "win_loss_ratio": self._calculate_win_loss_ratio(),
            "expectancy": self._calculate_expectancy(),
            "total_trades": self._estimate_trade_count(),

            # 进阶指标
            "skewness": self._calculate_skewness(),
            "kurtosis": self._calculate_kurtosis(),
            "tail_ratio": self._calculate_tail_ratio(),
            "common_sense_ratio": self._calculate_common_sense_ratio(),

            # 滚动指标
            "rolling_sharpe": self._calculate_rolling_sharpe(),
            "rolling_max_drawdown": self._calculate_rolling_max_drawdown(),
        }

    # ========== 收益率指标 ==========

    def _calculate_annualized_return(self) -> float:
        """计算年化收益率"""
        if len(self.equity_curve) < 2:
            return 0.0

        # 计算交易天数
        try:
            days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        except (TypeError, AttributeError):
            # 如果没有日期索引，假设每日数据
            days = len(self.equity_curve)

        if days <= 0:
            return 0.0

        years = days / 365.0
        total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]

        if total_return <= 0:
            return -1.0

        return total_return ** (1 / years) - 1

    def _calculate_monthly_returns(self) -> pd.Series:
        """计算月度收益率"""
        if isinstance(self.equity_curve.index, pd.DatetimeIndex):
            monthly = self.equity_curve.resample('M').last()
            return monthly.pct_change(fill_method=None).dropna()
        else:
            # 简化处理：每20个数据点作为一个"月"
            monthly = self.equity_curve.iloc[::20]
            return monthly.pct_change(fill_method=None).dropna()

    # ========== 风险指标 ==========

    def _calculate_volatility(self) -> float:
        """计算年化波动率"""
        if len(self.returns) < 2:
            return 0.0

        daily_vol = self.returns.std()
        return daily_vol * np.sqrt(252)

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        equity = self.equity_curve
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown.min()

    def _calculate_max_drawdown_duration(self) -> int:
        """计算最大回撤持续时间 (天数)"""
        equity = self.equity_curve
        cummax = equity.cummax()

        # 找到回撤序列
        is_drawdown = equity < cummax

        if not is_drawdown.any():
            return 0

        # 计算连续回撤天数
        drawdown_groups = is_drawdown.astype(int).cumsum()
        drawdown_duration = drawdown_groups.map(
            lambda x: (drawdown_groups == x).sum() if x > 0 else 0
        )

        return int(drawdown_duration.max())

    def _get_drawdown_series(self) -> pd.Series:
        """获取回撤序列"""
        equity = self.equity_curve
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown

    # ========== 风险调整收益指标 ==========

    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if self.annualized_volatility == 0:
            return 0.0

        excess_return = self.annualized_return - self.risk_free_rate
        return excess_return / self.annualized_volatility

    def _calculate_sortino_ratio(self) -> float:
        """计算索提诺比率 (只用下行波动率)"""
        if len(self.returns) < 2:
            return 0.0

        # 下行波动率
        negative_returns = self.returns[self.returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252)

        if downside_vol == 0:
            return 0.0

        excess_return = self.annualized_return - self.risk_free_rate
        return excess_return / downside_vol

    def _calculate_calmar_ratio(self) -> float:
        """计算卡玛比率 (年化收益 / 最大回撤)"""
        if self.max_drawdown >= 0:
            return float('inf') if self.annualized_return > 0 else 0.0

        return self.annualized_return / abs(self.max_drawdown)

    def _calculate_information_ratio(self, benchmark_returns: pd.Series = None) -> float:
        """
        计算信息比率 (相对于基准的超额收益 / 跟踪误差)

        Args:
            benchmark_returns: 基准收益率序列
        """
        if len(self.returns) < 2:
            return 0.0

        if benchmark_returns is None:
            # 假设基准收益率为0
            active_return = self.returns
        else:
            # 对齐数据
            min_len = min(len(self.returns), len(benchmark_returns))
            active_return = self.returns.iloc[-min_len] - benchmark_returns.iloc[-min_len:].values

        tracking_error = active_return.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        annualized_active = active_return.mean() * 252
        return annualized_active / tracking_error

    # ========== 交易统计指标 ==========

    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if len(self.returns) == 0:
            return 0.0

        winning_trades = (self.returns > 0).sum()
        total_trades = (self.returns != 0).sum()

        if total_trades == 0:
            return 0.0

        return winning_trades / total_trades

    def _calculate_profit_factor(self) -> float:
        """计算盈亏比 (总盈利 / 总亏损)"""
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = abs(self.returns[self.returns < 0].sum())

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_avg_win(self) -> float:
        """计算平均盈利"""
        positive_returns = self.returns[self.returns > 0]
        if len(positive_returns) == 0:
            return 0.0
        return positive_returns.mean()

    def _calculate_avg_loss(self) -> float:
        """计算平均亏损"""
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        return negative_returns.mean()

    def _calculate_win_loss_ratio(self) -> float:
        """计算盈亏比 (平均盈利 / |平均亏损|)"""
        avg_win = self._calculate_avg_win()
        avg_loss = abs(self._calculate_avg_loss())

        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0

        return avg_win / avg_loss

    def _calculate_expectancy(self) -> float:
        """
        计算交易期望值
        E = (WinRate × AvgWin) - ((1 - WinRate) × |AvgLoss|)
        """
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = abs(self._calculate_avg_loss())

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _estimate_trade_count(self) -> int:
        """估计交易次数"""
        # 通过收益率变化的sign变化来估计
        sign_changes = (self.returns.sign().diff() != 0).sum()
        return int(sign_changes)

    # ========== 进阶统计指标 ==========

    def _calculate_skewness(self) -> float:
        """计算偏度"""
        if len(self.returns) < 3:
            return 0.0
        return self.returns.skew()

    def _calculate_kurtosis(self) -> float:
        """计算峰度"""
        if len(self.returns) < 4:
            return 0.0
        return self.returns.kurtosis()

    def _calculate_tail_ratio(self) -> float:
        """计算尾部比率 (正收益绝对值均值 / 负收益绝对值均值)"""
        avg_win = self._calculate_avg_win()
        avg_loss = abs(self._calculate_avg_loss())

        if avg_loss == 0:
            return 1.0

        return avg_win / avg_loss

    def _calculate_common_sense_ratio(self) -> float:
        """
        "常识"比率
        综合考虑收益、波动性和回撤
        """
        if self.max_drawdown >= 0 or self.max_drawdown == 0:
            return 0.0 if self.annualized_return <= 0 else float('inf')

        # 简化公式
        return (self.annualized_return / self.annualized_volatility) / abs(self.max_drawdown)

    # ========== 滚动指标 ==========

    def _calculate_rolling_sharpe(self, window: int = 252) -> pd.Series:
        """计算滚动夏普比率"""
        rolling_mean = self.returns.rolling(window=window).mean() * 252
        rolling_std = self.returns.rolling(window=window).std() * np.sqrt(252)

        return rolling_mean / (rolling_std + 1e-10)

    def _calculate_rolling_max_drawdown(self, window: int = 252) -> pd.Series:
        """计算滚动最大回撤"""
        rolling_equity = self.equity_curve.rolling(window=window).max()
        drawdown = self.equity_curve / rolling_equity - 1
        return drawdown

    # ========== 辅助方法 ==========

    def _calculate_cumulative_returns(self) -> pd.Series:
        """计算累计收益率"""
        return (1 + self.returns).cumprod() - 1

    def get_summary(self) -> str:
        """获取指标摘要"""
        metrics = self.calculate_all()

        summary = f"""
========== 回测绩效摘要 ==========

【收益指标】
  总收益:     {metrics['total_return']:.2%}
  年化收益:   {metrics['annualized_return']:.2%}
  月均收益:   {metrics['monthly_returns'].mean():.2%}

【风险指标】
  年化波动率: {metrics['annualized_volatility']:.2%}
  最大回撤:   {metrics['max_drawdown']:.2%}
  回撤天数:   {metrics['max_drawdown_duration']}天

【风险调整收益】
  夏普比率:   {metrics['sharpe_ratio']:.3f}
  索提诺比率: {metrics['sortino_ratio']:.3f}
  卡玛比率:   {metrics['calmar_ratio']:.3f}

【交易统计】
  胜率:       {metrics['win_rate']:.2%}
  盈亏比:     {metrics['profit_factor']:.2f}
  交易次数:   {metrics['total_trades']}

==================================
"""
        return summary


# 便捷函数
def calculate_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> Dict:
    """
    便捷函数：计算回测指标

    Args:
        equity_curve: 资金曲线 (净值序列)
        risk_free_rate: 年化无风险利率

    Returns:
        Dict: 包含所有指标的字典
    """
    metrics = BacktestMetrics(equity_curve, risk_free_rate=risk_free_rate)
    return metrics.calculate_all()


def print_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> None:
    """
    便捷函数：打印回测指标

    Args:
        equity_curve: 资金曲线 (净值序列)
        risk_free_rate: 年化无风险利率
    """
    metrics = BacktestMetrics(equity_curve, risk_free_rate=risk_free_rate)
    print(metrics.get_summary())
