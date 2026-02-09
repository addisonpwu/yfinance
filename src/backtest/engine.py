"""
回测引擎模块

提供完整的回测功能:
- 避免前视偏差
- 交易成本建模
- 风险控制
- 蒙特卡洛测试
-  Walk-Forward 验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from src.backtest.metrics import BacktestMetrics, calculate_metrics


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0      # 初始资金
    commission: float = 0.001              # 手续费 (0.1%)
    slippage: float = 0.0005               # 滑点 (0.05%)
    risk_free_rate: float = 0.02           # 无风险利率
    max_position: float = 0.2              # 最大仓位 (20%)
    stop_loss: Optional[float] = None      # 止损比例
    take_profit: Optional[float] = None    # 止盈比例
    position_sizing: str = "equal"         # 仓位管理方式: "equal" | "kelly" | "volatility"
    kelly_fraction: float = 0.5            # Kelly 分数 (减半以降低风险)


@dataclass
class Trade:
    """交易记录"""
    entry_date: Any
    entry_price: float
    exit_date: Any
    exit_price: float
    direction: str  # "long" | "short"
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    reason: str = ""  # "stop_loss" | "take_profit" | "signal" | "end"


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series
    trades: List[Trade]
    metrics: Dict
    config: BacktestConfig
    signals: pd.Series = None
    drawdown_series: pd.Series = None


class BacktestEngine:
    """
    回测引擎

    支持:
    - 单标的回测
    - 多策略组合回测
    - 蒙特卡洛模拟
    - Walk-Forward 验证
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测引擎

        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = None

    def run(self, data: pd.DataFrame, strategy_func: Callable,
            initial_capital: float = None) -> BacktestResult:
        """
        运行回测

        Args:
            data: OHLCV 数据 (必须包含: Open, High, Low, Close, Volume)
            strategy_func: 策略函数，接受 DataFrame，返回信号 (1=买入, -1=卖出, 0=持有)
            initial_capital: 初始资金 (覆盖 config)

        Returns:
            BacktestResult: 回测结果
        """
        # 验证数据
        required_cols = ['Close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据必须包含 {col} 列")

        # 设置初始资金
        capital = initial_capital or self.config.initial_capital
        self.trades = []

        # 初始化资金曲线
        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = capital

        # 持仓状态
        position = 0  # 1=多头, 0=空仓, -1=空头
        entry_price = 0.0
        quantity = 0.0

        # 记录信号
        signals = pd.Series(index=data.index, dtype=float)

        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i] if hasattr(data.index[i], 'strftime') else i

            # 获取策略信号
            signal = strategy_func(data.iloc[:i+1])
            if hasattr(signal, '__len__'):
                signal = signal.iloc[-1] if len(signal) > 0 else 0
            signals.iloc[i] = signal

            # 计算交易成本
            commission = 0.0
            slippage_cost = 0.0

            # 检查止损/止盈
            stop_triggered = False
            profit_taken = False

            if position != 0 and self.config.stop_loss:
                if position > 0:
                    if current_price <= entry_price * (1 - self.config.stop_loss):
                        stop_triggered = True
                else:
                    if current_price >= entry_price * (1 + self.config.stop_loss):
                        stop_triggered = True

            if position != 0 and self.config.take_profit:
                if position > 0:
                    if current_price >= entry_price * (1 + self.config.take_profit):
                        profit_taken = True
                else:
                    if current_price <= entry_price * (1 - self.config.take_profit):
                        profit_taken = True

            # 平仓逻辑
            if (position != 0 and
                (signal == -position or stop_triggered or profit_taken or i == len(data) - 1)):

                # 计算平仓价格 (考虑滑点)
                if position > 0:  # 多头平仓
                    exit_price = current_price * (1 - self.config.slippage)
                else:  # 空头平仓
                    exit_price = current_price * (1 + self.config.slippage)

                # 计算收益
                if position > 0:
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                # 手续费
                commission = (entry_price + exit_price) * quantity * self.config.commission
                pnl -= commission

                # 记录交易
                self.trades.append(Trade(
                    entry_date=data.index[i-1] if i > 0 else current_date,
                    entry_price=entry_price,
                    exit_date=current_date,
                    exit_price=exit_price,
                    direction="long" if position > 0 else "short",
                    quantity=quantity,
                    pnl=pnl,
                    pnl_pct=pnl / (entry_price * quantity),
                    commission=commission,
                    reason="stop_loss" if stop_triggered else
                           "take_profit" if profit_taken else
                           "end" if i == len(data) - 1 else "signal"
                ))

                # 更新资金
                capital += pnl
                position = 0
                quantity = 0.0

            # 开仓逻辑
            if position == 0 and signal != 0:
                # 计算买入价格 (考虑滑点)
                if signal > 0:  # 买入
                    open_price = current_price * (1 + self.config.slippage)
                else:  # 卖出
                    open_price = current_price * (1 - self.config.slippage)

                # 计算仓位
                if self.config.position_sizing == "equal":
                    # 等额仓位
                    position_value = capital * self.config.max_position
                    quantity = position_value / open_price
                elif self.config.position_sizing == "volatility":
                    # 波动率仓位 (简化)
                    atr = self._calculate_atr(data.iloc[:i+1], 14)
                    vol_position = capital * 0.02 / (atr / open_price)  # 2%风险
                    quantity = min(vol_position / open_price, capital / open_price * self.config.max_position)
                else:
                    # 默认等额
                    quantity = (capital * self.config.max_position) / open_price

                entry_price = open_price
                position = signal

                # 扣手续费
                commission = entry_price * quantity * self.config.commission
                capital -= commission

            # 记录资金
            equity.iloc[i] = capital

        # 计算指标
        metrics = calculate_metrics(equity, self.config.risk_free_rate)

        # 计算回撤序列
        drawdown = (equity - equity.cummax()) / equity.cummax()

        return BacktestResult(
            equity_curve=equity,
            trades=self.trades,
            metrics=metrics,
            config=self.config,
            signals=signals,
            drawdown_series=drawdown
        )

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """计算 ATR"""
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]

    def monte_carlo_test(self, result: BacktestResult, n_runs: int = 1000,
                          confidence_level: float = 0.95) -> Dict:
        """
        蒙特卡洛测试

        通过打乱收益率序列来评估策略稳健性

        Args:
            result: 回测结果
            n_runs: 模拟次数
            confidence_level: 置信水平

        Returns:
            Dict: 包含置信区间的指标
        """
        returns = result.equity_curve.pct_change(fill_method=None).dropna()
        n = len(returns)

        # 存储结果
        all_returns = []
        all_sharpes = []
        all_max_dd = []

        for _ in range(n_runs):
            # 打乱收益率顺序
            shuffled_returns = np.random.choice(returns.values, size=n, replace=True)

            # 构建随机权益曲线
            equity = pd.Series([1.0] * (n + 1))
            for r in shuffled_returns:
                equity.iloc[len(equity) - n - 1:] = equity.iloc[len(equity) - n - 1:] * (1 + r)

            # 计算指标
            metrics = calculate_metrics(equity)
            all_returns.append(metrics['total_return'])
            all_sharpes.append(metrics['sharpe_ratio'])
            all_max_dd.append(metrics['max_drawdown'])

        all_returns = np.array(all_returns)
        all_sharpes = np.array(all_sharpes)
        all_max_dd = np.array(all_max_dd)

        # 计算置信区间
        lower = (1 - confidence_level) / 2
        upper = 1 - lower

        return {
            "confidence_level": confidence_level,
            "total_return": {
                "mean": float(all_returns.mean()),
                "std": float(all_returns.std()),
                "ci_low": float(np.percentile(all_returns, lower * 100)),
                "ci_high": float(np.percentile(all_returns, upper * 100)),
            },
            "sharpe_ratio": {
                "mean": float(all_sharpes.mean()),
                "std": float(all_sharpes.std()),
                "ci_low": float(np.percentile(all_sharpes, lower * 100)),
                "ci_high": float(np.percentile(all_sharpes, upper * 100)),
            },
            "max_drawdown": {
                "mean": float(all_max_dd.mean()),
                "std": float(all_max_dd.std()),
                "ci_low": float(np.percentile(all_max_dd, lower * 100)),
                "ci_high": float(np.percentile(all_max_dd, upper * 100)),
            },
            "runs": n_runs,
            "prob_positive": float((all_returns > 0).mean()),
            "prob_sharpe_gt_1": float((all_sharpes > 1).mean()),
            "prob_max_dd_lt_10": float((np.abs(all_max_dd) < 0.10).mean()),
        }

    def walk_forward_validation(self, data: pd.DataFrame, strategy_func: Callable,
                                  train_window: int = 252, test_window: int = 63,
                                  step: int = 21) -> List[BacktestResult]:
        """
        Walk-Forward 验证

        使用滑动窗口进行样本外测试

        Args:
            data: 完整数据
            strategy_func: 策略函数
            train_window: 训练窗口大小 (天数)
            test_window: 测试窗口大小 (天数)
            step: 滑动步长

        Returns:
            List[BacktestResult]: 每个测试窗口的回测结果
        """
        results = []
        n = len(data)

        start = train_window
        while start + test_window <= n:
            # 训练期
            train_data = data.iloc[start - train_window:start]

            # 测试期
            test_data = data.iloc[start:start + test_window]

            # 优化策略参数 (这里简化处理，实际应该根据训练数据调优)
            optimized_func = self._optimize_strategy(train_data, strategy_func)

            # 在测试期运行回测
            result = self.run(test_data, optimized_func)
            result.name = f"WF_{start - train_window}_{start}"

            results.append(result)

            # 滑动窗口
            start += step

        return results

    def _optimize_strategy(self, train_data: pd.DataFrame,
                           strategy_func: Callable) -> Callable:
        """
        优化策略参数 (简化版本)

        实际实现应该:
        1. 在训练数据上测试不同参数
        2. 选择最优参数
        3. 返回优化后的策略函数
        """
        # 这里直接返回原策略，实际应该进行参数优化
        return strategy_func

    def compare_strategies(self, data: pd.DataFrame,
                           strategies: Dict[str, Callable],
                           initial_capital: float = None) -> Dict:
        """
        比较多个策略

        Args:
            data: OHLCV 数据
            strategies: 策略字典 {名称: 策略函数}
            initial_capital: 初始资金

        Returns:
            Dict: 各策略的回测结果
        """
        results = {}

        for name, func in strategies.items():
            try:
                result = self.run(data, func, initial_capital)
                results[name] = {
                    "result": result,
                    "metrics": result.metrics,
                    "num_trades": len(result.trades),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        return results

    def get_trade_statistics(self, trades: List[Trade]) -> Dict:
        """
        获取交易统计

        Args:
            trades: 交易列表

        Returns:
            Dict: 交易统计
        """
        if not trades:
            return {"error": "无交易记录"}

        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades) if trades else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "max_pnl": max(pnls) if pnls else 0,
            "min_pnl": min(pnls) if pnls else 0,
            "avg_win": np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            "avg_pnl_pct": np.mean(pnl_pcts),
            "max_pnl_pct": max(pnl_pcts) if pnl_pcts else 0,
            "min_pnl_pct": min(pnl_pcts) if pnl_pcts else 0,
            "profit_factor": (
                sum([t.pnl for t in winning_trades]) /
                abs(sum([t.pnl for t in losing_trades]))
            ) if losing_trades else float('inf'),
            "avg_trade_duration": self._calculate_avg_duration(trades),
            "exit_reasons": self._count_exit_reasons(trades),
        }

    def _calculate_avg_duration(self, trades: List[Trade]) -> float:
        """计算平均交易持续时间"""
        if not trades:
            return 0

        durations = []
        for t in trades:
            try:
                if isinstance(t.entry_date, datetime) and isinstance(t.exit_date, datetime):
                    duration = (t.exit_date - t.entry_date).days
                else:
                    duration = 1  # 假设1天
                durations.append(duration)
            except (TypeError, AttributeError):
                durations.append(1)

        return np.mean(durations) if durations else 0

    def _count_exit_reasons(self, trades: List[Trade]) -> Dict[str, int]:
        """统计平仓原因"""
        reasons = {}
        for t in trades:
            reason = t.reason
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons


# 便捷函数
def backtest(data: pd.DataFrame, strategy_func: Callable,
             initial_capital: float = 100000, **kwargs) -> BacktestResult:
    """
    便捷函数：快速运行回测

    Args:
        data: OHLCV 数据
        strategy_func: 策略函数
        initial_capital: 初始资金
        **kwargs: 其他回测配置

    Returns:
        BacktestResult: 回测结果
    """
    config = BacktestConfig(initial_capital=initial_capital, **kwargs)
    engine = BacktestEngine(config)
    return engine.run(data, strategy_func)


def print_result(result: BacktestResult) -> None:
    """打印回测结果"""
    print("\n" + "=" * 50)
    print("回测结果")
    print("=" * 50)

    # 权益曲线摘要
    equity = result.equity_curve
    print(f"\n初始资金: {equity.iloc[0]:,.2f}")
    print(f"最终资金: {equity.iloc[-1]:,.2f}")
    print(f"总收益: {(equity.iloc[-1] / equity.iloc[0] - 1) * 100:.2f}%")

    # 指标摘要
    print("\n关键指标:")
    for key in ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']:
        if key in result.metrics:
            value = result.metrics[key]
            if isinstance(value, float):
                if 'rate' in key or 'win' in key:
                    print(f"  {key}: {value * 100:.2f}%")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # 交易统计
    print(f"\n交易次数: {len(result.trades)}")
    if result.trades:
        win_trades = [t for t in result.trades if t.pnl > 0]
        print(f"盈利交易: {len(win_trades)}")
        print(f"亏损交易: {len(result.trades) - len(win_trades)}")

    print("=" * 50 + "\n")
