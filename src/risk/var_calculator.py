"""
VaR 風險計算模組

計算投資組合的 Value at Risk：
- 歷史模擬法 VaR
- 95%/99% 置信度支持
- 單股票和組合 VaR 計算

作者: iFlow CLI Team
日期: 2026-03-18
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VaRResult:
    """VaR 計算結果"""
    var_95: float  # 95% 置信度 VaR
    var_99: float  # 99% 置信度 VaR
    cvar_95: float  # 條件 VaR (Expected Shortfall)
    confidence: float  # 置信水平
    holding_period: int  # 持有期（天）
    current_price: float  # 當前價格
    percentile_95: float  # 95%分位數收益率
    percentile_99: float  # 99%分位數收益率
    mean_return: float  # 平均收益率
    std_return: float  # 收益率標準差


def calculate_var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    holding_period: int = 1
) -> float:
    """
    歷史模擬法計算 VaR
    
    Args:
        returns: 收益率序列
        confidence: 置信度 (0.95 或 0.99)
        holding_period: 持有期
        
    Returns:
        float: VaR 值（正數表示損失）
    """
    if returns.empty or len(returns) < 30:
        return 0.0
    
    # 調整持有期（平方根法則）
    if holding_period > 1:
        returns = returns * np.sqrt(holding_period)
    
    # 計算分位數
    alpha = 1 - confidence
    var = np.percentile(returns, alpha * 100)
    
    return abs(var)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    holding_period: int = 1
) -> float:
    """
    計算條件 VaR (Expected Shortfall)
    
    Args:
        returns: 收益率序列
        confidence: 置信度
        holding_period: 持有期
        
    Returns:
        float: CVaR 值
    """
    if returns.empty or len(returns) < 30:
        return 0.0
    
    # 調整持有期
    if holding_period > 1:
        returns = returns * np.sqrt(holding_period)
    
    # 計算 VaR
    alpha = 1 - confidence
    var = np.percentile(returns, alpha * 100)
    
    # CVaR = 超出 VaR 的平均損失
    cvar = returns[returns <= var].mean()
    
    return abs(cvar) if not np.isnan(cvar) else 0.0


def calculate_stock_var(
    hist: pd.DataFrame,
    confidence: float = 0.95,
    holding_period: int = 1,
    lookback_days: int = 252
) -> VaRResult:
    """
    計算單股票的 VaR
    
    Args:
        hist: 歷史價格數據
        confidence: 置信度 (默認 95%)
        holding_period: 持有期（天，默認1天）
        lookback_days: 回看天數（默認1年）
        
    Returns:
        VaRResult: VaR 計算結果
    """
    if hist is None or hist.empty or len(hist) < 30:
        return VaRResult(0, 0, 0, 0.95, 1, 0, 0, 0, 0, 0)
    
    # 計算日收益率
    prices = hist['Close'].tail(min(lookback_days, len(hist)))
    returns = prices.pct_change().dropna()
    
    if len(returns) < 30:
        return VaRResult(0, 0, 0, 0.95, 1, 0, 0, 0, 0, 0)
    
    current_price = hist['Close'].iloc[-1]
    
    # 計算 VaR
    var_95 = calculate_var_historical(returns, 0.95, holding_period)
    var_99 = calculate_var_historical(returns, 0.99, holding_period)
    cvar_95 = calculate_cvar(returns, 0.95, holding_period)
    
    # 計算統計數據
    percentile_95 = np.percentile(returns, 5)
    percentile_99 = np.percentile(returns, 1)
    mean_return = returns.mean()
    std_return = returns.std()
    
    return VaRResult(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        confidence=confidence,
        holding_period=holding_period,
        current_price=current_price,
        percentile_95=percentile_95,
        percentile_99=percentile_99,
        mean_return=mean_return,
        std_return=std_return
    )


def calculate_portfolio_var(
    holdings: Dict[str, Dict],
    hist_data: Dict[str, pd.DataFrame],
    confidence: float = 0.95,
    holding_period: int = 1
) -> Dict:
    """
    計算投資組合的 VaR
    
    Args:
        holdings: 持倉字典 {symbol: {'shares': 股數, 'price': 價格}}
        hist_data: 各股票歷史數據 {symbol: DataFrame}
        confidence: 置信度
        holding_period: 持有期
        
    Returns:
        Dict: 組合 VaR 分析結果
    """
    if not holdings or not hist_data:
        return {'error': '持倉數據不足'}
    
    # 計算各資產收益率
    returns_dict = {}
    for symbol, hist in hist_data.items():
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            returns = hist['Close'].pct_change().dropna()
            if len(returns) >= 30:
                returns_dict[symbol] = returns
    
    if len(returns_dict) < len(holdings):
        missing = set(holdings.keys()) - set(returns_dict.keys())
        print(f"警告: 以下股票缺少足夠歷史數據: {missing}")
    
    if not returns_dict:
        return {'error': '沒有足夠的歷史數據'}
    
    # 構建收益率矩陣
    min_length = min(len(r) for r in returns_dict.values())
    returns_matrix = pd.DataFrame({
        symbol: r.tail(min_length).values 
        for symbol, r in returns_dict.items()
    })
    
    # 計算持倉權重
    total_value = sum(h['shares'] * h['price'] for h in holdings.values())
    weights = np.array([
        (holdings[symbol]['shares'] * holdings[symbol]['price']) / total_value
        for symbol in returns_matrix.columns
    ])
    
    # 確保權重為正且和為1
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    
    # 計算組合收益率
    portfolio_returns = (returns_matrix * weights).sum(axis=1)
    
    # 計算 VaR
    var_95 = calculate_var_historical(portfolio_returns, confidence, holding_period)
    var_99 = calculate_var_historical(portfolio_returns, 0.99, holding_period)
    cvar_95 = calculate_cvar(portfolio_returns, confidence, holding_period)
    
    # 計算各資產 VaR 貢獻
    var_contributions = {}
    for symbol in returns_matrix.columns:
        if symbol in holdings:
            asset_var = calculate_var_historical(returns_matrix[symbol], confidence, holding_period)
            weight = holdings[symbol]['shares'] * holdings[symbol]['price'] / total_value
            var_contributions[symbol] = {
                'var': asset_var,
                'weight': weight,
                'var_contribution': asset_var * weight
            }
    
    return {
        'portfolio_var_95': var_95,
        'portfolio_var_99': var_99,
        'portfolio_cvar_95': cvar_95,
        'total_value': total_value,
        'var_95_amount': total_value * var_95,  # VaR金額
        'var_99_amount': total_value * var_99,
        'confidence': confidence,
        'holding_period': holding_period,
        'var_contributions': var_contributions,
        'weights': dict(zip(returns_matrix.columns, weights.tolist()))
    }


def format_var_result(var_result: VaRResult, position_size: float = 10000) -> str:
    """
    格式化 VaR 結果
    
    Args:
        var_result: VaRResult 對象
        position_size: 持倉金額
        
    Returns:
        str: 格式化結果
    """
    if var_result.current_price == 0:
        return "VaR分析: 數據不足"
    
    var_95_amount = position_size * var_result.var_95
    var_99_amount = position_size * var_result.var_99
    cvar_95_amount = position_size * var_result.cvar_95
    
    lines = [
        "【VaR 風險分析】",
        f"當前價格: ${var_result.current_price:.2f}",
        f"持倉金額: ${position_size:,.0f}",
        "",
        "風險價值 (VaR):",
        f"  95% VaR: {var_result.var_95:.2%} (${var_95_amount:,.0f})",
        f"  99% VaR: {var_result.var_99:.2%} (${var_result.var_99_amount:,.0f})",
        "",
        "條件風險值 (CVaR):",
        f"  95% CVaR: {var_result.cvar_95:.2%} (${cvar_95_amount:,.0f})",
        "",
        "歷史統計:",
        f"  平均日收益率: {var_result.mean_return:.2%}",
        f"  收益率標準差: {var_result.std_return:.2%}",
        f"  5%分位數: {var_result.percentile_95:.2%}",
        f"  1%分位數: {var_result.percentile_99:.2%}",
        "",
        f"解讀: 在 {int(var_result.confidence*100)}% 置信度下，",
        f"      持有 {var_result.holding_period} 天的最大損失預期為 ${var_95_amount:,.0f}"
    ]
    
    return "\n".join(lines)


# 蒙特卡羅模擬模組
def monte_carlo_simulation(
    hist: pd.DataFrame,
    n_simulations: int = 1000,
    n_days: int = 30,
    initial_price: float = None
) -> Dict:
    """
    蒙特卡羅價格路徑模擬
    
    Args:
        hist: 歷史價格數據
        n_simulations: 模擬路徑數量
        n_days: 模擬天數
        initial_price: 初始價格，默認使用最新收盤價
        
    Returns:
        Dict: 模擬結果
    """
    if hist is None or hist.empty or len(hist) < 30:
        return {'error': '數據不足'}
    
    # 計算收益率參數
    returns = hist['Close'].pct_change().dropna()
    mu = returns.mean()  # 日均收益率
    sigma = returns.std()  # 日收益率標準差
    
    if initial_price is None:
        initial_price = hist['Close'].iloc[-1]
    
    # 幾何布朗運動模擬
    np.random.seed(42)
    dt = 1  # 每天
    
    # 生成隨機收益率矩陣
    random_returns = np.random.normal(mu, sigma, (n_simulations, n_days))
    
    # 計算價格路徑
    price_paths = np.zeros((n_simulations, n_days + 1))
    price_paths[:, 0] = initial_price
    
    for t in range(1, n_days + 1):
        price_paths[:, t] = price_paths[:, t-1] * (1 + random_returns[:, t-1])
    
    # 計算統計結果
    final_prices = price_paths[:, -1]
    
    results = {
        'initial_price': initial_price,
        'n_simulations': n_simulations,
        'n_days': n_days,
        'mu': mu,
        'sigma': sigma,
        'mean_price': np.mean(final_prices),
        'median_price': np.median(final_prices),
        'std_price': np.std(final_prices),
        'min_price': np.min(final_prices),
        'max_price': np.max(final_prices),
        'price_paths': price_paths,
        'final_prices': final_prices,
        # 概率計算
        'prob_up': np.sum(final_prices > initial_price) / n_simulations,
        'prob_down': np.sum(final_prices < initial_price) / n_simulations,
        # 分位數
        'percentile_5': np.percentile(final_prices, 5),
        'percentile_25': np.percentile(final_prices, 25),
        'percentile_75': np.percentile(final_prices, 75),
        'percentile_95': np.percentile(final_prices, 95),
        # 預期回報
        'expected_return': (np.mean(final_prices) - initial_price) / initial_price,
        'expected_return_5pct': (np.percentile(final_prices, 5) - initial_price) / initial_price,
        'expected_return_95pct': (np.percentile(final_prices, 95) - initial_price) / initial_price,
    }
    
    return results


def format_monte_carlo_result(mc_result: Dict) -> str:
    """格式化蒙特卡羅模擬結果"""
    if 'error' in mc_result:
        return f"蒙特卡羅模擬: {mc_result['error']}"
    
    lines = [
        "【蒙特卡羅模擬結果】",
        f"模擬次數: {mc_result['n_simulations']:,}",
        f"模擬天數: {mc_result['n_days']}天",
        f"初始價格: ${mc_result['initial_price']:.2f}",
        "",
        "價格預測:",
        f"  預期價格: ${mc_result['mean_price']:.2f}",
        f"  中位數價格: ${mc_result['median_price']:.2f}",
        f"  標準差: ${mc_result['std_price']:.2f}",
        "",
        "價格區間:",
        f"  最低: ${mc_result['min_price']:.2f}",
        f"  5%分位: ${mc_result['percentile_5']:.2f}",
        f"  25%分位: ${mc_result['percentile_25']:.2f}",
        f"  75%分位: ${mc_result['percentile_75']:.2f}",
        f"  95%分位: ${mc_result['percentile_95']:.2f}",
        f"  最高: ${mc_result['max_price']:.2f}",
        "",
        "概率預測:",
        f"  上漲概率: {mc_result['prob_up']:.1%}",
        f"  下跌概率: {mc_result['prob_down']:.1%}",
        "",
        "預期回報:",
        f"  平均預期: {mc_result['expected_return']:.1%}",
        f"  5%分位(保守): {mc_result['expected_return_5pct']:.1%}",
        f"  95%分位(樂觀): {mc_result['expected_return_95pct']:.1%}"
    ]
    
    return "\n".join(lines)


# 測試函數
if __name__ == "__main__":
    # 簡單測試
    import numpy as np
    
    # 模擬收益率數據
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    
    result = calculate_stock_var(
        pd.DataFrame({'Close': [100] * 252}),
        confidence=0.95,
        holding_period=1
    )
    
    print(f"VaR (95%): {result.var_95:.2%}")
    print(f"VaR (99%): {result.var_99:.2%}")
    print(f"CVaR (95%): {result.cvar_95:.2%}")