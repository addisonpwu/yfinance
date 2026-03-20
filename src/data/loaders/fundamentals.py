"""
基本面分析模組

結構化獲取和處理財報數據：
- EPS、營收、現金流
- 毛利率、淨利率、ROE
- 負債權益比
- 基本面評分維度

作者: iFlow CLI Team
日期: 2026-03-18
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf


def get_financial_data(symbol: str) -> Dict:
    """
    獲取股票的基本面數據
    
    Args:
        symbol: 股票代碼 (如 'AAPL' 或 '0700.HK')
        
    Returns:
        Dict: 結構化的基本面數據
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # 獲取財務報表數據
        try:
            financials = ticker.financials
            if financials is not None and not financials.empty:
                # 最近一個財年的營收和淨利潤
                latest = financials.iloc[0] if len(financials) > 0 else None
            else:
                latest = None
        except Exception:
            latest = None
        
        try:
            balance_sheet = ticker.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                bs_latest = balance_sheet.iloc[0] if len(balance_sheet) > 0 else None
            else:
                bs_latest = None
        except Exception:
            bs_latest = None
        
        try:
            cashflow = ticker.cashflow
            if cashflow is not None and not cashflow.empty:
                cf_latest = cashflow.iloc[0] if len(cashflow) > 0 else None
            else:
                cf_latest = None
        except Exception:
            cf_latest = None
        
        # 構建結構化數據
        data = {
            # 基本信息
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', 'N/A')),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            
            # 估值指標
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_value': info.get('enterpriseValue'),
            'ev_to_ebitda': info.get('enterpriseToRevenue'),
            
            # 盈利能力
            'eps': info.get('trailingEps'),
            'eps_forward': info.get('forwardEps'),
            'profit_margin': info.get('profitMargins'),
            'gross_margin': info.get('grossMargins'),
            'operating_margin': info.get('operatingMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            
            # 營收和增長
            'revenue': info.get('totalRevenue'),
            'revenue_growth': info.get('revenueGrowth'),
            'revenue_per_share': info.get('revenuePerShare'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
            
            # 現金流
            'operating_cashflow': info.get('operatingCashflow') if cf_latest is None else _safe_get_value(cf_latest, 'Operating Cash Flow'),
            'free_cashflow': info.get('freeCashflow'),
            
            # 財務健康
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'total_debt': info.get('totalDebt'),
            'total_cash': info.get('totalCash'),
            
            # 股息
            'dividend_yield': info.get('dividendYield'),
            'dividend_rate': info.get('dividendRate'),
            'payout_ratio': info.get('payoutRatio'),
            
            # 其他
            'beta': info.get('beta'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'target_mean_price': info.get('targetMeanPrice'),
            'analyst_count': info.get('numberOfAnalystOpinions'),
        }
        
        return data
        
    except Exception as e:
        return {'error': str(e), 'symbol': symbol}


def _safe_get_value(series, key):
    """安全獲取財務數據"""
    try:
        if key in series.index:
            val = series[key]
            return float(val) if pd.notna(val) else None
    except Exception:
        pass
    return None


def score_fundamentals(financial_data: Dict) -> Dict:
    """
    對基本面進行四維度評分
    
    Args:
        financial_data: get_financial_data 返回的數據
        
    Returns:
        Dict: 包含四維度評分和綜合評分
    """
    scores = {
        'profitability': 0,
        'growth': 0,
        'financial_health': 0,
        'valuation': 0
    }
    
    details = {
        'profitability': [],
        'growth': [],
        'financial_health': [],
        'valuation': []
    }
    
    # 1. 盈利能力評分 (ROE, 淨利率, 毛利率)
    roe = financial_data.get('roe')
    if roe and roe > 0.15:
        scores['profitability'] += 3
        details['profitability'].append(f"ROE {roe:.1%} 優秀")
    elif roe and roe > 0.10:
        scores['profitability'] += 2
        details['profitability'].append(f"ROE {roe:.1%} 良好")
    elif roe and roe > 0.05:
        scores['profitability'] += 1
        details['profitability'].append(f"ROE {roe:.1%} 一般")
    else:
        details['profitability'].append("ROE 較低或為負")
    
    pm = financial_data.get('profit_margin')
    if pm and pm > 0.15:
        scores['profitability'] += 3
        details['profitability'].append(f"淨利率 {pm:.1%} 優秀")
    elif pm and pm > 0.08:
        scores['profitability'] += 2
        details['profitability'].append(f"淨利率 {pm:.1%} 良好")
    elif pm and pm > 0:
        scores['profitability'] += 1
        details['profitability'].append(f"淨利率 {pm:.1%} 一般")
    
    gm = financial_data.get('gross_margin')
    if gm and gm > 0.40:
        scores['profitability'] += 2
        details['profitability'].append(f"毛利率 {gm:.1%} 優秀")
    elif gm and gm > 0.25:
        scores['profitability'] += 1
        details['profitability'].append(f"毛利率 {gm:.1%} 良好")
    
    # 標準化盈利能力評分 (0-10)
    scores['profitability'] = min(10, scores['profitability'])
    
    # 2. 成長性評分
    rev_growth = financial_data.get('revenue_growth')
    if rev_growth and rev_growth > 0.20:
        scores['growth'] += 3
        details['growth'].append(f"營收增長 {rev_growth:.1%} 強勁")
    elif rev_growth and rev_growth > 0.10:
        scores['growth'] += 2
        details['growth'].append(f"營收增長 {rev_growth:.1%} 良好")
    elif rev_growth and rev_growth > 0:
        scores['growth'] += 1
        details['growth'].append(f"營收增長 {rev_growth:.1%} 平穩")
    
    eps_growth = financial_data.get('earnings_growth')
    if eps_growth and eps_growth > 0.20:
        scores['growth'] += 3
        details['growth'].append(f"EPS增長 {eps_growth:.1%} 強勁")
    elif eps_growth and eps_growth > 0.10:
        scores['growth'] += 2
        details['growth'].append(f"EPS增長 {eps_growth:.1%} 良好")
    elif eps_growth and eps_growth > 0:
        scores['growth'] += 1
        details['growth'].append(f"EPS增長 {eps_growth:.1%} 平穩")
    
    scores['growth'] = min(10, scores['growth'])
    
    # 3. 財務健康評分
    de = financial_data.get('debt_to_equity')
    if de is not None:
        if de < 0:
            scores['financial_health'] += 3
            details['financial_health'].append("淨現金，財務非常健康")
        elif de < 50:
            scores['financial_health'] += 3
            details['financial_health'].append(f"負債率 {de:.0f}% 較低")
        elif de < 100:
            scores['financial_health'] += 2
            details['financial_health'].append(f"負債率 {de:.0f}% 適中")
        elif de < 150:
            scores['financial_health'] += 1
            details['financial_health'].append(f"負債率 {de:.0f}% 較高")
        else:
            details['financial_health'].append(f"負債率 {de:.0f}% 過高")
    
    cr = financial_data.get('current_ratio')
    if cr and cr > 2:
        scores['financial_health'] += 2
        details['financial_health'].append(f"流動比率 {cr:.1f} 優秀")
    elif cr and cr > 1.5:
        scores['financial_health'] += 1
        details['financial_health'].append(f"流動比率 {cr:.1f} 良好")
    
    scores['financial_health'] = min(10, scores['financial_health'])
    
    # 4. 估值評分 (PE, PB, PEG)
    pe = financial_data.get('pe_ratio')
    if pe and pe > 0:
        if pe < 15:
            scores['valuation'] += 3
            details['valuation'].append(f"PE {pe:.1f} 偏低")
        elif pe < 25:
            scores['valuation'] += 2
            details['valuation'].append(f"PE {pe:.1f} 合理")
        elif pe < 40:
            scores['valuation'] += 1
            details['valuation'].append(f"PE {pe:.1f} 偏高")
        else:
            details['valuation'].append(f"PE {pe:.1f} 過高")
    
    peg = financial_data.get('peg_ratio')
    if peg and peg > 0:
        if peg < 1:
            scores['valuation'] += 2
            details['valuation'].append(f"PEG {peg:.2f} 優秀")
        elif peg < 1.5:
            scores['valuation'] += 1
            details['valuation'].append(f"PEG {peg:.2f} 合理")
        else:
            details['valuation'].append(f"PEG {peg:.2f} 偏高")
    
    pb = financial_data.get('price_to_book')
    if pb and pb > 0:
        if pb < 2:
            scores['valuation'] += 2
            details['valuation'].append(f"PB {pb:.1f} 偏低")
        elif pb < 5:
            scores['valuation'] += 1
            details['valuation'].append(f"PB {pb:.1f} 合理")
    
    scores['valuation'] = min(10, scores['valuation'])
    
    # 計算綜合評分 (加權平均)
    weights = {
        'profitability': 0.30,
        'growth': 0.25,
        'financial_health': 0.20,
        'valuation': 0.25
    }
    
    total_score = (
        scores['profitability'] * weights['profitability'] +
        scores['growth'] * weights['growth'] +
        scores['financial_health'] * weights['financial_health'] +
        scores['valuation'] * weights['valuation']
    )
    
    return {
        'scores': scores,
        'details': details,
        'total_score': round(total_score, 1),
        'total_score_10': round(total_score, 0),  # 0-10整數
        'grade': _get_grade(total_score),
        'weighted_formula': weights
    }


def _get_grade(score: float) -> str:
    """根據評分獲取等級"""
    if score >= 8:
        return 'A'
    elif score >= 7:
        return 'B+'
    elif score >= 6:
        return 'B'
    elif score >= 5:
        return 'C+'
    elif score >= 4:
        return 'C'
    else:
        return 'D'


def format_fundamental_analysis(financial_data: Dict, scores: Dict) -> str:
    """
    格式化基本面分析結果
    
    Args:
        financial_data: 原始財務數據
        scores: 評分結果
        
    Returns:
        str: 格式化的分析結果
    """
    lines = [
        "【基本面分析】",
        f"股票: {financial_data.get('name', 'N/A')} ({financial_data.get('symbol', 'N/A')})",
        f"行業: {financial_data.get('sector', 'N/A')} / {financial_data.get('industry', 'N/A')}",
        "",
        f"綜合評分: {scores['total_score']:.1f}/10 等級: {scores['grade']}",
        "",
        "估值指標:",
        f"  市值: {format_value(financial_data.get('market_cap'), 'money')}",
        f"  PE: {format_value(financial_data.get('pe_ratio'), 'ratio')}",
        f"  PEG: {format_value(financial_data.get('peg_ratio'), 'ratio')}",
        f"  PB: {format_value(financial_data.get('price_to_book'), 'ratio')}",
        "",
        "盈利能力:",
        f"  ROE: {format_value(financial_data.get('roe'), 'percent')}",
        f"  淨利率: {format_value(financial_data.get('profit_margin'), 'percent')}",
        f"  毛利率: {format_value(financial_data.get('gross_margin'), 'percent')}",
        "",
        "成長性:",
        f"  營收增長: {format_value(financial_data.get('revenue_growth'), 'percent')}",
        f"  EPS增長: {format_value(financial_data.get('earnings_growth'), 'percent')}",
        "",
        "財務健康:",
        f"  負債率: {format_value(financial_data.get('debt_to_equity'), 'percent')}",
        f"  流動比率: {format_value(financial_data.get('current_ratio'), 'ratio')}",
        "",
        "評分明細:",
        f"  盈利能力: {scores['scores']['profitability']}/10",
        f"  成長性: {scores['scores']['growth']}/10",
        f"  財務健康: {scores['scores']['financial_health']}/10",
        f"  估值水平: {scores['scores']['valuation']}/10",
    ]
    
    return "\n".join(lines)


def format_value(value, format_type='number'):
    """格式化數值"""
    if value is None:
        return "N/A"
    
    try:
        if format_type == 'money':
            if value >= 1e12:
                return f"${value/1e12:.2f}T"
            elif value >= 1e9:
                return f"${value/1e9:.2f}B"
            elif value >= 1e6:
                return f"${value/1e6:.2f}M"
            else:
                return f"${value:.0f}"
        elif format_type == 'percent':
            return f"{value*100:.1f}%"
        elif format_type == 'ratio':
            return f"{value:.2f}"
        else:
            return str(value)
    except:
        return str(value)


# 測試函數
if __name__ == "__main__":
    import time
    
    # 測試美股
    print("測試 AAPL 基本面數據...")
    data = get_financial_data('AAPL')
    
    if 'error' not in data:
        scores = score_fundamentals(data)
        print(format_fundamental_analysis(data, scores))
    else:
        print(f"錯誤: {data['error']}")
    
    time.sleep(1)
