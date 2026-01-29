# ai_analyzer.py

import os
import json
import hashlib
import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# 心流 API 配置
IFLOW_API_URL = "https://apis.iflow.cn/v1/chat/completions"
IFLOW_API_KEY = os.environ.get("IFLOW_API_KEY", "")
DEFAULT_MODEL_NAME = "deepseek-v3.2"

# AI分析结果缓存路径
AI_CACHE_DIR = os.path.join('data_cache', 'ai_analysis')
os.makedirs(AI_CACHE_DIR, exist_ok=True)


def _get_cache_key(stock_data: Dict, hist: pd.DataFrame, interval: str, model: str = 'iflow-rome-30ba3b') -> str:
    """
    生成AI分析结果的缓存键，基于股票数据和历史数据的哈希值
    
    Args:
        stock_data: 股票数据字典
        hist: 历史数据DataFrame
        interval: 数据时段类型
        model: AI模型名称
    
    Returns:
        缓存键字符串
    """
    # 创建一个包含股票数据和历史数据关键信息的字典
    cache_content = {
        'symbol': stock_data.get('symbol', ''),
        'strategies': sorted(stock_data.get('strategies', [])),
        'market': stock_data.get('market', 'HK'),
        'interval': interval,
        'model': model,  # 添加模型信息到缓存键
        'data_timestamp': datetime.now().strftime('%Y-%m-%d'),  # 按天缓存，每天的数据可能不同
        'hist_shape': hist.shape if hist is not None else None,
        'hist_last_date': str(hist.index[-1]) if hist is not None and not hist.empty else None,
        'info_keys': {k: v for k, v in stock_data.get('info', {}).items() 
                      if k in ['marketCap', 'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook', 
                               'profitMargins', 'returnOnEquity', 'revenueGrowth', 'earningsGrowth',
                               'dividendYield', 'beta', '52WeekChange', 'targetMeanPrice']}
    }
    
    # 将字典转换为JSON字符串并生成哈希
    cache_str = json.dumps(cache_content, sort_keys=True, default=str)
    return hashlib.md5(cache_str.encode()).hexdigest()


def _get_cached_result(cache_key: str) -> Optional[Dict]:
    """
    尝试从缓存中获取AI分析结果
    
    Args:
        cache_key: 缓存键
    
    Returns:
        缓存的分析结果，如果不存在或已过期则返回None
    """
    cache_file = os.path.join(AI_CACHE_DIR, f"{cache_key}.json")
    
    try:
        if os.path.exists(cache_file):
            # 检查缓存文件是否在有效期内（默认7天）
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(days=7):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_result = json.load(f)
                print(f" - [AI分析] 从缓存加载 {cached_result.get('symbol', 'Unknown')} 的分析结果")
                return cached_result
            else:
                # 缓存已过期，删除旧文件
                os.remove(cache_file)
                print(f" - [AI分析] 缓存已过期，删除 {cache_key} 的缓存文件")
    except Exception as e:
        print(f" - [AI分析] 读取缓存时出错: {e}")
    
    return None


def _save_result_to_cache(cache_key: str, stock_data: Dict, result: Dict) -> None:
    """
    将AI分析结果保存到缓存
    
    Args:
        cache_key: 缓存键
        stock_data: 原始股票数据
        result: AI分析结果
    """
    try:
        cache_file = os.path.join(AI_CACHE_DIR, f"{cache_key}.json")
        cache_data = {
            'symbol': stock_data.get('symbol', ''),
            'timestamp': datetime.now().isoformat(),
            'stock_data_summary': {
                'strategies': stock_data.get('strategies', []),
                'market': stock_data.get('market', 'HK'),
            },
            'analysis_result': result
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f" - [AI分析] 保存 {stock_data.get('symbol', 'Unknown')} 的分析结果到缓存")
    except Exception as e:
        print(f" - [AI分析] 保存缓存时出错: {e}")


def analyze_stock_with_ai(stock_data: Dict, hist: pd.DataFrame = None, interval: str = '1d', model: str = 'iflow-rome-30ba3b') -> Optional[Dict]:
    """
    使用心流 AI 对股票进行综合分析

    Args:
        stock_data: 包含股票信息的字典，包括：
            - symbol: 股票代码
            - strategies: 符合的策略列表
            - info: 股票基本信息（市值、PE 等）
            - analyzed_news: 新闻分析结果
        hist: 股票历史数据 DataFrame（可选，如果不提供则从缓存读取）
        interval: 数据时段类型 ('1d' 日线, '1h' 小时线, '1m' 分钟线)
        model: 要使用的AI模型名称

    Returns:
        包含 AI 分析结果的字典，或 None（如果分析失败）
    """
    # 如果没有提供 hist，尝试从缓存读取
    if hist is None:
        hist = _load_stock_data_from_cache(stock_data['symbol'], stock_data.get('market', 'HK'), interval)
        if hist is None or hist.empty:
            print(f" - [AI分析] 无法加载 {stock_data['symbol']} 的历史数据，跳过 AI 分析")
            return None
    
    # 生成缓存键 - 包含模型信息
    cache_key = _get_cache_key(stock_data, hist, interval, model)
    
    # 如果模型为 'all'，则对所有模型进行分析
    if model == 'all':
        models_to_use = ['iflow-rome-30ba3b', 'qwen3-max', 'tstars2.0', 'deepseek-v3.2', 'qwen3-coder-plus']
        all_results = []
        for model_name in models_to_use:
            # 为每个模型生成独立的缓存键
            individual_cache_key = _get_cache_key(stock_data, hist, interval, model_name)
            cached_result = _get_cached_result(individual_cache_key)
            
            if cached_result:
                all_results.append(cached_result.get('analysis_result'))
            elif not IFLOW_API_KEY:
                print(f" - [AI分析] 未找到 IFLOW_API_KEY 环境变量，跳过 {model_name} 模型分析（缓存中无结果）")
            else:
                # 构建分析提示词
                prompt = _build_analysis_prompt(stock_data, hist)

                try:
                    # 调用心流 API
                    response, model_used = _call_iflow_api(prompt, model_name)

                    if response:
                        print(f" - [AI分析] 成功完成 {stock_data['symbol']} 的 {model_name} 模型 AI 分析")
                        result = {
                            'summary': response,
                            'model_used': model_used
                        }
                        
                        # 保存结果到缓存
                        _save_result_to_cache(individual_cache_key, stock_data, result)
                        all_results.append(result)
                    else:
                        print(f" - [AI分析] {stock_data['symbol']} 的 {model_name} 模型 AI 分析失败")
                except Exception as e:
                    print(f" - [AI分析] {stock_data['symbol']} 使用 {model_name} 模型分析时出错: {e}")
        
        # 如果所有模型都分析失败，返回None
        if not all_results:
            return None
        
        # 合并所有模型的分析结果
        combined_summary = "【多模型分析结果】\n\n"
        for i, result in enumerate(all_results):
            combined_summary += f"--- {result['model_used']} 模型分析 ---\n"
            combined_summary += result['summary']
            combined_summary += "\n\n"
        
        return {
            'summary': combined_summary,
            'model_used': 'all_models'
        }
    else:
        # 尝试从缓存获取结果（即使没有API密钥也尝试从缓存获取）
        cached_result = _get_cached_result(cache_key)
        if cached_result:
            return cached_result.get('analysis_result')
        
        # 如果没有API密钥且缓存中没有结果，则跳过AI分析
        if not IFLOW_API_KEY:
            print(f" - [AI分析] 未找到 IFLOW_API_KEY 环境变量，跳过 AI 分析（缓存中无结果）")
            return None
        
        # 构建分析提示词
        prompt = _build_analysis_prompt(stock_data, hist)

        try:
            # 调用心流 API
            response, model_used = _call_iflow_api(prompt, model)

            if response:
                print(f" - [AI分析] 成功完成 {stock_data['symbol']} 的 {model} 模型 AI 分析")
                result = {
                    'summary': response,
                    'model_used': model_used
                }
                
                # 保存结果到缓存
                _save_result_to_cache(cache_key, stock_data, result)
                
                return result
            else:
                print(f" - [AI分析] {stock_data['symbol']} 的 {model} 模型 AI 分析失败")
                return None

        except Exception as e:
            print(f" - [AI分析] {stock_data['symbol']} 分析时出错: {e}")
            return None


def _load_stock_data_from_cache(symbol: str, market: str, interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    从 data_cache 加载股票历史数据

    Args:
        symbol: 股票代码
        market: 市场代码 ('US' 或 'HK')
        interval: 数据时段类型 ('1d' 日线, '1h' 小时线, '1m' 分钟线)

    Returns:
        股票历史数据 DataFrame，或 None（如果加载失败）
    """
    def _read_csv_with_auto_index(csv_file: str) -> pd.DataFrame:
        """
        读取 CSV 文件，自动检测并使用正确的索引列名（Date 或 Datetime）
        """
        # 先读取第一行来检测列名
        with open(csv_file, 'r') as f:
            first_line = f.readline()
        
        # 检测索引列名
        if 'Datetime,' in first_line:
            index_col = 'Datetime'
        else:
            index_col = 'Date'
        
        # 使用正确的索引列名读取
        return pd.read_csv(csv_file, index_col=index_col, parse_dates=True)

    try:
        cache_dir = os.path.join('data_cache', market.upper())
        safe_symbol = symbol.replace(":", "_")
        csv_file = os.path.join(cache_dir, f"{safe_symbol}_{interval}.csv")

        if os.path.exists(csv_file):
            hist = _read_csv_with_auto_index(csv_file)
            # 根据不同的 interval 返回对应 100 天的数据量
            if interval == '1m':
                # 分钟线：100天 * 24小时 * 60分钟 = 144000条
                limit = 144000
            elif interval == '1h':
                # 小时线：100天 * 24小时 = 2400条
                limit = 2400
            else:
                # 日线：100天 = 100条
                limit = 100

            if len(hist) > limit:
                hist = hist.tail(limit)
            return hist
        else:
            return None
    except Exception as e:
        print(f" - [AI分析] 加载缓存数据时出错: {e}")
        return None


def _build_analysis_prompt(stock_data: Dict, hist: pd.DataFrame) -> str:
    """
    构建用于股票分析的优化提示词
    """
    symbol = stock_data.get('symbol', 'N/A')
    info = stock_data.get('info', {})
    strategies = stock_data.get('strategies', [])
    analyzed_news = stock_data.get('analyzed_news', [])
    
    # 格式化基本信息
    market_cap = info.get('marketCap')
    market_cap_str = f"{market_cap / 1e8:.2f} 億" if isinstance(market_cap, (int, float)) and market_cap > 0 else "N/A"

    pe_ratio = info.get('trailingPE')
    pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) and pe_ratio > 0 else "N/A"

    forward_pe = info.get('forwardPE')
    forward_pe_str = f"{forward_pe:.2f}" if isinstance(forward_pe, (int, float)) and forward_pe > 0 else "N/A"

    peg_ratio = info.get('pegRatio')
    peg_ratio_str = f"{peg_ratio:.2f}" if isinstance(peg_ratio, (int, float)) and peg_ratio > 0 else "N/A"

    pb_ratio = info.get('priceToBook')
    pb_ratio_str = f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) and pb_ratio > 0 else "N/A"

    profit_margin = info.get('profitMargins')
    profit_margin_str = f"{profit_margin * 100:.2f}%" if isinstance(profit_margin, (int, float)) and profit_margin > 0 else "N/A"

    roe = info.get('returnOnEquity')
    roe_str = f"{roe * 100:.2f}%" if isinstance(roe, (int, float)) and roe > 0 else "N/A"

    revenue_growth = info.get('revenueGrowth')
    revenue_growth_str = f"{revenue_growth * 100:.2f}%" if isinstance(revenue_growth, (int, float)) and revenue_growth > 0 else "N/A"

    earnings_growth = info.get('earningsGrowth')
    earnings_growth_str = f"{earnings_growth * 100:.2f}%" if isinstance(earnings_growth, (int, float)) and earnings_growth > 0 else "N/A"

    dividend_yield = info.get('dividendYield')
    dividend_yield_str = f"{dividend_yield * 100:.2f}%" if isinstance(dividend_yield, (int, float)) and dividend_yield > 0 else "N/A"

    beta = info.get('beta')
    beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) and beta > 0 else "N/A"

    week52_change = info.get('52WeekChange')
    week52_change_str = f"{week52_change * 100:.2f}%" if isinstance(week52_change, (int, float)) and week52_change > 0 else "N/A"

    target_mean_price = info.get('targetMeanPrice')
    target_mean_price_str = f"{target_mean_price:.2f}" if isinstance(target_mean_price, (int, float)) and target_mean_price > 0 else "N/A"

    volume = info.get('volume')
    volume_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) and volume > 0 else "N/A"

    float_shares = info.get('floatShares')
    float_shares_str = f"{float_shares:,.0f}" if isinstance(float_shares, (int, float)) and float_shares > 0 else "N/A"

    shares_short = info.get('sharesShort')
    short_ratio = info.get('shortRatio')
    short_ratio_str = f"{short_ratio:.2f}%" if isinstance(short_ratio, (int, float)) and short_ratio > 0 else "N/A"

    # 格式化历史数据（最近 100 天）
    hist_summary = ""
    technical_indicators = ""
    if hist is not None and not hist.empty:
        recent_data = hist.tail(100)

        # 计算技术指标
        # RSI
        delta = recent_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1] if not rsi.empty else "N/A"

        # MACD
        exp12 = recent_data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = recent_data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        latest_macd = macd.iloc[-1] if not macd.empty else "N/A"
        latest_signal = signal.iloc[-1] if not signal.empty else "N/A"

        # ATR
        high_low = recent_data['High'] - recent_data['Low']
        high_close = abs(recent_data['High'] - recent_data['Close'].shift())
        low_close = abs(recent_data['Low'] - recent_data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=1).mean()
        latest_atr = atr.iloc[-1] if not atr.empty else "N/A"

        # 布林带
        sma20 = recent_data['Close'].rolling(window=20, min_periods=1).mean()
        std20 = recent_data['Close'].rolling(window=20, min_periods=1).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        latest_upper = upper_band.iloc[-1] if not upper_band.empty else "N/A"
        latest_lower = lower_band.iloc[-1] if not lower_band.empty else "N/A"
        latest_close = recent_data['Close'].iloc[-1]

        # 添加更多技术指标
        # 20日移动平均线
        ma20 = recent_data['Close'].rolling(window=20, min_periods=1).mean()
        latest_ma20 = ma20.iloc[-1] if not ma20.empty else "N/A"
        
        # 50日移动平均线
        ma50 = recent_data['Close'].rolling(window=50, min_periods=1).mean()
        latest_ma50 = ma50.iloc[-1] if not ma50.empty else "N/A"
        
        # 200日移动平均线
        ma200 = recent_data['Close'].rolling(window=200, min_periods=1).mean()
        latest_ma200 = ma200.iloc[-1] if not ma200.empty else "N/A"

        # 14日威廉指标
        highest_high = recent_data['High'].rolling(window=14, min_periods=1).max()
        lowest_low = recent_data['Low'].rolling(window=14, min_periods=1).min()
        williams_r = ((highest_high - recent_data['Close']) / (highest_high - lowest_low)) * -100
        latest_williams_r = williams_r.iloc[-1] if not williams_r.empty else "N/A"

        # 相对强弱指标和成交量
        volume_sma = recent_data['Volume'].rolling(window=20, min_periods=1).mean()
        latest_volume_sma = volume_sma.iloc[-1] if not volume_sma.empty else "N/A"

        # 格式化技术指标
        technical_indicators = f"""
【技术指标】
- RSI (14): {latest_rsi:.2f} if isinstance(latest_rsi, (int, float)) else latest_rsi
- MACD: {latest_macd:.2f} / Signal: {latest_signal:.2f}
- ATR (14): {latest_atr:.2f}
- 布林带上轨: {latest_upper:.2f} / 下轨: {latest_lower:.2f} / 当前价: {latest_close:.2f}
- 移动平均线: MA20: {latest_ma20:.2f}, MA50: {latest_ma50:.2f}, MA200: {latest_ma200:.2f}
- 威廉指标 (14): {latest_williams_r:.2f}
- 平均成交量 (20日): {latest_volume_sma:,.0f}
"""

        # 格式化历史数据（只显示最近20天以节省token）
        hist_lines = []
        for idx, row in recent_data.tail(20).iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            hist_lines.append(f"{date_str}: 收盘价 {row.get('Close', 'N/A'):.2f}, 成交量 {row.get('Volume', 'N/A'):,.0f}")
        hist_summary = "\n".join(hist_lines)
    else:
        hist_summary = "无历史数据"
        technical_indicators = ""
    
    # 构建完整提示词
    prompt = f"""你是一位专业的短期股票分析师。请基于以下信息对股票进行结构化分析，并给出短期投资建议（1-4周内）。

股票代码: {symbol}
公司名称: {info.get('longName', 'N/A')}
行业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}

【基本面指标】
- 市值: {market_cap_str}
- 市盈率 (TTM): {pe_ratio_str} / 预期PE: {forward_pe_str}
- PEG比率: {peg_ratio_str} / 市净率: {pb_ratio_str}
- 利润率: {profit_margin_str} / ROE: {roe_str}
- 营收增长: {revenue_growth_str} / 盈利增长: {earnings_growth_str}
- 股息收益率: {dividend_yield_str}
- Beta系数: {beta_str} / 52周涨跌: {week52_change_str}
- 分析师目标价: {target_mean_price_str}
- 成交量: {volume_str} / 流通股本: {float_shares_str}
- 卖空比率: {short_ratio_str}

【技术面分析】
- 符合的交易策略: {', '.join(strategies) if strategies else '无'}
{technical_indicators}

【最近 20 天历史数据】
{hist_summary}

【分析要求】

你必须使用结构化的分析方法（Chain of Thought）：

第一步：数据梳理与观察 (Data Review)
- 观察当前价格相对于各关键移动平均线的位置 (MA20, MA50, MA200)
- 观察RSI是否处于超买（>70）或超卖（<30）区域
- 观察MACD是否出现金叉或死叉信号
- 观察价格在布林带中的位置（上轨、中轨、下轨）
- 观察成交量是否异常放大或缩小

第二步：趋势判定 (Trend Analysis)  
- 确定短期（20日）、中期（50日）、长期（200日）趋势
- 根据移动平均线的排列顺序判定趋势（多头排列/空头排列/震荡）
- 分析价格相对于均线的位置关系

第三步：技术指标确认 (Indicator Confirmation)
- 解读RSI数值：30以下为超卖可能反弹，70以上为超买可能回调，30-70为中性区域
- 解读MACD：DIF线上穿DEA线为金叉（看涨信号），下穿为死叉（看跌信号）
- 解读布林带：价格触及上轨可能回调，触及下轨可能反弹，收窄后突破方向重要
- 解读威廉指标：-20以上为超买区，-80以下为超卖区

第四步：基本面评估 (Fundamental Assessment)
- 评估市盈率是否合理（对比行业平均、历史平均）
- 评估PEG比率（1以下可能被低估，1以上可能被高估）
- 评估ROE和利润率是否健康
- 评估营收和盈利增长趋势

第五步：风险识别 (Risk Identification)
- 识别技术面风险：指标背离、趋势反转信号
- 识别基本面风险：高PE、负增长、高负债等
- 识别市场风险：Beta系数、市场情绪（卖空比率）

第六步：综合判断 (Synthesis)
- 综合技术面和基本面信息
- 判断短期（1-4周）走势
- 形成投资建议

【具体分析维度】
1. 综合评分（1-10分）：
   - 技术面评分（1-10分）：基于价格走势、成交量变化、技术指标等
   - 基本面评分（1-10分）：基于市值、市盈率、行业地位等
   - 综合评分：技术面和基本面的加权平均

2. 价格趋势分析：
   - 100天内的整体趋势（上升/下降/震荡）
   - 关键支撑位和阻力位（参考最近的高低点、整数关口、均线位置）
   - 近期价格波动特征

3. 成交量分析：
   - 近期成交量变化趋势
   - 成交量与价格的关系（量价配合/背离）
   - 是否有异常放量或缩量（突破时需要量能配合）

4. 技术形态分析：
   - RSI指标分析（超买/超卖状态、与价格的背离）
   - MACD金叉/死叉判断、柱状线变化趋势
   - ATR反映的波动性及对止损的指导意义
   - 布林带位置（是否接近上轨/下轨、带宽变化）
   - 威廉指标分析（超买超卖状态）
   - 均线系统状况（多头/空头排列、价格与均线关系）
   - 动量指标分析

5. 基本面评估：
   - 市盈率（TTM和预期）是否合理，PEG比率是否健康
   - 利润率、ROE等盈利能力指标分析
   - 营收增长和盈利增长趋势
   - 股息收益率和派息政策
   - Beta系数反映的波动性
   - 分析师目标价与当前价格的差距
   - 卖空比率反映的市场情绪
   - 行业地位和竞争优势
   - 财务健康状况

6. 风险评估：
   - Beta系数反映的市场系统性风险
   - 卖空比率反映的市场情绪风险
   - ATR反映的波动性风险
   - 技术面风险（指标背离、趋势反转信号）
   - 基本面风险（高估值、盈利下滑、财务恶化）
   - 主要风险点（市场风险、行业风险、个股风险）
   - 风险等级（低/中/高）
   - 止损和止盈建议（严格止损，建议使用ATR的2倍作为止损）

7. 投资建议（以短期投资为主）：
   - 明确建议：强烈买入/买入/持有/卖出/强烈卖出
   - 建议理由：基于以上结构化分析的详细解释，重点关注短期走势
   - 建议仓位：建议投入资金比例（短期仓位）
   - 建议持仓周期：短期（1-7天）/短期（1-2周）/中期（2-4周）
   - 如果建议是买入或强烈买入，必须给出：
     * 建议买入价位：具体的买入价格区间或价格点
     * 建议卖出价位：具体的卖出目标价格（短期目标）
     * 止损价位：具体的止损价格（严格止损）

8. 短期走势预测（1-4周）：
   - 未来1-2周的价格走势判断
   - 关键突破点和回调点
   - 交易时机建议

【输出格式要求】
请严格按照以下格式输出分析结果，不要使用 Markdown 格式（如 #### **1. 或 - **）：

数据梳理与观察：
[第一步分析内容]

趋势判定：
[第二步分析内容]

技术指标确认：
[第三步分析内容]

基本面评估：
[第四步分析内容]

风险识别：
[第五步分析内容]

综合判断：
[第六步分析内容]

综合评分：技术面 [X]/10，基本面 [X]/10，综合 [X]/10

价格趋势分析：
[分析内容]

成交量分析：
[分析内容]

技术形态分析：
[分析内容]

基本面评估：
[分析内容]

风险评估：
[分析内容]

投资建议：
[分析内容]
（如果是买入建议，请包含：建议买入价位、建议卖出价位、止损价位）

短期走势预测（1-4周）：
[分析内容]

请以专业、详细的语言给出分析结果，控制在 1000 字以内。"""

    return prompt


def _call_iflow_api(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> tuple[Optional[str], Optional[str]]:
    """
    调用心流 API

    Args:
        prompt: 分析提示词
        model_name: 要使用的模型名称

    Returns:
        (API 返回的文本内容, 实际使用的模型名称) 的元组，如果调用失败返回 (None, None)
    """
    headers = {
        "Authorization": f"Bearer {IFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 800,
        "temperature": 0.0,
        "top_p": 0.7
    }

    try:
        response = requests.post(
            IFLOW_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        # 提取返回的文本和模型名称
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            model = result.get('model', model_name)  # 从响应中提取 model，如果没有则使用默认值
            return content, model
        else:
            return None, None

    except requests.exceptions.RequestException as e:
        print(f" - [AI分析] API 调用失败: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f" - [AI分析] 解析 API 响应失败: {e}")
        return None, None


def batch_analyze_stocks(stocks_data: List[Dict], model: str = 'iflow-rome-30ba3b') -> List[Dict]:
    """
    批量分析多支股票
    
    Args:
        stocks_data: 股票数据列表
        model: 要使用的AI模型名称
    
    Returns:
        添加了 AI 分析结果的股票数据列表
    """
    print(f"\n--- 开始 AI 综合分析 ({len(stocks_data)} 支股票, 模型: {model}) ---")
    
    for i, stock in enumerate(stocks_data):
        print(f"AI 分析进度: [{i+1}/{len(stocks_data)}] {stock['symbol']}...", end='')
        
        ai_result = analyze_stock_with_ai(stock, model=model)
        
        if ai_result:
            stock['ai_analysis'] = ai_result
        else:
            stock['ai_analysis'] = None
    
    print(f"\n--- AI 分析完成 ---")
    return stocks_data
