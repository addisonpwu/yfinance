# ai_analyzer.py

import os
import json
import requests
import pandas as pd
from typing import Dict, List, Optional

# 心流 API 配置
IFLOW_API_URL = "https://apis.iflow.cn/v1/chat/completions"
IFLOW_API_KEY = os.environ.get("IFLOW_API_KEY", "")
MODEL_NAME = "deepseek-v3.2"


def analyze_stock_with_ai(stock_data: Dict, hist: pd.DataFrame = None, interval: str = '1d') -> Optional[Dict]:
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

    Returns:
        包含 AI 分析结果的字典，或 None（如果分析失败）
    """
    if not IFLOW_API_KEY:
        print(f" - [AI分析] 未找到 IFLOW_API_KEY 环境变量，跳过 AI 分析")
        return None

    # 如果没有提供 hist，尝试从缓存读取
    if hist is None:
        hist = _load_stock_data_from_cache(stock_data['symbol'], stock_data.get('market', 'HK'), interval)
        if hist is None or hist.empty:
            print(f" - [AI分析] 无法加载 {stock_data['symbol']} 的历史数据，跳过 AI 分析")
            return None
    
    # 构建分析提示词
    prompt = _build_analysis_prompt(stock_data, hist)

    try:
        # 调用心流 API
        response, model_used = _call_iflow_api(prompt)

        if response:
            print(f" - [AI分析] 成功完成 {stock_data['symbol']} 的 AI 分析")
            return {
                'summary': response,
                'model_used': model_used
            }
        else:
            print(f" - [AI分析] {stock_data['symbol']} 的 AI 分析失败")
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
    构建用于股票分析的提示词
    """
    symbol = stock_data.get('symbol', 'N/A')
    info = stock_data.get('info', {})
    strategies = stock_data.get('strategies', [])
    analyzed_news = stock_data.get('analyzed_news', [])
    
    # 格式化基本信息
    market_cap = info.get('marketCap')
    market_cap_str = f"{market_cap / 1e8:.2f} 億" if isinstance(market_cap, (int, float)) else "N/A"

    pe_ratio = info.get('trailingPE')
    pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

    volume = info.get('volume')
    volume_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) else "N/A"

    # 格式化历史数据（最近 100 天）
    hist_summary = ""
    if hist is not None and not hist.empty:
        recent_data = hist.tail(100)
        hist_lines = []
        for idx, row in recent_data.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            hist_lines.append(f"{date_str}: 收盘价 {row.get('Close', 'N/A'):.2f}, 成交量 {row.get('Volume', 'N/A'):,.0f}")
        hist_summary = "\n".join(hist_lines)
    else:
        hist_summary = "无历史数据"
    
    # 构建完整提示词
    prompt = f"""你是一位专业的短期股票分析师。请基于以下信息对股票进行综合分析，并给出短期投资建议（1-4周内）。

股票代码: {symbol}
公司名称: {info.get('longName', 'N/A')}
行业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}

【基本面指标】
- 市值: {market_cap_str}
- 市盈率 (PE): {pe_ratio_str}
- 成交量: {volume_str}
- 流通股本: {info.get('floatShares', 'N/A')}

【技术面分析】
- 符合的交易策略: {', '.join(strategies) if strategies else '无'}

【最近 100 天历史数据】
{hist_summary}

【分析要求】
请从以下几个维度进行深入分析，重点关注短期走势：

1. 综合评分（1-10分）：
   - 技术面评分（1-10分）：基于价格走势、成交量变化、技术指标等
   - 基本面评分（1-10分）：基于市值、市盈率、行业地位等
   - 综合评分：技术面和基本面的加权平均

2. 价格趋势分析：
   - 100天内的整体趋势（上升/下降/震荡）
   - 关键支撑位和阻力位
   - 近期价格波动特征

3. 成交量分析：
   - 近期成交量变化趋势
   - 成交量与价格的关系
   - 是否有异常放量或缩量

4. 技术形态分析：
   - 是否存在明显的技术形态（如双底、头肩顶等）
   - 均线系统状况
   - 动量指标分析

5. 基本面评估：
   - 市盈率是否合理
   - 行业地位和竞争优势
   - 财务健康状况

6. 风险评估：
   - 主要风险点（市场风险、行业风险、个股风险）
   - 风险等级（低/中/高）
   - 止损和止盈建议（严格止损）

7. 投资建议（以短期投资为主）：
   - 明确建议：强烈买入/买入/持有/卖出/强烈卖出
   - 建议理由：基于以上分析的详细解释，重点关注短期走势
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

请以专业、详细的语言给出分析结果，控制在 800 字以内。"""

    return prompt


def _call_iflow_api(prompt: str) -> tuple[Optional[str], Optional[str]]:
    """
    调用心流 API

    Args:
        prompt: 分析提示词

    Returns:
        (API 返回的文本内容, 实际使用的模型名称) 的元组，如果调用失败返回 (None, None)
    """
    headers = {
        "Authorization": f"Bearer {IFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
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
            model = result.get('model', MODEL_NAME)  # 从响应中提取 model，如果没有则使用默认值
            return content, model
        else:
            return None, None

    except requests.exceptions.RequestException as e:
        print(f" - [AI分析] API 调用失败: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f" - [AI分析] 解析 API 响应失败: {e}")
        return None, None


def batch_analyze_stocks(stocks_data: List[Dict]) -> List[Dict]:
    """
    批量分析多支股票
    
    Args:
        stocks_data: 股票数据列表
    
    Returns:
        添加了 AI 分析结果的股票数据列表
    """
    print(f"\n--- 开始 AI 综合分析 ({len(stocks_data)} 支股票) ---")
    
    for i, stock in enumerate(stocks_data):
        print(f"AI 分析进度: [{i+1}/{len(stocks_data)}] {stock['symbol']}...", end='')
        
        ai_result = analyze_stock_with_ai(stock)
        
        if ai_result:
            stock['ai_analysis'] = ai_result
        else:
            stock['ai_analysis'] = None
    
    print(f"\n--- AI 分析完成 ---")
    return stocks_data
