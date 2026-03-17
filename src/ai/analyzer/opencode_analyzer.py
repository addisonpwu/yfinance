"""
OpenCode AI 分析器

使用 OpenCode API 进行股票分析
- API 端点: https://opencode.ai/zen/go/v1/chat/completions
- 环境变量: OPENCODE_API_KEY
- 支持模型: glm-5
"""

from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import re

from src.core.models.entities import AIAnalysisResult
from src.data.cache.cache_service import OptimizedCache
from src.config.settings import config_manager
from src.config.constants import VIX_LOW, VIX_NORMAL, VIX_HIGH
from src.utils.logger import get_ai_logger

# 尝试导入 OpenAI SDK
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


class OpenCodeAIAnalyzer:
    """
    OpenCode AI 分析器
    
    使用 OpenCode API 进行股票分析，基于 OpenAI SDK 兼容格式
    
    使用示例:
        analyzer = OpenCodeAIAnalyzer()
        result = analyzer.analyze(stock_data, hist)
    """
    
    # AI 分析缓存子目录
    AI_CACHE_SUBDIR = "ai_analysis_opencode"
    
    # API 配置
    API_URL = "https://opencode.ai/zen/go/v1"
    
    # 支持的模型
    DEFAULT_MODEL = "glm-5"
    AVAILABLE_MODELS = ["glm-5"]
    
    # Few-shot 学习案例（复用 iFlow 的案例）
    FEW_SHOT_EXAMPLES = """
【历史成功预测案例】

案例1: 超卖反弹型
股票: AAPL (2024-01-15)
- RSI=32, 价格低于MA50约8%
- 成交量萎缩30%
- MACD柱状线收窄，即将金叉
- AI判断: 超卖反弹，建议买入
- 实际结果: 2周后上涨11%
- 关键特征: RSI超卖 + 量价背离 + MACD金叉信号

案例2: 趋势延续型  
股票: NVDA (2024-03-01)
- 价格站上所有均线，多头排列
- RSI=58，处于强势区但未超买
- 成交量较20日均量放大40%
- MACD持续走高，柱状线扩大
- AI判断: 趋势延续，建议买入
- 实际结果: 4周后上涨28%
- 关键特征: 多头排列 + 量能配合 + MACD强势

案例3: 风险警示型
股票: XYZ (2024-02-20)
- RSI=78, 明显超买
- 价格与MACD出现顶背离
- 成交量萎缩但价格上涨
- AI判断: 高位风险，建议观望
- 实际结果: 2周后下跌15%
- 关键特征: RSI超买 + 顶背离 + 量价背离

案例4: 突破确认型
股票: TSM (2024-04-10)
- 布林带收窄后突破上轨
- 成交量放大至20日均量的2.5倍
- RSI从45快速升至62
- 价格突破前期高点
- AI判断: 突破有效，建议买入
- 实际结果: 3周后上涨18%
- 关键特征: 布林带突破 + 放量 + RSI走强
"""
    
    def __init__(self, enable_cache: bool = False, enable_streaming: bool = False):
        """
        初始化 OpenCode AI 分析器
        
        Args:
            enable_cache: 是否启用缓存（默认禁用）
            enable_streaming: 是否启用流式响应（暂不支持）
        """
        self.api_key = os.environ.get("OPENCODE_API_KEY", "")
        self.enable_streaming = enable_streaming
        self.logger = get_ai_logger()
        
        # 加载配置
        config = config_manager.get_config()
        self.config = config
        
        # 缓存服务（默认禁用）
        self.cache_service = OptimizedCache(enabled=enable_cache)
        
        # 从配置读取模型
        if config.ai.providers and hasattr(config.ai.providers, 'opencode') and config.ai.providers.opencode:
            self.default_model = config.ai.providers.opencode.default_model or self.DEFAULT_MODEL
            self.available_models = config.ai.providers.opencode.available_models or self.AVAILABLE_MODELS
        else:
            self.default_model = self.DEFAULT_MODEL
            self.available_models = self.AVAILABLE_MODELS
        
        # 初始化 OpenAI 客户端
        self.client = None
        if HAS_OPENAI and self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.API_URL
            )
            self.logger.info(f"OpenCode AI 分析器初始化成功，模型: {self.default_model}")
        else:
            if not HAS_OPENAI:
                self.logger.warning("OpenAI SDK 未安装，请运行: pip install openai")
            if not self.api_key:
                self.logger.warning("未找到 OPENCODE_API_KEY 环境变量")
    
    def analyze(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        **kwargs
    ) -> Optional[AIAnalysisResult]:
        """
        分析单只股票
        
        Args:
            stock_data: 股票数据字典
            hist: 历史数据 DataFrame
            **kwargs: 额外参数
                - model: 模型名称（仅支持 glm-5）
                - interval: 数据间隔
        
        Returns:
            AIAnalysisResult 或 None
        """
        import time as time_module
        
        # 检查 API Key
        if not self.api_key:
            self.logger.warning("未找到 OPENCODE_API_KEY 环境变量，跳过 AI 分析")
            return None
        
        # 检查 OpenAI SDK
        if not HAS_OPENAI:
            self.logger.warning("OpenAI SDK 未安装，跳过 OpenCode AI 分析")
            return None
        
        # 检查客户端
        if not self.client:
            self.logger.warning("OpenCode 客户端未初始化，跳过分析")
            return None
        
        interval = kwargs.get('interval', '1d')
        model = kwargs.get('model', self.default_model)
        
        # OpenCode 只支持 glm-5 模型
        if model not in self.available_models:
            self.logger.warning(f"OpenCode 不支持模型 '{model}'，使用默认模型 '{self.default_model}'")
            model = self.default_model
        
        symbol = stock_data.get('symbol', 'Unknown')
        start_time = time_module.time()
        
        self.logger.info(f"[OpenCode] 开始分析 {symbol}，模型: {model}")
        
        try:
            # 构建分析提示词
            prompt = self._build_analysis_prompt(stock_data, hist)
            
            self.logger.info(f"[OpenCode] {symbol} - 调用 API 中...")
            
            # 调用 API
            response, model_used = self._call_opencode_api(prompt, model)
            
            if not response:
                self.logger.warning(f"[OpenCode] {symbol} - API 返回空结果")
                return None
            
            # 提取方向和置信度
            direction, confidence = self._extract_direction_and_confidence(response)
            
            elapsed = time_module.time() - start_time
            self.logger.info(f"[OpenCode] {symbol} - 分析完成: 方向={direction}, 置信度={confidence:.0%}, 模型={model_used}, 耗时={elapsed:.1f}秒")
            
            return AIAnalysisResult(
                summary=response,
                confidence=confidence,
                model_used=f"opencode/{model_used}",
                detailed_analysis={
                    'direction': direction,
                    'provider': 'opencode'
                }
            )
            
        except Exception as e:
            self.logger.error(f"[OpenCode] 分析 {symbol} 时出错: {e}")
            return None
    
    def _call_opencode_api(self, prompt: str, model: str) -> Tuple[str, str]:
        """
        调用 OpenCode API
        
        Args:
            prompt: 提示词
            model: 模型名称
        
        Returns:
            (响应文本, 使用的模型名称)
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的短期股票分析师，擅长1-4周内的短线交易分析。请基于技术指标和市场数据给出专业、客观的投资建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4096,
                timeout=120.0  # 2 分钟超时
            )
            
            content = response.choices[0].message.content
            model_used = response.model or model
            
            return content, model_used
            
        except Exception as e:
            self.logger.error(f"OpenCode API 调用失败: {e}")
            return None, model
    
    def _build_analysis_prompt(self, stock_data: Dict, hist: pd.DataFrame) -> str:
        """构建分析提示词"""
        symbol = stock_data.get('symbol', 'N/A')
        info = stock_data.get('info', {})
        strategies = stock_data.get('strategies', [])
        
        # 格式化基本面信息
        fundamentals = self._format_fundamentals(info)
        
        # 格式化技术指标
        tech_indicators = self._get_technical_indicators(hist)
        
        # 历史数据摘要
        hist_summary = self._get_hist_summary(hist)
        
        # 格式化新闻
        news_section = self._format_news(stock_data.get('news', []))
        
        prompt = f"""你是一位专业的短期股票分析师（擅长1-4周内的短线交易）。请基于以下数据给出综合投资建议。

{self.FEW_SHOT_EXAMPLES}

════════════════════════════════════════════════════════════

【股票基本信息】
代码: {symbol}
名称: {info.get('longName', 'N/A')}
行业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}
符合策略: {', '.join(strategies) if strategies else '无'}

{fundamentals}

{tech_indicators}

{news_section}

【历史数据分析】
{hist_summary}

════════════════════════════════════════════════════════════

【重要分析框架 - 请严格按照此框架分析】

## 一、市场环境评估（权重30%）
根据VIX和市场风险等级评估当前市场环境：
- VIX < 15: 低波动市场，可适度激进
- VIX 15-25: 正常波动市场，谨慎操作
- VIX > 25: 高波动市场，降低仓位

## 二、技术面分析（权重40%）
必须分析以下关键信号：
1. RSI信号：RSI>70超买可能回调，RSI<30超卖可能反弹，RSI在40-60最为健康
2. MACD信号：金叉（DIF上穿DEA）是买入信号，死叉是卖出信号
3. 均线信号：价格站上均线看多，跌破均线看空；多头排列（MA5>MA10>MA20）看多
4. 布林带信号：价格突破上轨可能回调，突破下轨可能反弹
5. 量价配合：价涨量增健康，价涨量缩需警惕

## 三、新闻影响分析（权重20%）
分析新闻对短期走势的影响：
- 正面新闻：业绩增长、产品发布、获得订单等 → 看多
- 负面新闻：业绩下滑、诉讼、减持等 → 看空
- 中性新闻：无实质影响 → 维持原有趋势

## 四、风险评估（权重10%）
必须给出：
- 最大风险点
- 建议止损价位（必须具体数值）
- 建议仓位（轻仓/半仓/重仓）

════════════════════════════════════════════════════════════

【输出格式要求 - 必须严格遵守】

1. 综合评分
   技术面评分: X/10
   基本面评分: X/10  
   综合评分: X/10

2. 短期走势判断
   方向: [看涨/看跌/中性]
   置信度: [高(>70%)/中(50-70%)/低(<50%)]
   主要驱动因素: [不超过50字]

3. 投资建议（必须全部填写）
   建议: [强烈买入/买入/持有/卖出/强烈卖出]
   理由: [不超过150字]
   买入价位: [具体数值或"现价"]
   目标价位: [具体数值]
   止损价位: [具体数值]

4. 风险提示
   最大风险点: [具体描述]
   建议仓位: [轻仓(<20%)/半仓(20-40%)/重仓(40-60%)]

请严格按照上述格式输出，AI分析结果将用于实盘交易决策，请务必严谨客观。
"""
        return prompt
    
    def _format_fundamentals(self, info: Dict) -> str:
        """格式化基本面信息"""
        def fmt(val, suffix="", decimals=2):
            if isinstance(val, (int, float)) and val > 0:
                return f"{val:.{decimals}f}{suffix}"
            return "N/A"
        
        return f"""【基本面指标】
- 市值: {fmt(info.get('marketCap'), suffix='', decimals=0) if info.get('marketCap') else 'N/A'}
- 市盈率: {fmt(info.get('trailingPE'))}
- PEG: {fmt(info.get('pegRatio'))}
- 市净率: {fmt(info.get('priceToBook'))}
- ROE: {fmt(info.get('returnOnEquity'), '%') if info.get('returnOnEquity') else 'N/A'}
- 利润率: {fmt(info.get('profitMargins'), '%') if info.get('profitMargins') else 'N/A'}
- 营收增长: {fmt(info.get('revenueGrowth'), '%') if info.get('revenueGrowth') else 'N/A'}
- Beta: {fmt(info.get('beta'))}"""
    
    def _get_technical_indicators(self, hist: pd.DataFrame) -> str:
        """获取技术指标"""
        if hist is None or hist.empty:
            return "【技术指标】\n无数据"
        
        try:
            close = hist['Close'].iloc[-1]
            
            # 移动平均线
            ma_values = {}
            for period in [5, 10, 20, 50, 200]:
                col = f'MA_{period}'
                if col in hist.columns:
                    ma_values[period] = hist[col].iloc[-1]
            
            # RSI
            rsi = hist['RSI_14'].iloc[-1] if 'RSI_14' in hist.columns else None
            
            # MACD
            macd = hist['MACD'].iloc[-1] if 'MACD' in hist.columns else None
            macd_signal = hist['MACD_Signal'].iloc[-1] if 'MACD_Signal' in hist.columns else None
            macd_hist = hist['MACD_Hist'].iloc[-1] if 'MACD_Hist' in hist.columns else None
            
            # 布林带
            bb_upper = hist['BB_Upper'].iloc[-1] if 'BB_Upper' in hist.columns else None
            bb_lower = hist['BB_Lower'].iloc[-1] if 'BB_Lower' in hist.columns else None
            bb_middle = hist['BB_Middle'].iloc[-1] if 'BB_Middle' in hist.columns else None
            
            # ATR
            atr = hist['ATR_14'].iloc[-1] if 'ATR_14' in hist.columns else None
            
            # 成交量
            volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else None
            volume_ma = hist['Volume_MA_20'].iloc[-1] if 'Volume_MA_20' in hist.columns else None
            
            lines = ["【技术指标】"]
            
            # 价格信息
            lines.append(f"当前价格: {close:.2f}")
            
            # 均线
            if ma_values:
                lines.append("\n移动平均线:")
                for period, value in sorted(ma_values.items()):
                    position = "上方" if close > value else "下方"
                    lines.append(f"  MA{period}: {value:.2f} (价格在{position})")
            
            # RSI
            if rsi is not None:
                rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "正常"
                lines.append(f"\nRSI(14): {rsi:.1f} ({rsi_status})")
            
            # MACD
            if macd is not None and macd_signal is not None:
                macd_status = "金叉" if macd > macd_signal else "死叉"
                lines.append(f"\nMACD: {macd:.4f}")
                lines.append(f"信号线: {macd_signal:.4f} ({macd_status})")
                if macd_hist is not None:
                    lines.append(f"柱状线: {macd_hist:.4f}")
            
            # 布林带
            if bb_upper is not None and bb_lower is not None:
                bb_width = (bb_upper - bb_lower) / bb_middle * 100 if bb_middle else 0
                bb_position = (close - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50
                lines.append(f"\n布林带:")
                lines.append(f"  上轨: {bb_upper:.2f}")
                lines.append(f"  中轨: {bb_middle:.2f}" if bb_middle else "  中轨: N/A")
                lines.append(f"  下轨: {bb_lower:.2f}")
                lines.append(f"  带宽: {bb_width:.1f}%")
                lines.append(f"  价格位置: {bb_position:.1f}%")
            
            # ATR
            if atr is not None:
                atr_pct = atr / close * 100
                lines.append(f"\nATR(14): {atr:.2f} ({atr_pct:.2f}%)")
            
            # 成交量
            if volume is not None:
                vol_ratio = volume / volume_ma if volume_ma else 1
                lines.append(f"\n成交量: {volume:,.0f}")
                if volume_ma:
                    lines.append(f"量比: {vol_ratio:.2f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.debug(f"获取技术指标失败: {e}")
            return "【技术指标】\n数据解析失败"
    
    def _get_hist_summary(self, hist: pd.DataFrame) -> str:
        """获取历史数据摘要"""
        if hist is None or hist.empty:
            return "无历史数据"
        
        try:
            close = hist['Close']
            
            # 计算涨跌幅
            change_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) >= 6 else 0
            change_20d = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) >= 21 else 0
            change_60d = (close.iloc[-1] - close.iloc[-61]) / close.iloc[-61] * 100 if len(close) >= 61 else 0
            
            # 近期高低点
            high_20d = hist['High'].tail(20).max()
            low_20d = hist['Low'].tail(20).min()
            
            # 当前价格位置
            current = close.iloc[-1]
            price_position = (current - low_20d) / (high_20d - low_20d) * 100 if high_20d != low_20d else 50
            
            return f"""近20日波动区间: {low_20d:.2f} - {high_20d:.2f}
当前价格位置: {price_position:.1f}% (相对20日区间)

涨跌幅:
- 5日: {change_5d:+.2f}%
- 20日: {change_20d:+.2f}%
- 60日: {change_60d:+.2f}%"""
            
        except Exception as e:
            self.logger.debug(f"获取历史摘要失败: {e}")
            return "数据解析失败"
    
    def _format_news(self, news_list: List[Dict]) -> str:
        """格式化新闻数据"""
        if not news_list:
            return ""
        
        # 按发布时间排序（最新的在前）
        sorted_news = sorted(
            news_list, 
            key=lambda x: x.get('published', ''), 
            reverse=True
        )[:5]  # 最多显示5条
        
        lines = ["【近期新闻】（按时间倒序）"]
        
        for i, item in enumerate(sorted_news, 1):
            title = item.get('title', 'N/A')
            published = item.get('published', '')
            publisher = item.get('publisher', '')
            
            lines.append(f"{i}. [{published}] {title}")
            if publisher:
                lines.append(f"   来源: {publisher}")
        
        # 添加分析指引
        lines.append("")
        lines.append("【新闻分析指引】")
        lines.append("- 关注发布时间越近的新闻，影响力越大")
        lines.append("- 业绩公告、产品发布、重大合同为利好")
        lines.append("- 业绩亏损、诉讼、减持、监管处罚为利空")
        
        return "\n".join(lines)
    
    def _extract_direction_and_confidence(self, text: str) -> Tuple[str, float]:
        """从分析结果中提取方向和置信度"""
        direction = "中性"
        confidence = 0.5
        
        # 优先从"短期走势判断"部分提取方向
        short_term_section = ""
        if "短期走势判断" in text:
            match = re.search(r'短期走势判断[：:\s]*(.*?)(?=\n[═\-\s]{10,}|\n【|$)', text, re.DOTALL)
            if match:
                short_term_section = match.group(1)
        
        # 从短期走势判断部分提取方向
        if short_term_section:
            dir_match = re.search(r'方向[：:]\s*(看涨|看跌|中性|上升|下降|震荡)', short_term_section)
            if dir_match:
                dir_value = dir_match.group(1)
                if dir_value in ['看涨', '上升']:
                    direction = "看涨"
                    confidence = 0.7
                elif dir_value in ['看跌', '下降']:
                    direction = "看跌"
                    confidence = 0.7
                elif dir_value in ['中性', '震荡']:
                    direction = "中性"
                    confidence = 0.5
        
        # 如果没有找到，尝试从"投资建议"部分提取
        if direction == "中性" and "投资建议" in text:
            advice_match = re.search(r'建议[：:]\s*(买入|卖出|持有|观望|增持|减持)', text)
            if advice_match:
                advice = advice_match.group(1)
                if advice in ['买入', '增持']:
                    direction = "看涨"
                    confidence = 0.7
                elif advice in ['卖出', '减持']:
                    direction = "看跌"
                    confidence = 0.7
                elif advice in ['持有', '观望']:
                    direction = "中性"
                    confidence = 0.5
        
        # 如果还是没有找到，使用全文搜索
        if direction == "中性":
            if re.search(r'看涨|强烈买入|买入|上升', text):
                direction = "看涨"
                confidence = 0.7
            elif re.search(r'看跌|强烈卖出|卖出|下降', text):
                direction = "看跌"
                confidence = 0.7
            elif re.search(r'中性|持有|震荡', text):
                direction = "中性"
                confidence = 0.5
        
        # 提取置信度
        conf_match = re.search(r'置信度[：:]\s*(高|中|低)', text)
        if conf_match:
            level = conf_match.group(1)
            if level == '高':
                confidence = min(confidence + 0.2, 0.95)
            elif level == '低':
                confidence = max(confidence - 0.2, 0.3)
        
        return direction, confidence
