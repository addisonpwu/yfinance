"""
Google Gemini API 分析器

使用 Google Gemini API 进行股票技术分析，支持：
- 流式响应
- 多模型分析投票
- 预测追踪
- 缓存支持
- 多模态分析（文本为主）

官方文档: https://ai.google.dev/gemini-api/docs/quickstart
"""

from typing import Dict, Optional, List, Union
import pandas as pd
import numpy as np
from src.core.models.entities import AIAnalysisResult
from src.data.cache.cache_service import OptimizedCache
from src.config.settings import config_manager
from src.utils.logger import get_ai_logger
import os
import sys
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re

# 导入共享的数据类和抽象基类
from src.ai.analyzer.iflow_analyzer import (
    AIAnalyzer,
    PredictionDirection,
    PredictionRecord,
    MultiModelConsensus,
    PredictionTracker
)

# 尝试导入 Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None
    types = None


class GeminiAIAnalyzer(AIAnalyzer):
    """Google Gemini API 分析器实现"""
    
    # AI 分析缓存子目录（独立于其他分析器）
    AI_CACHE_SUBDIR = "ai_analysis_gemini"
    
    # 默认模型（将在 __init__ 中从配置读取）
    DEFAULT_MODEL = "gemini-2.5-flash"
    
    # 可用的 Gemini 模型列表（将在 __init__ 中从配置读取）
    AVAILABLE_MODELS = []
    
    # Few-shot 学习案例（复用现有案例）
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
    
    def __init__(self, enable_cache: bool = True, enable_streaming: bool = False):
        """
        初始化 Gemini AI 分析器
        
        Args:
            enable_cache: 是否启用缓存
            enable_streaming: 是否启用流式响应
        """
        # 检查 Google GenAI SDK
        if not HAS_GENAI:
            raise ImportError("需要安装 google-genai 包: pip install google-genai")
        
        # 获取 API Key
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        
        # 配置
        config = config_manager.get_config()
        # 禁用 AI 分析缓存
        self.cache_service = OptimizedCache(enabled=False)
        self.config = config
        self.logger = get_ai_logger()
        
        # 从配置读取模型列表（配置为唯一数据源）
        if config.ai.providers and hasattr(config.ai.providers, 'gemini') and config.ai.providers.gemini:
            self.DEFAULT_MODEL = config.ai.providers.gemini.default_model
            self.AVAILABLE_MODELS = config.ai.providers.gemini.available_models
        else:
            # 后备默认值
            self.DEFAULT_MODEL = "gemini-2.5-flash"
            self.AVAILABLE_MODELS = [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-3-flash-preview",
                "gemini-3-pro-preview"
            ]
        
        # 流式响应配置
        self.enable_streaming = enable_streaming
        self._use_color = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
        self._reasoning_color = "\033[90m" if self._use_color else ""
        self._reset_color = "\033[0m" if self._use_color else ""
        
        # 初始化 Gemini 客户端
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        
        # 预测追踪器
        self.prediction_tracker = PredictionTracker()
        
        # 多时间框架数据缓存
        self._multi_timeframe_cache: Dict[str, Dict] = {}
        
        # 市场情绪缓存
        self._market_sentiment_cache: Dict = {}
        self._sentiment_last_update: Optional[datetime] = None
    
    def analyze(self, stock_data: Dict, hist: pd.DataFrame, **kwargs) -> Optional[AIAnalysisResult]:
        """
        分析单只股票
        
        Args:
            stock_data: 股票数据字典
            hist: 历史数据 DataFrame
            **kwargs: 额外参数
                - interval: 数据间隔 (默认 '1d')
                - model: 模型名称 (默认 'gemini-2.5-flash')
                - use_multi_timeframe: 是否使用多时间框架 (默认 True)
                - enable_cache: 是否启用缓存 (默认 True)
        
        Returns:
            AIAnalysisResult 或 None
        """
        interval = kwargs.get('interval', '1d')
        model = kwargs.get('model', self.DEFAULT_MODEL)
        use_multi_timeframe = kwargs.get('use_multi_timeframe', True)
        enable_cache = kwargs.get('enable_cache', True)
        
        # 处理 'all' 模型选项
        if model == 'all':
            return self._analyze_with_all_models(stock_data, hist, interval, enable_cache)
        
        # 检查 API Key
        if not self.api_key:
            self.logger.warning("未找到 GEMINI_API_KEY 环境变量，跳过 AI 分析")
            return None
        
        # 初始化客户端（如果尚未初始化）
        if not self.client:
            self.client = genai.Client(api_key=self.api_key)
        
        # 尝试从缓存读取
        if enable_cache:
            cache_key = self._get_cache_key(stock_data, hist, interval, model)
            cached_result = self.cache_service.get_json(cache_key, self.AI_CACHE_SUBDIR)
            if cached_result:
                self.logger.info(f"从缓存获取 {stock_data.get('symbol', 'Unknown')} 的 Gemini AI 分析结果")
                return AIAnalysisResult(
                    summary=cached_result.get('summary', ''),
                    confidence=cached_result.get('confidence', 0.5),
                    model_used=cached_result.get('model_used', model)
                )
        
        try:
            # 分步分析流程
            analysis_result = self._step_by_step_analysis(stock_data, hist, model, use_multi_timeframe)
            
            if analysis_result:
                # 缓存结果
                if enable_cache:
                    cache_key = self._get_cache_key(stock_data, hist, interval, model)
                    self.cache_service.set_json(
                        cache_key,
                        {
                            'summary': analysis_result.summary,
                            'confidence': analysis_result.confidence,
                            'model_used': analysis_result.model_used
                        },
                        self.AI_CACHE_SUBDIR
                    )
                
                return analysis_result
            
        except Exception as e:
            self.logger.error(f"Gemini AI 分析失败: {e}")
            return None
        
        return None
    
    def _step_by_step_analysis(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        model: str,
        use_multi_timeframe: bool = True
    ) -> Optional[AIAnalysisResult]:
        """
        分步分析流程
        
        三步分析：
        1. 趋势判断（简单任务，准确率高）
        2. 关键价位识别
        3. 综合分析（结合前两步结果）
        """
        symbol = stock_data.get('symbol', 'Unknown')
        
        try:
            # 第一步：趋势判断
            trend_prompt = self._build_trend_prompt(stock_data, hist)
            trend_result = self._call_gemini_api(trend_prompt, model)
            
            if not trend_result:
                self.logger.warning(f"{symbol} 趋势分析失败，使用完整分析")
                return self._full_analysis(stock_data, hist, model)
            
            # 第二步：关键价位识别
            levels_prompt = self._build_levels_prompt(stock_data, hist, trend_result)
            levels_result = self._call_gemini_api(levels_prompt, model)
            
            # 第三步：综合分析
            final_prompt = self._build_final_prompt(
                stock_data, hist, trend_result, levels_result
            )
            final_result, model_used = self._call_gemini_api(final_prompt, model, return_model=True)
            
            if not final_result:
                return None
            
            # 提取方向和置信度
            direction, confidence = self._extract_direction_and_confidence(final_result)
            
            # 记录预测
            self._record_prediction(symbol, direction, confidence, model_used)
            
            return AIAnalysisResult(
                summary=final_result,
                confidence=confidence,
                model_used=model_used
            )
            
        except Exception as e:
            self.logger.error(f"分步分析失败: {e}")
            return self._full_analysis(stock_data, hist, model)
    
    def _full_analysis(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        model: str
    ) -> Optional[AIAnalysisResult]:
        """
        完整分析（不分步）
        """
        prompt = self._build_complete_prompt(stock_data, hist)
        result, model_used = self._call_gemini_api(prompt, model, return_model=True)
        
        if not result:
            return None
        
        direction, confidence = self._extract_direction_and_confidence(result)
        symbol = stock_data.get('symbol', 'Unknown')
        
        self._record_prediction(symbol, direction, confidence, model_used)
        
        return AIAnalysisResult(
            summary=result,
            confidence=confidence,
            model_used=model_used
        )
    
    def _call_gemini_api(
        self, 
        prompt: str, 
        model: str,
        return_model: bool = False,
        max_retries: int = 3
    ) -> Union[str, tuple]:
        """
        调用 Gemini API
        
        Args:
            prompt: 提示词
            model: 模型名称
            return_model: 是否返回模型名称
            max_retries: 最大重试次数
        
        Returns:
            响应文本，或 (响应文本, 模型名称) 元组
        """
        for attempt in range(max_retries):
            try:
                if self.enable_streaming:
                    # 流式响应
                    response = self.client.models.generate_content_stream(
                        model=model,
                        contents=prompt
                    )
                    result_text = ""
                    for chunk in response:
                        if chunk.text:
                            result_text += chunk.text
                            print(chunk.text, end="", flush=True)
                    print()  # 换行
                else:
                    # 标准响应
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                    result_text = response.text
                
                if return_model:
                    return (result_text, model)
                return result_text
                
            except Exception as e:
                self.logger.warning(f"Gemini API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Gemini API 调用最终失败: {e}")
                    if return_model:
                        return (None, model)
                    return None
        
        if return_model:
            return (None, model)
        return None
    
    def _build_trend_prompt(self, stock_data: Dict, hist: pd.DataFrame) -> str:
        """构建趋势判断提示词"""
        symbol = stock_data.get('symbol', 'Unknown')
        name = stock_data.get('name', '')
        
        # 获取技术指标摘要
        indicators = self._get_technical_indicators(hist)
        price_summary = self._get_price_summary(hist)
        
        prompt = f"""你是一位专业的股票技术分析师。请分析以下股票的趋势方向。

股票信息：
- 代码: {symbol}
- 名称: {name}

价格数据（最近20个交易日）：
{price_summary}

技术指标：
{indicators}

请用简洁的语言（100字以内）回答：
1. 当前趋势方向（上涨/下跌/横盘）
2. 趋势强度（强/中/弱）
3. 主要支撑因素

直接给出结论，无需详细解释。"""
        
        return prompt
    
    def _build_levels_prompt(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        trend_result: str
    ) -> str:
        """构建关键价位识别提示词"""
        symbol = stock_data.get('symbol', 'Unknown')
        
        # 计算支撑阻力位
        levels = self._calculate_support_resistance(hist)
        
        prompt = f"""基于趋势分析结果，识别关键价位。

股票代码: {symbol}
趋势分析: {trend_result}

计算的关键价位：
- 近期高点: {levels.get('recent_high', 'N/A')}
- 近期低点: {levels.get('recent_low', 'N/A')}
- 20日均线: {levels.get('ma20', 'N/A')}
- 50日均线: {levels.get('ma50', 'N/A')}
- 布林带上轨: {levels.get('bb_upper', 'N/A')}
- 布林带下轨: {levels.get('bb_lower', 'N/A')}

请用简洁的语言（100字以内）指出：
1. 上方关键阻力位
2. 下方关键支撑位
3. 当前价格位置判断

直接给出结论。"""
        
        return prompt
    
    def _build_final_prompt(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        trend_result: str,
        levels_result: str
    ) -> str:
        """构建综合分析提示词"""
        symbol = stock_data.get('symbol', 'Unknown')
        name = stock_data.get('name', '')
        
        # 获取完整的分析数据
        indicators = self._get_technical_indicators(hist)
        price_summary = self._get_price_summary(hist)
        fundamentals = self._format_fundamentals(stock_data)
        
        prompt = f"""你是一位资深股票分析师，请综合分析以下股票的投资价值。

{self.FEW_SHOT_EXAMPLES}

====================================
股票基本信息
====================================
代码: {symbol}
名称: {name}
{fundamentals}

====================================
价格数据（最近30个交易日）
====================================
{price_summary}

====================================
技术指标
====================================
{indicators}

====================================
前置分析结果
====================================
趋势分析: {trend_result}
关键价位: {levels_result}

====================================
【重要分析框架】

## 一、市场环境评估（权重30%）
根据VIX和市场风险等级评估当前市场环境：
- VIX < 15: 低波动市场，可适度激进
- VIX 15-25: 正常波动市场，谨慎操作  
- VIX > 25: 高波动市场，降低仓位

## 二、技术面分析（权重40%）
必须分析以下关键信号：
1. RSI信号：RSI>70超买可能回调，RSI<30超卖可能反弹，RSI在40-60最为健康
2. MACD信号：金叉是买入信号，死叉是卖出信号
3. 均线信号：价格站上均线看多；多头排列看多
4. 布林带信号：价格突破上轨可能回调，突破下轨可能反弹
5. 量价配合：价涨量增健康，价涨量缩需警惕

## 四、风险评估（权重10%）
- 最大风险点
- 建议止损价位（必须具体数值）
- 建议仓位（轻仓/半仓/重仓）

====================================
【输出格式要求】

1. 综合评分
   技术面评分: X/10
   基本面评分: X/10  
   综合评分: X/10

2. 短期走势判断
   方向: [看涨/看跌/中性]
   置信度: [高(>70%)/中(50-70%)/低(<50%)]
   主要驱动因素: [不超过50字]

3. 投资建议
   建议: [强烈买入/买入/持有/卖出/强烈卖出]
   理由: [不超过150字]
   买入价位: [具体数值或"现价"]
   目标价位: [具体数值]
   止损价位: [具体数值]

4. 风险提示
   最大风险点: [具体描述]
   建议仓位: [轻仓(<20%)/半仓(20-40%)/重仓(40-60%)]

请严格按照上述格式输出。AI分析结果将用于实盘交易决策，请务必严谨客观。
"""
        
        return prompt
    
    def _build_complete_prompt(self, stock_data: Dict, hist: pd.DataFrame) -> str:
        """构建完整分析提示词（不分步版本）"""
        symbol = stock_data.get('symbol', 'Unknown')
        name = stock_data.get('name', '')
        
        indicators = self._get_technical_indicators(hist)
        price_summary = self._get_price_summary(hist)
        fundamentals = self._format_fundamentals(stock_data)
        
        prompt = f"""你是一位资深股票分析师，请全面分析以下股票。

{self.FEW_SHOT_EXAMPLES}

====================================
股票基本信息
====================================
代码: {symbol}
名称: {name}
{fundamentals}

====================================
价格数据（最近30个交易日）
====================================
{price_summary}

====================================
技术指标
====================================
{indicators}

====================================

====================================
【重要分析框架】

## 一、市场环境评估（权重30%）
根据VIX和市场风险等级评估当前市场环境：
- VIX < 15: 低波动市场，可适度激进
- VIX 15-25: 正常波动市场，谨慎操作  
- VIX > 25: 高波动市场，降低仓位

## 二、技术面分析（权重40%）
必须分析以下关键信号：
1. RSI信号：RSI>70超买可能回调，RSI<30超卖可能反弹，RSI在40-60最为健康
2. MACD信号：金叉是买入信号，死叉是卖出信号
3. 均线信号：价格站上均线看多；多头排列看多
4. 布林带信号：价格突破上轨可能回调，突破下轨可能反弹
5. 量价配合：价涨量增健康，价涨量缩需警惕

## 三、新闻影响分析（权重20%）
- 正面新闻：业绩增长、产品发布、获得订单等 → 看多
- 负面新闻：业绩下滑、诉讼、减持等 → 看空

## 四、风险评估（权重10%）
- 最大风险点
- 建议止损价位（必须具体数值）
- 建议仓位（轻仓/半仓/重仓）

====================================
【输出格式要求】

1. 综合评分
   技术面评分: X/10
   基本面评分: X/10  
   综合评分: X/10

2. 短期走势判断
   方向: [看涨/看跌/中性]
   置信度: [高(>70%)/中(50-70%)/低(<50%)]
   主要驱动因素: [不超过50字]

3. 投资建议
   建议: [强烈买入/买入/持有/卖出/强烈卖出]
   理由: [不超过150字]
   买入价位: [具体数值或"现价"]
   目标价位: [具体数值]
   止损价位: [具体数值]

4. 风险提示
   最大风险点: [具体描述]
   建议仓位: [轻仓(<20%)/半仓(20-40%)/重仓(40-60%)]

请严格按照上述格式输出。AI分析结果将用于实盘交易决策，请务必严谨客观。
"""
        
        return prompt
    
    def _get_technical_indicators(self, hist: pd.DataFrame) -> str:
        """获取技术指标摘要"""
        if hist is None or hist.empty:
            return "无数据"
        
        try:
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest
            
            indicators = []
            
            # RSI
            if 'RSI_14' in hist.columns:
                rsi = latest.get('RSI_14', 0)
                rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "正常"
                indicators.append(f"RSI(14): {rsi:.1f} ({rsi_status})")
            
            # MACD
            if 'MACD' in hist.columns:
                macd = latest.get('MACD', 0)
                signal = latest.get('MACD_Signal', 0)
                macd_status = "金叉" if macd > signal else "死叉"
                indicators.append(f"MACD: {macd:.4f}, Signal: {signal:.4f} ({macd_status})")
            
            # 布林带
            if 'BB_Upper' in hist.columns:
                bb_upper = latest.get('BB_Upper', 0)
                bb_lower = latest.get('BB_Lower', 0)
                close = latest.get('Close', 0)
                bb_position = (close - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50
                indicators.append(f"布林带位置: {bb_position:.1f}%")
            
            # 均线
            for ma in [5, 10, 20, 50]:
                col = f'MA_{ma}'
                if col in hist.columns:
                    ma_val = latest.get(col, 0)
                    indicators.append(f"MA{ma}: {ma_val:.2f}")
            
            # ATR
            if 'ATR_14' in hist.columns:
                atr = latest.get('ATR_14', 0)
                close = latest.get('Close', 1)
                atr_pct = (atr / close * 100) if close else 0
                indicators.append(f"ATR(14): {atr:.2f} ({atr_pct:.2f}%)")
            
            # ===== 新增技术指标 (2026-03-04) =====
            
            # ADX 趋势强度
            if 'ADX_14' in hist.columns:
                adx = latest.get('ADX_14', 0)
                plus_di = latest.get('Plus_DI_14', 0)
                minus_di = latest.get('Minus_DI_14', 0)
                trend_status = "强趋势" if adx > 25 else "横盘" if adx < 20 else "趋势形成"
                indicators.append(f"ADX(14): {adx:.1f} (+DI: {plus_di:.1f}, -DI: {minus_di:.1f}) - {trend_status}")
            
            # CMF 资金流量
            if 'CMF_20' in hist.columns:
                cmf = latest.get('CMF_20', 0)
                cmf_status = "吸筹" if cmf > 0 else "派发"
                indicators.append(f"CMF(20): {cmf:.3f} ({cmf_status})")
            
            # VWAP
            if 'VWAP' in hist.columns:
                vwap = latest.get('VWAP', 0)
                close = latest.get('Close', 0)
                vwap_status = "机构获利的" if close > vwap else "机构被套的"
                indicators.append(f"VWAP: {vwap:.2f} ({vwap_status})")
            
            # Stochastic RSI
            if 'Stoch_RSI_K_14' in hist.columns:
                stoch_rsi_k = latest.get('Stoch_RSI_K_14', 0)
                stoch_rsi_d = latest.get('Stoch_RSI_D_14', 0)
                stoch_status = "超卖" if stoch_rsi_k < 20 else "超买" if stoch_rsi_k > 80 else "正常"
                indicators.append(f"Stoch RSI: K: {stoch_rsi_k:.1f}, D: {stoch_rsi_d:.1f} ({stoch_status})")
            
            return "\n".join(indicators)
            
        except Exception as e:
            return f"指标计算错误: {e}"
    
    def _get_price_summary(self, hist: pd.DataFrame, days: int = 30) -> str:
        """获取价格数据摘要"""
        if hist is None or hist.empty:
            return "无数据"
        
        try:
            recent = hist.tail(days)
            lines = []
            
            for idx, row in recent.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                close = row.get('Close', 0)
                volume = row.get('Volume', 0)
                change = ""
                if 'Price_Change_Pct' in hist.columns:
                    change_pct = row.get('Price_Change_Pct', 0) * 100
                    change = f" ({change_pct:+.2f}%)"
                
                lines.append(f"{date_str}: 收盘 {close:.2f}, 成交量 {volume:,.0f}{change}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"数据获取错误: {e}"
    
    def _calculate_support_resistance(self, hist: pd.DataFrame) -> Dict:
        """计算支撑阻力位"""
        if hist is None or hist.empty:
            return {}
        
        try:
            latest = hist.iloc[-1]
            recent = hist.tail(20)
            
            return {
                'recent_high': recent['High'].max() if 'High' in recent.columns else None,
                'recent_low': recent['Low'].min() if 'Low' in recent.columns else None,
                'ma20': latest.get('MA_20', None),
                'ma50': latest.get('MA_50', None),
                'bb_upper': latest.get('BB_Upper', None),
                'bb_lower': latest.get('BB_Lower', None),
            }
        except Exception as e:
            self.logger.debug(f"计算支撑阻力位失败: {e}")
            return {}
    
    def _format_fundamentals(self, stock_data: Dict) -> str:
        """格式化基本面数据"""
        info = stock_data.get('info', {})
        if not info:
            return "无基本面数据"
        
        lines = []
        
        if info.get('marketCap'):
            lines.append(f"市值: {info['marketCap']:,.0f}")
        if info.get('trailingPE'):
            lines.append(f"市盈率(TTM): {info['trailingPE']:.2f}")
        if info.get('forwardPE'):
            lines.append(f"市盈率(预期): {info['forwardPE']:.2f}")
        if info.get('priceToBook'):
            lines.append(f"市净率: {info['priceToBook']:.2f}")
        if info.get('dividendYield'):
            lines.append(f"股息率: {info['dividendYield']*100:.2f}%")
        if info.get('beta'):
            lines.append(f"Beta: {info['beta']:.2f}")
        
        return "\n".join(lines) if lines else "无基本面数据"
    
    def _extract_direction_and_confidence(self, analysis: str) -> tuple:
        """从分析结果中提取方向和置信度"""
        if not analysis:
            return (PredictionDirection.NEUTRAL, 0.5)
        
        analysis_lower = analysis.lower()
        
        # 优先从"短期走势判断"部分提取方向（更精确）
        short_term_section = ""
        if "短期走势判断" in analysis:
            match = re.search(r'短期走势判断[：:\s]*(.*?)(?=\n[═\-\s]{10,}|\n【|$)', analysis, re.DOTALL)
            if match:
                short_term_section = match.group(1)
        
        direction = PredictionDirection.NEUTRAL
        confidence = 0.5
        
        # 从短期走势判断部分提取方向
        if short_term_section:
            dir_match = re.search(r'方向[：:]\s*(看涨|看跌|中性|上升|下降|震荡)', short_term_section)
            if dir_match:
                dir_value = dir_match.group(1)
                if dir_value in ['看涨', '上升']:
                    direction = PredictionDirection.BULLISH
                    confidence = 0.7
                elif dir_value in ['看跌', '下降']:
                    direction = PredictionDirection.BEARISH
                    confidence = 0.7
                elif dir_value in ['中性', '震荡']:
                    direction = PredictionDirection.NEUTRAL
                    confidence = 0.5
        
        # 如果没有找到，尝试从"投资建议"部分提取
        if direction == PredictionDirection.NEUTRAL and "投资建议" in analysis:
            advice_match = re.search(r'建议[：:]\s*(买入|卖出|持有|观望|增持|减持)', analysis)
            if advice_match:
                advice = advice_match.group(1)
                if advice in ['买入', '增持']:
                    direction = PredictionDirection.BULLISH
                    confidence = 0.7
                elif advice in ['卖出', '减持']:
                    direction = PredictionDirection.BEARISH
                    confidence = 0.7
                elif advice in ['持有', '观望']:
                    direction = PredictionDirection.NEUTRAL
                    confidence = 0.5
        
        # 如果还是没有找到，使用关键词统计（备选方案）
        if direction == PredictionDirection.NEUTRAL:
            bullish_keywords = ['看涨', '买入', '上涨', '突破', '多头', 'bullish', 'buy', 'up']
            bearish_keywords = ['看跌', '卖出', '下跌', '跌破', '空头', 'bearish', 'sell', 'down']
            
            bullish_count = sum(1 for kw in bullish_keywords if kw in analysis_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in analysis_lower)
            
            if bullish_count > bearish_count:
                direction = PredictionDirection.BULLISH
            elif bearish_count > bullish_count:
                direction = PredictionDirection.BEARISH
            else:
                direction = PredictionDirection.NEUTRAL
        
        # 提取置信度
        confidence_patterns = [
            r'置信度[：:]\s*(\d+(?:\.\d+)?)\s*%?',
            r'信心[度]?[：:]\s*(\d+(?:\.\d+)?)\s*%?',
            r'confidence[：:]\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%\s*置信',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, analysis_lower)
            if match:
                try:
                    confidence = float(match.group(1)) / 100
                    confidence = max(0.1, min(1.0, confidence))
                    break
                except ValueError:
                    continue
        
        return (direction, confidence)
    
    def _record_prediction(
        self, 
        symbol: str, 
        direction: PredictionDirection, 
        confidence: float, 
        model: str
    ):
        """记录预测结果"""
        try:
            record = PredictionRecord(
                symbol=symbol,
                date=datetime.now().strftime('%Y-%m-%d'),
                direction=direction,
                confidence=confidence,
                model_used=model
            )
            self.prediction_tracker.record_prediction(record)
        except Exception as e:
            self.logger.warning(f"记录预测失败: {e}")
    
    def _get_cache_key(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        interval: str, 
        model: str
    ) -> str:
        """生成缓存键"""
        import hashlib
        
        symbol = stock_data.get('symbol', 'unknown')
        
        # 使用数据哈希
        hist_hash = hashlib.md5(
            pd.util.hash_pandas_object(hist.tail(30)).values.tobytes()
        ).hexdigest()[:8] if hist is not None and not hist.empty else 'no_hist'
        
        cache_content = f"{symbol}_{interval}_{model}_{hist_hash}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _analyze_with_all_models(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        interval: str,
        enable_cache: bool
    ) -> Optional[AIAnalysisResult]:
        """使用所有可用模型进行分析并计算共识"""
        models = self.AVAILABLE_MODELS[:3]  # 最多使用3个模型
        
        if not models:
            self.logger.warning("没有可用的 Gemini 模型配置")
            return None
        
        results = {}
        directions = []
        confidences = []
        
        for model in models:
            try:
                result = self.analyze(
                    stock_data, hist,
                    interval=interval,
                    model=model,
                    enable_cache=enable_cache
                )
                if result:
                    results[model] = result
                    direction, confidence = self._extract_direction_and_confidence(result.summary)
                    directions.append(direction)
                    confidences.append(confidence)
                    
            except Exception as e:
                self.logger.warning(f"模型 {model} 分析失败: {e}")
        
        if not results:
            return None
        
        # 计算共识
        if directions:
            consensus = self._calculate_consensus(directions, results)
            
            # 综合摘要
            combined_summary = self._combine_summaries(results, consensus)
            
            return AIAnalysisResult(
                summary=combined_summary,
                confidence=consensus.confidence,
                model_used=f"gemini_consensus_{len(results)}models"
            )
        
        # 返回第一个成功的结果
        first_result = list(results.values())[0]
        return first_result
    
    def _calculate_consensus(
        self, 
        directions: List[PredictionDirection], 
        results: Dict[str, AIAnalysisResult]
    ) -> MultiModelConsensus:
        """计算多模型共识"""
        bullish = sum(1 for d in directions if d == PredictionDirection.BULLISH)
        bearish = sum(1 for d in directions if d == PredictionDirection.BEARISH)
        neutral = sum(1 for d in directions if d == PredictionDirection.NEUTRAL)
        
        total = len(directions)
        
        if bullish > bearish and bullish > neutral:
            consensus_direction = PredictionDirection.BULLISH
        elif bearish > bullish and bearish > neutral:
            consensus_direction = PredictionDirection.BEARISH
        else:
            consensus_direction = PredictionDirection.NEUTRAL
        
        # 一致率
        agreement_count = max(bullish, bearish, neutral)
        agreement_ratio = agreement_count / total if total > 0 else 0
        
        # 平均置信度
        avg_confidence = sum(r.confidence for r in results.values()) / len(results)
        
        # 综合置信度
        final_confidence = avg_confidence * agreement_ratio
        
        return MultiModelConsensus(
            direction=consensus_direction,
            confidence=final_confidence,
            agreement_ratio=agreement_ratio,
            models_voted=total,
            bullish_votes=bullish,
            bearish_votes=bearish,
            neutral_votes=neutral,
            key_agreements=[],
            disagreements=[]
        )
    
    def _combine_summaries(
        self, 
        results: Dict[str, AIAnalysisResult], 
        consensus: MultiModelConsensus
    ) -> str:
        """合并多个模型的摘要"""
        lines = ["=== Gemini 多模型共识分析 ===\n"]
        
        lines.append(f"模型数量: {consensus.models_voted}")
        lines.append(f"共识方向: {consensus.direction.value}")
        lines.append(f"一致率: {consensus.agreement_ratio:.1%}")
        lines.append(f"综合置信度: {consensus.confidence:.1%}\n")
        
        lines.append("各模型观点:")
        for model, result in results.items():
            lines.append(f"\n--- {model} ---")
            # 完整显示分析结果（不截断）
            lines.append(result.summary)
        
        return "\n".join(lines)


# 便捷函数
def analyze_with_gemini(
    stock_data: Dict, 
    hist: pd.DataFrame, 
    model: str = None,
    **kwargs
) -> Optional[AIAnalysisResult]:
    """
    便捷函数：使用 Gemini 分析股票
    
    Args:
        stock_data: 股票数据
        hist: 历史数据
        model: 模型名称 (默认 gemini-2.5-flash)
        **kwargs: 其他参数
    
    Returns:
        AIAnalysisResult 或 None
    """
    analyzer = GeminiAIAnalyzer()
    return analyzer.analyze(stock_data, hist, model=model, **kwargs)
