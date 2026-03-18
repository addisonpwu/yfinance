"""
NVIDIA API 分析器

使用 NVIDIA NIM API 进行股票技术分析，支持：
- 流式响应（含 reasoning_content）
- 多模型分析投票
- 预测追踪
- 缓存支持
"""

from typing import Dict, Optional, List, Tuple
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
from dataclasses import dataclass, field
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

# 尝试导入 OpenAI SDK
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


class NvidiaAIAnalyzer(AIAnalyzer):
    """NVIDIA API 分析器实现"""
    
    # AI 分析缓存子目录（独立于 iFlow）
    AI_CACHE_SUBDIR = "ai_analysis_nvidia"
    
    # NVIDIA API 配置
    NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
    
    # 默认模型（将在 __init__ 中从配置读取）
    DEFAULT_MODEL = "z-ai/glm5"
    
    # 可用的 NVIDIA 模型列表（将在 __init__ 中从配置读取）
    AVAILABLE_MODELS = []
    
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
    
    def __init__(self, enable_cache: bool = True, enable_streaming: bool = False):
        """
        初始化 NVIDIA AI 分析器
        
        Args:
            enable_cache: 是否启用缓存
            enable_streaming: 是否启用流式响应
        """
        # 检查 OpenAI SDK
        if not HAS_OPENAI:
            raise ImportError("需要安装 openai 包: pip install openai")
        
        # 获取 API Key
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        
        # 配置
        config = config_manager.get_config()
        # 禁用 AI 分析缓存
        self.cache_service = OptimizedCache(enabled=False)
        self.config = config
        self.logger = get_ai_logger()
        
        # 从配置读取模型列表（配置为唯一数据源）
        if config.ai.providers and config.ai.providers.nvidia:
            self.DEFAULT_MODEL = config.ai.providers.nvidia.default_model
            self.AVAILABLE_MODELS = config.ai.providers.nvidia.available_models
        else:
            # 后备：使用空列表，实际使用时会报错提示检查配置
            self.DEFAULT_MODEL = ""
            self.AVAILABLE_MODELS = []
        
        # 流式响应配置
        self.enable_streaming = enable_streaming
        self._use_color = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
        self._reasoning_color = "\033[90m" if self._use_color else ""
        self._reset_color = "\033[0m" if self._use_color else ""
        
        # 初始化 OpenAI 客户端（设置 5 分钟超时，避免 NVIDIA API 504 错误）
        self.client = OpenAI(
            base_url=self.NVIDIA_API_BASE_URL,
            api_key=self.api_key,
            timeout=300.0  # 5 分钟超时
        )
        
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
                - model: 模型名称 (默认 'z-ai/glm5')
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
            self.logger.warning("未找到 NVIDIA_API_KEY 环境变量，跳过 AI 分析")
            return None
        
        # 尝试从缓存读取
        if enable_cache:
            cache_key = self._get_cache_key(stock_data, hist, interval, model)
            cached_result = self.cache_service.get_json(cache_key, self.AI_CACHE_SUBDIR)
            if cached_result:
                self.logger.info(f"从缓存获取 {stock_data.get('symbol', 'Unknown')} 的 NVIDIA AI 分析结果")
                return AIAnalysisResult(
                    summary=cached_result.get('summary', ''),
                    confidence=cached_result.get('confidence', 0.5),
                    model_used=cached_result.get('model_used', model)
                )
        
        try:
            # 分步分析流程
            analysis_result = self._step_by_step_analysis(stock_data, hist, model, use_multi_timeframe)
            
            if analysis_result:
                # 记录预测
                self._record_prediction(stock_data, analysis_result, model)
                
                # 缓存结果
                if enable_cache:
                    cache_key = self._get_cache_key(stock_data, hist, interval, model)
                    cache_data = {
                        'symbol': stock_data.get('symbol', ''),
                        'summary': analysis_result.summary,
                        'confidence': analysis_result.confidence,
                        'model_used': analysis_result.model_used,
                        'direction': analysis_result.detailed_analysis.get('direction', '中性') if analysis_result.detailed_analysis else '中性',
                    }
                    self.cache_service.set_json(cache_key, cache_data, self.AI_CACHE_SUBDIR)
            
            return analysis_result
                
        except Exception as e:
            self.logger.error(f"NVIDIA AI 分析时出错: {e}")
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
        
        步骤：
        1. 趋势判断
        2. 关键价位识别
        3. 综合分析
        """
        symbol = stock_data.get('symbol', 'Unknown')
        self.logger.info(f"[NVIDIA] 开始分析 {symbol}，模型: {model}")
        
        # 获取增强数据
        self.logger.info(f"[NVIDIA] {symbol} - 步骤0: 获取多时间框架数据和市场情绪")
        multi_tf_data = self._get_multi_timeframe_data(stock_data, hist) if use_multi_timeframe else None
        market_sentiment = self._get_market_sentiment()
        
        # 第一步：趋势判断
        self.logger.info(f"[NVIDIA] {symbol} - 步骤1/3: 趋势判断分析中...")
        trend_prompt = self._build_trend_prompt(stock_data, hist, multi_tf_data, market_sentiment)
        trend_result, _ = self._call_nvidia_api(trend_prompt, model, stream=False)
        
        if not trend_result:
            self.logger.warning(f"[NVIDIA] {symbol} - 步骤1失败: 趋势判断返回空结果")
            return None
        self.logger.info(f"[NVIDIA] {symbol} - 步骤1完成: 趋势判断成功")
        
        # 第二步：关键价位和风险评估
        self.logger.info(f"[NVIDIA] {symbol} - 步骤2/3: 关键价位和风险评估中...")
        levels_prompt = self._build_levels_prompt(stock_data, hist, trend_result)
        levels_result, _ = self._call_nvidia_api(levels_prompt, model, stream=False)
        self.logger.info(f"[NVIDIA] {symbol} - 步骤2完成: 关键价位分析成功")
        
        # 第三步：综合分析
        self.logger.info(f"[NVIDIA] {symbol} - 步骤3/3: 综合分析中...")
        final_prompt = self._build_final_prompt(
            stock_data, hist, trend_result, levels_result, 
            multi_tf_data, market_sentiment
        )
        final_result, model_used = self._call_nvidia_api(final_prompt, model, stream=self.enable_streaming)
        
        if not final_result:
            self.logger.warning(f"[NVIDIA] {symbol} - 步骤3失败: 综合分析返回空结果")
            return None
        
        # 提取关键信息
        direction, confidence = self._extract_direction_and_confidence(final_result)
        self.logger.info(f"[NVIDIA] {symbol} - 分析完成: 方向={direction}, 置信度={confidence:.0%}, 模型={model_used}")
        
        return AIAnalysisResult(
            summary=final_result,
            confidence=confidence,
            model_used=model_used,
            detailed_analysis={
                'direction': direction,
                'trend_analysis': trend_result,
                'levels_analysis': levels_result,
                'multi_timeframe': multi_tf_data is not None,
                'market_sentiment': market_sentiment
            }
        )
    
    def _call_nvidia_api(
        self, 
        prompt: str, 
        model_name: str, 
        stream: bool = True,
        max_retries: int = 3
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        调用 NVIDIA API
        
        Args:
            prompt: 分析提示词
            model_name: 模型名称
            stream: 是否使用流式响应
            max_retries: 最大重试次数
        
        Returns:
            (API 返回的文本内容, 实际使用的模型名称) 的元组
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if stream:
                    return self._call_nvidia_api_stream(prompt, model_name)
                else:
                    return self._call_nvidia_api_sync(prompt, model_name)
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"NVIDIA API 调用失败 (尝试 {attempt + 1}/{max_retries})，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                continue
        
        self.logger.error(f"NVIDIA API 调用最终失败 (共 {max_retries} 次尝试): {last_error}")
        return None, None
    
    def _call_nvidia_api_sync(
        self, 
        prompt: str, 
        model_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        同步调用 NVIDIA API
        
        Args:
            prompt: 分析提示词
            model_name: 模型名称
        
        Returns:
            (API 返回的文本内容, 实际使用的模型名称) 的元组
        """
        start_time = time.time()
        self.logger.info(f"[NVIDIA API] 开始同步调用，模型: {model_name}，提示词长度: {len(prompt)} 字符")
        
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=1,
            max_tokens=16384,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "clear_thinking": False
                }
            },
            stream=False
        )
        
        elapsed = time.time() - start_time
        self.logger.info(f"[NVIDIA API] 同步调用完成，耗时: {elapsed:.1f}秒")
        
        if completion.choices and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            model = completion.model or model_name
            return content, model
        
        return None, None
    
    def _call_nvidia_api_stream(
        self, 
        prompt: str, 
        model_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        调用 NVIDIA API（流式响应）
        
        支持 reasoning_content 和 content 的分离输出
        """
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=1,
            max_tokens=16384,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "clear_thinking": False
                }
            },
            stream=True
        )
        
        reasoning_content = ""
        content = ""
        model_used = model_name
        
        for chunk in completion:
            # 检查有效的 chunk
            if not getattr(chunk, "choices", None):
                continue
            if len(chunk.choices) == 0:
                continue
            if getattr(chunk.choices[0], "delta", None) is None:
                continue
            
            delta = chunk.choices[0].delta
            
            # 获取模型名称
            if hasattr(chunk, 'model') and chunk.model:
                model_used = chunk.model
            
            # 处理 reasoning_content（思考过程）
            if getattr(delta, "reasoning_content", None):
                reasoning = delta.reasoning_content
                reasoning_content += reasoning
                # 输出思考过程（灰色）
                print(f"{self._reasoning_color}{reasoning}{self._reset_color}", end="", flush=True)
            
            # 处理 content（最终回答）
            if getattr(delta, "content", None) is not None:
                text = delta.content
                content += text
                print(text, end="", flush=True)
        
        # 输出换行
        if content or reasoning_content:
            print()
        
        return content if content else None, model_used
    
    def _analyze_with_all_models(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        interval: str,
        enable_cache: bool = True
    ) -> Optional[AIAnalysisResult]:
        """
        使用所有可用 NVIDIA 模型进行分析并投票
        """
        all_results = []
        directions = []
        
        for model_name in self.AVAILABLE_MODELS:
            cache_key = self._get_cache_key(stock_data, hist, interval, f"multi_{model_name}")
            
            # 尝试从缓存读取
            if enable_cache:
                cached_result = self.cache_service.get_json(cache_key, self.AI_CACHE_SUBDIR)
                if cached_result:
                    all_results.append({
                        'summary': cached_result.get('summary', ''),
                        'model_used': cached_result.get('model_used', model_name),
                        'direction': cached_result.get('direction', '中性'),
                        'confidence': cached_result.get('confidence', 0.5)
                    })
                    directions.append(cached_result.get('direction', '中性'))
                    continue
            
            # 调用 API
            if self.api_key:
                try:
                    result = self._step_by_step_analysis(stock_data, hist, model_name, use_multi_timeframe=True)
                    if result:
                        direction = result.detailed_analysis.get('direction', '中性') if result.detailed_analysis else '中性'
                        all_results.append({
                            'summary': result.summary,
                            'model_used': result.model_used,
                            'direction': direction,
                            'confidence': result.confidence
                        })
                        directions.append(direction)
                        
                        # 缓存
                        if enable_cache:
                            cache_data = {
                                'symbol': stock_data.get('symbol', ''),
                                'summary': result.summary,
                                'confidence': result.confidence,
                                'model_used': result.model_used,
                                'direction': direction,
                            }
                            self.cache_service.set_json(cache_key, cache_data, self.AI_CACHE_SUBDIR)
                except Exception as e:
                    self.logger.error(f"{model_name} 模型分析时出错: {e}")
        
        if not all_results:
            return None
        
        # 计算共识
        consensus = self._calculate_consensus(directions, all_results)
        
        # 构建合并结果
        combined_summary = f"""【NVIDIA 多模型共识分析】

共识方向: {consensus.direction.value}
共识置信度: {consensus.confidence:.0%}
模型一致率: {consensus.agreement_ratio:.0%} ({consensus.models_voted} 个模型投票)

投票分布:
- 看涨: {consensus.bullish_votes} 票
- 看跌: {consensus.bearish_votes} 票
- 中性: {consensus.neutral_votes} 票

════════════════════════════════════════════════════════════

【各模型详细分析】
"""
        for result in all_results:
            combined_summary += f"\n--- {result['model_used']} 模型 ---\n"
            combined_summary += f"判断: {result['direction']} (置信度: {result['confidence']:.0%})\n"
            # 完整显示分析结果（不截断）
            combined_summary += result['summary'] + "\n"
        
        return AIAnalysisResult(
            summary=combined_summary,
            confidence=consensus.confidence,
            model_used='nvidia_multi_model_consensus',
            detailed_analysis={
                'direction': consensus.direction.value,
                'consensus': {
                    'agreement_ratio': consensus.agreement_ratio,
                    'bullish_votes': consensus.bullish_votes,
                    'bearish_votes': consensus.bearish_votes,
                    'neutral_votes': consensus.neutral_votes,
                }
            }
        )
    
    def _calculate_consensus(
        self, 
        directions: List[str], 
        results: List[Dict]
    ) -> MultiModelConsensus:
        """计算多模型共识"""
        bullish = sum(1 for d in directions if '看涨' in d or '上升' in d)
        bearish = sum(1 for d in directions if '看跌' in d or '下降' in d)
        neutral = len(directions) - bullish - bearish
        
        total = len(directions)
        if total == 0:
            return MultiModelConsensus(
                direction=PredictionDirection.NEUTRAL,
                confidence=0.5,
                agreement_ratio=0,
                models_voted=0,
                bullish_votes=0,
                bearish_votes=0,
                neutral_votes=0,
                key_agreements=[],
                disagreements=[]
            )
        
        # 确定共识方向
        if bullish > bearish and bullish > neutral:
            consensus_direction = PredictionDirection.BULLISH
            agreement_ratio = bullish / total
        elif bearish > bullish and bearish > neutral:
            consensus_direction = PredictionDirection.BEARISH
            agreement_ratio = bearish / total
        else:
            consensus_direction = PredictionDirection.NEUTRAL
            agreement_ratio = neutral / total
        
        # 计算平均置信度
        avg_confidence = sum(r.get('confidence', 0.5) for r in results) / len(results)
        
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
    
    # ==================== 以下方法复用自 IFlowAIAnalyzer ====================
    
    def _build_trend_prompt(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame,
        multi_tf_data: Optional[Dict],
        market_sentiment: Optional[Dict]
    ) -> str:
        """构建趋势判断提示词（第一步）"""
        symbol = stock_data.get('symbol', 'N/A')
        info = stock_data.get('info', {})
        
        tech_indicators = self._get_technical_indicators(hist)
        
        prompt = f"""你是一位专业的技术分析师。请仅判断这支股票的短期趋势方向。

股票: {symbol} - {info.get('longName', 'N/A')}

{tech_indicators}

{self._format_multi_timeframe(multi_tf_data) if multi_tf_data else ''}

{self._format_market_sentiment(market_sentiment) if market_sentiment else ''}

【任务】仅回答以下问题，不要给出投资建议：

1. 当前趋势方向：上升 / 下降 / 震荡（三选一）
2. 趋势强度：强 / 中 / 弱（三选一）
3. 趋势判断理由（不超过100字）

请严格按照以下格式回答：
趋势方向: [上升/下降/震荡]
趋势强度: [强/中/弱]
判断理由: [简要说明]
"""
        return prompt
    
    def _build_levels_prompt(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame,
        trend_result: str
    ) -> str:
        """构建关键价位提示词（第二步）"""
        symbol = stock_data.get('symbol', 'N/A')
        
        recent_high = hist['High'].tail(60).max()
        recent_low = hist['Low'].tail(60).min()
        current_price = hist['Close'].iloc[-1]
        
        ma_values = {}
        for period in [20, 50, 200]:
            col = f'MA_{period}'
            if col in hist.columns:
                ma_values[period] = hist[col].iloc[-1]
        
        bb_upper = hist['BB_Upper'].iloc[-1] if 'BB_Upper' in hist.columns else None
        bb_lower = hist['BB_Lower'].iloc[-1] if 'BB_Lower' in hist.columns else None
        
        bb_upper_str = f"{bb_upper:.2f}" if isinstance(bb_upper, (int, float)) else "N/A"
        bb_lower_str = f"{bb_lower:.2f}" if isinstance(bb_lower, (int, float)) else "N/A"
        
        prompt = f"""基于以下趋势判断，识别关键价位：

股票: {symbol}
当前价格: {current_price:.2f}
60日最高: {recent_high:.2f}
60日最低: {recent_low:.2f}

趋势判断: {trend_result}

移动平均线:
{chr(10).join([f'- MA{p}: {v:.2f}' for p, v in ma_values.items() if v])}

布林带:
- 上轨: {bb_upper_str}
- 下轨: {bb_lower_str}

【任务】识别以下价位（每个价位给出具体数值）：

1. 最近阻力位（2个）
2. 最近支撑位（2个）
3. 建议止损价位
4. 建议止盈目标价

请严格按照以下格式回答：
阻力位: [价位1], [价位2]
支撑位: [价位1], [价位2]
止损价: [具体数值]
目标价: [具体数值]
"""
        return prompt
    
    def _build_final_prompt(
        self,
        stock_data: Dict,
        hist: pd.DataFrame,
        trend_result: str,
        levels_result: str,
        multi_tf_data: Optional[Dict],
        market_sentiment: Optional[Dict]
    ) -> str:
        """构建综合分析提示词（第三步）- 增强版"""
        symbol = stock_data.get('symbol', 'N/A')
        info = stock_data.get('info', {})
        strategies = stock_data.get('strategies', [])
        
        fundamentals = self._format_fundamentals(info)
        tech_indicators = self._get_technical_indicators(hist)
        hist_summary = self._get_hist_summary(hist)
        market_env = self._format_market_sentiment(market_sentiment) if market_sentiment else ''
        
        prompt = f"""你是一位专业的短期股票分析师（擅长1-4周内的短线交易）。请基于前期分析给出综合投资建议。

{self.FEW_SHOT_EXAMPLES}

════════════════════════════════════════════════════════════

【股票基本信息】
代码: {symbol}
名称: {info.get('longName', 'N/A')}
行业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}
符合策略: {', '.join(strategies) if strategies else '无'}

{fundamentals}

{tech_indicators}

{self._format_multi_timeframe(multi_tf_data) if multi_tf_data else ''}

{market_env}

【前期分析结果】
趋势判断: {trend_result}

关键价位: {levels_result}

【最近20天数据】
{hist_summary}

════════════════════════════════════════════════════════════

【重要分析框架 - 请严格按照此框架分析】

## 一、市场环境评估（权重30%）
根据VIX和市场风险等级评估当前市场环境：
- VIX < 15: 低波动市场，可适度激进
- VIX 15-25: 正常波动市场，谨慎操作
- VIX > 25: 高波动市场，降低仓位

## 二、技术面分析（权重50%）
必须分析以下关键信号：
1. RSI信号：RSI>70超买可能回调，RSI<30超卖可能反弹，RSI在40-60最为健康
2. MACD信号：金叉（ DIF上穿DEA）是买入信号，死叉是卖出信号
3. 均线信号：价格站上均线看多，跌破均线看空；多头排列（MA5>MA10>MA20）看多
4. 布林带信号：价格突破上轨可能回调，突破下轨可能反弹
5. 量价配合：价涨量增健康，价涨量缩需警惕

## 三、风险评估（权重10%）
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
    
    def _get_technical_indicators(self, hist: pd.DataFrame) -> str:
        """获取技术指标摘要"""
        if hist is None or hist.empty:
            return "无技术指标数据"
        
        recent_data = hist.tail(100)
        
        def format_value(val, decimals=2):
            if isinstance(val, (int, float)):
                return f"{val:.{decimals}f}"
            return str(val)
        
        # RSI
        if 'RSI_14' in hist.columns:
            latest_rsi = hist['RSI_14'].iloc[-1]
        else:
            delta = recent_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            latest_rsi = (100 - (100 / (1 + rs))).iloc[-1] if not rs.empty else "N/A"
        
        # MACD
        if 'MACD' in hist.columns and 'MACD_Signal' in hist.columns:
            latest_macd = hist['MACD'].iloc[-1]
            latest_signal = hist['MACD_Signal'].iloc[-1]
        else:
            exp12 = recent_data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = recent_data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            latest_macd = macd.iloc[-1] if not macd.empty else "N/A"
            latest_signal = signal.iloc[-1] if not signal.empty else "N/A"
        
        # ATR
        if 'ATR_14' in hist.columns:
            latest_atr = hist['ATR_14'].iloc[-1]
        else:
            high_low = recent_data['High'] - recent_data['Low']
            high_close = abs(recent_data['High'] - recent_data['Close'].shift())
            low_close = abs(recent_data['Low'] - recent_data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            latest_atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1] if not tr.empty else "N/A"
        
        # 布林带
        if 'BB_Upper' in hist.columns and 'BB_Lower' in hist.columns:
            latest_upper = hist['BB_Upper'].iloc[-1]
            latest_lower = hist['BB_Lower'].iloc[-1]
        else:
            sma20 = recent_data['Close'].rolling(window=20, min_periods=1).mean()
            std20 = recent_data['Close'].rolling(window=20, min_periods=1).std()
            latest_upper = (sma20 + std20 * 2).iloc[-1] if not sma20.empty else "N/A"
            latest_lower = (sma20 - std20 * 2).iloc[-1] if not sma20.empty else "N/A"
        
        latest_close = hist['Close'].iloc[-1] if not hist.empty else "N/A"
        
        # 移动平均线
        ma_values = {}
        for period in [20, 50, 200]:
            col_name = f'MA_{period}'
            if col_name in hist.columns:
                ma_values[period] = hist[col_name].iloc[-1]
            else:
                ma_values[period] = recent_data['Close'].rolling(window=period, min_periods=1).mean().iloc[-1]
        
        # ===== 新增技术指标 (2026-03-04) =====
        
        # ADX 趋势强度
        if 'ADX_14' in hist.columns:
            latest_adx = hist['ADX_14'].iloc[-1]
            latest_plus_di = hist['Plus_DI_14'].iloc[-1]
            latest_minus_di = hist['Minus_DI_14'].iloc[-1]
        else:
            high_low = recent_data['High'] - recent_data['Low']
            high_close = abs(recent_data['High'] - recent_data['Close'].shift())
            low_close = abs(recent_data['Low'] - recent_data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            diff_high = recent_data['High'].diff()
            diff_low = -recent_data['Low'].diff()
            plus_dm = diff_high.where((diff_high > diff_low) & (diff_high > 0), 0)
            minus_dm = diff_low.where((diff_low > diff_high) & (diff_low > 0), 0)
            plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / tr)
            minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / tr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            latest_adx = dx.rolling(window=14, min_periods=1).mean().iloc[-1] if not dx.empty else "N/A"
            latest_plus_di = plus_di.iloc[-1] if not plus_di.empty else "N/A"
            latest_minus_di = minus_di.iloc[-1] if not minus_di.empty else "N/A"
        
        # CMF 资金流量
        if 'CMF_20' in hist.columns:
            latest_cmf = hist['CMF_20'].iloc[-1]
        else:
            mf_multiplier = ((recent_data['Close'] - recent_data['Low']) - (recent_data['High'] - recent_data['Close'])) / \
                           (recent_data['High'] - recent_data['Low'] + 1e-10)
            mf_volume = mf_multiplier * recent_data['Volume']
            latest_cmf = mf_volume.rolling(window=20, min_periods=1).sum().iloc[-1] / \
                        recent_data['Volume'].rolling(window=20, min_periods=1).sum().iloc[-1] if not recent_data.empty else "N/A"
        
        # VWAP
        if 'VWAP' in hist.columns:
            latest_vwap = hist['VWAP'].iloc[-1]
        else:
            typical_price = (recent_data['High'] + recent_data['Low'] + recent_data['Close']) / 3
            latest_vwap = (typical_price * recent_data['Volume']).cumsum().iloc[-1] / \
                         recent_data['Volume'].cumsum().iloc[-1] if not recent_data.empty else "N/A"
        
        # Stochastic RSI
        if 'Stoch_RSI_K_14' in hist.columns:
            latest_stoch_rsi_k = hist['Stoch_RSI_K_14'].iloc[-1]
            latest_stoch_rsi_d = hist['Stoch_RSI_D_14'].iloc[-1]
        else:
            latest_stoch_rsi_k = "N/A"
            latest_stoch_rsi_d = "N/A"
        
        return f"""
【技术指标】
- RSI (14): {format_value(latest_rsi)}
- MACD: {format_value(latest_macd)} / Signal: {format_value(latest_signal)}
- ATR (14): {format_value(latest_atr)}
- 布林带上轨: {format_value(latest_upper)} / 下轨: {format_value(latest_lower)} / 当前价: {format_value(latest_close)}
- 移动平均线: MA20: {format_value(ma_values[20])}, MA50: {format_value(ma_values[50])}, MA200: {format_value(ma_values[200])}
- ADX (14): {format_value(latest_adx)} / +DI: {format_value(latest_plus_di)} / -DI: {format_value(latest_minus_di)}
- CMF (20): {format_value(latest_cmf)}
- VWAP: {format_value(latest_vwap)}
- Stochastic RSI: K: {format_value(latest_stoch_rsi_k)} / D: {format_value(latest_stoch_rsi_d)}
"""
    
    def _get_hist_summary(self, hist: pd.DataFrame) -> str:
        """获取历史数据摘要（最近20天）"""
        if hist is None or hist.empty:
            return "无历史数据"
        
        hist_lines = []
        for idx, row in hist.tail(20).iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            close = row.get('Close', 'N/A')
            volume = row.get('Volume', 'N/A')
            close_str = f"{close:.2f}" if isinstance(close, (int, float)) else str(close)
            volume_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) else str(volume)
            hist_lines.append(f"{date_str}: 收盘价 {close_str}, 成交量 {volume_str}")
        
        return "\n".join(hist_lines)
    
    def _get_multi_timeframe_data(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame
    ) -> Optional[Dict]:
        """获取多时间框架数据"""
        try:
            if hist is None or hist.empty:
                return None
            
            short_term = self._calculate_trend(hist.tail(10))
            mid_term = self._calculate_trend(hist.tail(30))
            long_term = self._calculate_trend(hist.tail(60)) if len(hist) >= 60 else None
            
            return {
                'short_term': short_term,
                'mid_term': mid_term,
                'long_term': long_term,
            }
            
        except Exception as e:
            self.logger.error(f"获取多时间框架数据失败: {e}")
            return None
    
    def _calculate_trend(self, data: pd.DataFrame) -> Dict:
        """计算趋势指标"""
        if data.empty or len(data) < 5:
            return {'direction': 'unknown', 'strength': 0}
        
        close = data['Close']
        price_change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
        
        if price_change > 3:
            direction = '上升'
        elif price_change < -3:
            direction = '下降'
        else:
            direction = '震荡'
        
        x = np.arange(len(close))
        y = close.values
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            strength = abs(correlation)
        except Exception as e:
            self.logger.debug(f"计算趋势强度失败，使用默认值: {e}")
            strength = 0.5
        
        return {
            'direction': direction,
            'strength': round(strength, 2),
            'price_change': round(price_change, 2),
        }
    
    def _get_market_sentiment(self) -> Optional[Dict]:
        """获取市场情绪指标"""
        try:
            if (self._sentiment_last_update and 
                (datetime.now() - self._sentiment_last_update).seconds < 3600):
                return self._market_sentiment_cache
            
            vix_value = self._fetch_vix()
            
            if vix_value:
                if vix_value < 15:
                    market_state = "低波动/乐观"
                    risk_level = "低"
                elif vix_value < 20:
                    market_state = "正常波动"
                    risk_level = "中"
                elif vix_value < 30:
                    market_state = "高波动/谨慎"
                    risk_level = "中高"
                else:
                    market_state = "极度恐慌"
                    risk_level = "高"
            else:
                market_state = "未知"
                risk_level = "中"
                vix_value = None
            
            self._market_sentiment_cache = {
                'vix': vix_value,
                'market_state': market_state,
                'risk_level': risk_level,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            self._sentiment_last_update = datetime.now()
            
            return self._market_sentiment_cache
            
        except Exception as e:
            self.logger.error(f"获取市场情绪失败: {e}")
            return None
    
    def _fetch_vix(self) -> Optional[float]:
        """获取 VIX 指数"""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            self.logger.debug(f"获取VIX数据失败: {e}")
        return None
    
    def _format_multi_timeframe(self, data: Optional[Dict]) -> str:
        """格式化多时间框架数据"""
        if not data:
            return ""
        
        lines = ["【多时间框架分析】"]
        
        if data.get('short_term'):
            st = data['short_term']
            lines.append(f"- 短期(5日): {st['direction']} ({st['price_change']:+.1f}%)")
        
        if data.get('mid_term'):
            mt = data['mid_term']
            lines.append(f"- 中期(20日): {mt['direction']} ({mt['price_change']:+.1f}%)")
        
        if data.get('long_term'):
            lt = data['long_term']
            lines.append(f"- 长期(60日): {lt['direction']} ({lt['price_change']:+.1f}%)")
        
        return "\n".join(lines)
    
    def _format_market_sentiment(self, data: Optional[Dict]) -> str:
        """格式化市场情绪数据"""
        if not data:
            return ""
        
        lines = ["【市场情绪】"]
        if data.get('vix'):
            lines.append(f"- VIX指数: {data['vix']:.1f}")
        lines.append(f"- 市场状态: {data.get('market_state', '未知')}")
        lines.append(f"- 风险等级: {data.get('risk_level', '中')}")
        
        return "\n".join(lines)
    
    
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
    
    def _extract_direction_and_confidence(self, text: str) -> Tuple[str, float]:
        """从分析结果中提取方向和置信度"""
        direction = "中性"
        confidence = 0.5
        
        # 优先从"短期走势判断"部分提取方向（更精确）
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
        
        # 如果还是没有找到，使用全文搜索（备选方案）
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
        
        conf_match = re.search(r'置信度[：:]\s*(高|中|低)', text)
        if conf_match:
            level = conf_match.group(1)
            if level == '高':
                confidence = min(confidence + 0.2, 0.95)
            elif level == '低':
                confidence = max(confidence - 0.2, 0.3)
        
        return direction, confidence
    
    def _record_prediction(self, stock_data: Dict, result: AIAnalysisResult, model: str):
        """记录预测结果"""
        try:
            symbol = stock_data.get('symbol', '')
            direction_str = result.detailed_analysis.get('direction', '中性') if result.detailed_analysis else '中性'
            
            direction_map = {
                '看涨': PredictionDirection.BULLISH,
                '看跌': PredictionDirection.BEARISH,
                '中性': PredictionDirection.NEUTRAL,
                '上升': PredictionDirection.BULLISH,
                '下降': PredictionDirection.BEARISH,
            }
            
            direction = direction_map.get(direction_str, PredictionDirection.NEUTRAL)
            
            record = PredictionRecord(
                symbol=symbol,
                date=datetime.now().strftime('%Y-%m-%d'),
                direction=direction,
                confidence=result.confidence,
                target_price=None,
                stop_loss=None,
                model_used=f"nvidia_{model}",
                key_factors=[]
            )
            
            self.prediction_tracker.record_prediction(record)
            
        except Exception as e:
            self.logger.error(f"记录预测失败: {e}")
    
    def _get_cache_key(
        self, stock_data: Dict, hist: pd.DataFrame, interval: str, model: str
    ) -> str:
        """生成缓存键"""
        hist_hash = None
        if hist is not None and not hist.empty:
            recent_hist = hist.tail(100)
            hist_hash = OptimizedCache.compute_data_hash(
                recent_hist[['Close', 'Volume']].to_dict()
            )
        
        info = stock_data.get('info', {})
        key_info = {
            k: v for k, v in info.items()
            if k in ['marketCap', 'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook',
                     'profitMargins', 'returnOnEquity', 'revenueGrowth', 'earningsGrowth',
                     'dividendYield', 'beta', '52WeekChange', 'targetMeanPrice']
        }
        info_hash = OptimizedCache.compute_data_hash(key_info)
        
        return self.cache_service.generate_ai_cache_key(
            symbol=stock_data.get('symbol', ''),
            strategies=stock_data.get('strategies', []),
            market=stock_data.get('market', 'HK'),
            interval=interval,
            model=f"nvidia_{model}",
            hist_hash=hist_hash,
            info_hash=info_hash
        )
