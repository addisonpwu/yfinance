from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from src.core.models.entities import AIAnalysisResult
from src.data.cache.cache_service import OptimizedCache
from src.config.settings import config_manager
from src.utils.logger import get_ai_logger
import os
import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re

# 尝试导入 tenacity，如果不存在则使用简单的重试逻辑
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False


class PredictionDirection(Enum):
    """预测方向"""
    BULLISH = "看涨"
    BEARISH = "看跌"
    NEUTRAL = "中性"


@dataclass
class PredictionRecord:
    """预测记录"""
    symbol: str
    date: str
    direction: PredictionDirection
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    model_used: str
    key_factors: List[str] = field(default_factory=list)
    
    # 验证结果（后续填充）
    actual_return: Optional[float] = None
    verified: bool = False


@dataclass
class MultiModelConsensus:
    """多模型共识结果"""
    direction: PredictionDirection
    confidence: float
    agreement_ratio: float  # 模型一致比例
    models_voted: int
    bullish_votes: int
    bearish_votes: int
    neutral_votes: int
    key_agreements: List[str]
    disagreements: List[str]


class PredictionTracker:
    """预测追踪器 - 记录和验证预测结果"""
    
    TRACKER_FILE = "data_cache/prediction_tracker.json"
    
    def __init__(self):
        self.predictions: List[Dict] = []
        self._load_predictions()
    
    def _load_predictions(self):
        """加载历史预测记录"""
        try:
            if os.path.exists(self.TRACKER_FILE):
                with open(self.TRACKER_FILE, 'r', encoding='utf-8') as f:
                    self.predictions = json.load(f)
        except Exception:
            self.predictions = []
    
    def _save_predictions(self):
        """保存预测记录"""
        try:
            os.makedirs(os.path.dirname(self.TRACKER_FILE), exist_ok=True)
            with open(self.TRACKER_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.predictions, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"保存预测记录失败: {e}")
    
    def record_prediction(self, record: PredictionRecord):
        """记录预测"""
        self.predictions.append({
            'symbol': record.symbol,
            'date': record.date,
            'direction': record.direction.value,
            'confidence': record.confidence,
            'target_price': record.target_price,
            'stop_loss': record.stop_loss,
            'model_used': record.model_used,
            'key_factors': record.key_factors,
            'verified': False,
            'actual_return': None
        })
        self._save_predictions()
    
    def get_model_accuracy(self, model_name: str, days: int = 30) -> Dict:
        """获取某模型的历史准确率"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        model_predictions = [
            p for p in self.predictions 
            if p['model_used'] == model_name and p['date'] >= cutoff and p['verified']
        ]
        
        if not model_predictions:
            return {'accuracy': None, 'total': 0, 'correct': 0}
        
        correct = sum(
            1 for p in model_predictions
            if (p['direction'] == '看涨' and p['actual_return'] > 0) or
               (p['direction'] == '看跌' and p['actual_return'] < 0) or
               (p['direction'] == '中性' and abs(p['actual_return']) < 0.02)
        )
        
        return {
            'accuracy': correct / len(model_predictions),
            'total': len(model_predictions),
            'correct': correct
        }
    
    def verify_predictions(self, get_current_price_func):
        """验证历史预测"""
        for pred in self.predictions:
            if pred['verified']:
                continue
            
            pred_date = datetime.strptime(pred['date'], '%Y-%m-%d')
            days_passed = (datetime.now() - pred_date).days
            
            # 2周后验证
            if days_passed >= 14:
                try:
                    current_price = get_current_price_func(pred['symbol'])
                    if current_price and pred.get('entry_price'):
                        pred['actual_return'] = (current_price - pred['entry_price']) / pred['entry_price']
                        pred['verified'] = True
                except Exception:
                    pass
        
        self._save_predictions()


class AIAnalyzer(ABC):
    """AI 分析器抽象基类"""
    
    @abstractmethod
    def analyze(self, stock_data: Dict, hist: pd.DataFrame, **kwargs) -> Optional[AIAnalysisResult]:
        pass


class IFlowAIAnalyzer(AIAnalyzer):
    """心流 AI 分析器实现 - 增强版"""
    
    # AI 分析缓存子目录
    AI_CACHE_SUBDIR = "ai_analysis"
    
    # Few-shot 学习案例（历史成功预测）
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
    
    def __init__(self):
        self.api_key = os.environ.get("IFLOW_API_KEY", "")
        config = config_manager.get_config()
        self.cache_service = OptimizedCache(enabled=config.data.enable_cache)
        self.config = config_manager.get_config()
        self.logger = get_ai_logger()
        self.api_url = "https://apis.iflow.cn/v1/chat/completions"
        
        # 新增：预测追踪器
        self.prediction_tracker = PredictionTracker()
        
        # 新增：多时间框架数据缓存
        self._multi_timeframe_cache: Dict[str, Dict] = {}
        
        # 新增：市场情绪缓存
        self._market_sentiment_cache: Dict = {}
        self._sentiment_last_update: Optional[datetime] = None
    
    def analyze(self, stock_data: Dict, hist: pd.DataFrame, **kwargs) -> Optional[AIAnalysisResult]:
        """
        分析单只股票 - 增强版
        
        Args:
            stock_data: 股票数据字典
            hist: 历史数据 DataFrame
            **kwargs: 额外参数 (interval, model, use_multi_timeframe, use_consensus 等)
        
        Returns:
            AIAnalysisResult 或 None
        """
        interval = kwargs.get('interval', '1d')
        model = kwargs.get('model', 'deepseek-v3.2')
        use_multi_timeframe = kwargs.get('use_multi_timeframe', True)
        use_consensus = kwargs.get('use_consensus', False)
        
        # 处理 'all' 模型选项
        if model == 'all':
            return self._analyze_with_all_models_enhanced(stock_data, hist, interval)
        
        # 生成缓存键（使用数据哈希而非日期）
        cache_key = self._get_cache_key(stock_data, hist, interval, model)
        
        # 尝试从缓存获取
        cached_result = self.cache_service.get_json(cache_key, self.AI_CACHE_SUBDIR)
        if cached_result:
            self.logger.info(f"从缓存获取 {stock_data.get('symbol', 'Unknown')} 的AI分析结果")
            return AIAnalysisResult(
                summary=cached_result.get('summary', ''),
                confidence=cached_result.get('confidence', 0.5),
                model_used=cached_result.get('model_used', model)
            )
        
        # 检查 API Key
        if not self.api_key:
            self.logger.warning(f"未找到 IFLOW_API_KEY 环境变量，跳过 AI 分析（缓存中无结果）")
            return None
        
        try:
            # 新增：分步分析流程
            analysis_result = self._step_by_step_analysis(stock_data, hist, model, use_multi_timeframe)
            
            if analysis_result:
                # 新增：记录预测
                self._record_prediction(stock_data, analysis_result, model)
                
                # 保存结果到缓存
                cache_data = {
                    'symbol': stock_data.get('symbol', ''),
                    'summary': analysis_result.summary,
                    'confidence': analysis_result.confidence,
                    'model_used': analysis_result.model_used,
                    'direction': analysis_result.details.get('direction', '中性') if analysis_result.details else '中性',
                }
                self.cache_service.set_json(cache_key, cache_data, self.AI_CACHE_SUBDIR)
            
            return analysis_result
                
        except Exception as e:
            self.logger.error(f"AI分析时出错: {e}")
            return None
    
    def _step_by_step_analysis(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        model: str,
        use_multi_timeframe: bool = True
    ) -> Optional[AIAnalysisResult]:
        """
        分步分析流程 - 提高准确性
        
        步骤：
        1. 趋势判断（简单任务，准确率高）
        2. 关键价位识别
        3. 风险评估
        4. 综合建议
        """
        symbol = stock_data.get('symbol', 'Unknown')
        
        # 获取增强数据
        multi_tf_data = self._get_multi_timeframe_data(stock_data, hist) if use_multi_timeframe else None
        market_sentiment = self._get_market_sentiment()
        
        # 第一步：趋势判断
        trend_prompt = self._build_trend_prompt(stock_data, hist, multi_tf_data, market_sentiment)
        trend_result, _ = self._call_iflow_api(trend_prompt, model, max_retries=2)
        
        if not trend_result:
            return None
        
        # 第二步：关键价位和风险评估
        levels_prompt = self._build_levels_prompt(stock_data, hist, trend_result)
        levels_result, _ = self._call_iflow_api(levels_prompt, model, max_retries=2)
        
        # 第三步：综合分析
        final_prompt = self._build_final_prompt(
            stock_data, hist, trend_result, levels_result, 
            multi_tf_data, market_sentiment
        )
        final_result, model_used = self._call_iflow_api(final_prompt, model)
        
        if not final_result:
            return None
        
        # 提取关键信息
        direction, confidence = self._extract_direction_and_confidence(final_result)
        
        return AIAnalysisResult(
            summary=final_result,
            confidence=confidence,
            model_used=model_used,
            details={
                'direction': direction,
                'trend_analysis': trend_result,
                'levels_analysis': levels_result,
                'multi_timeframe': multi_tf_data is not None,
                'market_sentiment': market_sentiment
            }
        )
    
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
        
        # 基础技术指标
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
        
        # 获取近期高低点
        recent_high = hist['High'].tail(60).max()
        recent_low = hist['Low'].tail(60).min()
        current_price = hist['Close'].iloc[-1]
        
        # 获取移动平均线
        ma_values = {}
        for period in [20, 50, 200]:
            col = f'MA_{period}'
            if col in hist.columns:
                ma_values[period] = hist[col].iloc[-1]
        
        # 获取布林带
        bb_upper = hist['BB_Upper'].iloc[-1] if 'BB_Upper' in hist.columns else None
        bb_lower = hist['BB_Lower'].iloc[-1] if 'BB_Lower' in hist.columns else None
        
        prompt = f"""基于以下趋势判断，识别关键价位：

股票: {symbol}
当前价格: {current_price:.2f}
60日最高: {recent_high:.2f}
60日最低: {recent_low:.2f}

趋势判断: {trend_result}

移动平均线:
{chr(10).join([f'- MA{p}: {v:.2f}' for p, v in ma_values.items() if v])}

布林带:
- 上轨: {bb_upper:.2f if bb_upper else 'N/A'}
- 下轨: {bb_lower:.2f if bb_lower else 'N/A'}

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
        """构建综合分析提示词（第三步）"""
        symbol = stock_data.get('symbol', 'N/A')
        info = stock_data.get('info', {})
        strategies = stock_data.get('strategies', [])
        
        # 格式化基本面信息
        fundamentals = self._format_fundamentals(info)
        
        # 格式化技术指标
        tech_indicators = self._get_technical_indicators(hist)
        
        # 历史数据摘要
        hist_summary = self._get_hist_summary(hist)
        
        prompt = f"""你是一位专业的短期股票分析师。请基于前期分析给出综合投资建议。

{self.FEW_SHOT_EXAMPLES}

════════════════════════════════════════════════════════════

【股票信息】
代码: {symbol}
名称: {info.get('longName', 'N/A')}
行业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}
符合策略: {', '.join(strategies) if strategies else '无'}

{fundamentals}

{tech_indicators}

{self._format_multi_timeframe(multi_tf_data) if multi_tf_data else ''}

{self._format_market_sentiment(market_sentiment) if market_sentiment else ''}

【前期分析结果】
趋势判断: {trend_result}

关键价位: {levels_result}

【最近20天数据】
{hist_summary}

════════════════════════════════════════════════════════════

【综合分析要求】

请综合以上所有信息，给出结构化分析：

1. 综合评分（必填）
   - 技术面评分: X/10
   - 基本面评分: X/10
   - 综合评分: X/10

2. 短期走势判断（必填）
   - 方向: 看涨 / 看跌 / 中性
   - 置信度: 高(>70%) / 中(50-70%) / 低(<50%)
   - 主要驱动因素

3. 投资建议（必填）
   - 建议: 强烈买入 / 买入 / 持有 / 卖出 / 强烈卖出
   - 理由（不超过200字）
   - 如果是买入，必须给出具体的买入价、目标价、止损价

4. 风险提示
   - 最大风险点
   - 建议仓位（占总资金比例）

请严格按照上述格式输出，控制在800字以内。
"""
        return prompt
    
    def _analyze_with_all_models(
        self, stock_data: Dict, hist: pd.DataFrame, interval: str
    ) -> Optional[AIAnalysisResult]:
        """使用所有可用模型进行分析并合并结果"""
        models_to_use = [
            'iflow-rome-30ba3b',
            'qwen3-max',
            'tstars2.0',
            'deepseek-v3.2',
            'qwen3-coder-plus'
        ]
        all_results = []
        
        for model_name in models_to_use:
            cache_key = self._get_cache_key(stock_data, hist, interval, model_name)
            cached_result = self.cache_service.get_json(cache_key, self.AI_CACHE_SUBDIR)
            
            if cached_result:
                all_results.append({
                    'summary': cached_result.get('summary', ''),
                    'model_used': cached_result.get('model_used', model_name),
                    'direction': cached_result.get('direction', '中性'),
                    'confidence': cached_result.get('confidence', 0.5)
                })
            elif not self.api_key:
                self.logger.warning(f"未找到 IFLOW_API_KEY，跳过 {model_name} 模型分析")
            else:
                prompt = self._build_analysis_prompt(stock_data, hist)
                try:
                    response, model_used = self._call_iflow_api(prompt, model_name)
                    if response:
                        direction, confidence = self._extract_direction_and_confidence(response)
                        result = {
                            'summary': response,
                            'model_used': model_used,
                            'direction': direction,
                            'confidence': confidence
                        }
                        cache_data = {
                            'symbol': stock_data.get('symbol', ''),
                            'summary': response,
                            'confidence': confidence,
                            'model_used': model_used,
                            'direction': direction,
                        }
                        self.cache_service.set_json(cache_key, cache_data, self.AI_CACHE_SUBDIR)
                        all_results.append(result)
                except Exception as e:
                    self.logger.error(f"{model_name} 模型分析时出错: {e}")
        
        if not all_results:
            return None
        
        # 合并所有模型的分析结果
        combined_summary = "【多模型分析结果】\n\n"
        for result in all_results:
            combined_summary += f"--- {result['model_used']} 模型分析 ---\n"
            combined_summary += result['summary']
            combined_summary += "\n\n"
        
        return AIAnalysisResult(
            summary=combined_summary,
            confidence=0.8,
            model_used='all_models'
        )
    
    def _analyze_with_all_models_enhanced(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame, 
        interval: str
    ) -> Optional[AIAnalysisResult]:
        """
        增强版多模型分析 - 使用投票机制
        """
        models_to_use = [
            'deepseek-v3.2',
            'qwen3-max',
            'tstars2.0',
        ]
        
        all_results = []
        directions = []
        
        for model_name in models_to_use:
            cache_key = self._get_cache_key(stock_data, hist, interval, f"enhanced_{model_name}")
            cached_result = self.cache_service.get_json(cache_key, self.AI_CACHE_SUBDIR)
            
            if cached_result:
                all_results.append({
                    'summary': cached_result.get('summary', ''),
                    'model_used': cached_result.get('model_used', model_name),
                    'direction': cached_result.get('direction', '中性'),
                    'confidence': cached_result.get('confidence', 0.5)
                })
                directions.append(cached_result.get('direction', '中性'))
            elif self.api_key:
                try:
                    result = self._step_by_step_analysis(stock_data, hist, model_name, use_multi_timeframe=True)
                    if result:
                        direction = result.details.get('direction', '中性') if result.details else '中性'
                        all_results.append({
                            'summary': result.summary,
                            'model_used': result.model_used,
                            'direction': direction,
                            'confidence': result.confidence
                        })
                        directions.append(direction)
                        
                        # 缓存
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
        combined_summary = f"""【多模型共识分析】

共识方向: {consensus.direction.value}
共识置信度: {consensus.confidence:.0%}
模型一致率: {consensus.agreement_ratio:.0%} ({consensus.models_voted} 个模型投票)

投票分布:
- 看涨: {consensus.bullish_votes} 票
- 看跌: {consensus.bearish_votes} 票
- 中性: {consensus.neutral_votes} 票

主要共识点:
{chr(10).join(['- ' + a for a in consensus.key_agreements]) if consensus.key_agreements else '- 暂无'}

分歧点:
{chr(10).join(['- ' + d for d in consensus.disagreements]) if consensus.disagreements else '- 暂无'}

════════════════════════════════════════════════════════════

【各模型详细分析】
"""
        for result in all_results:
            combined_summary += f"\n--- {result['model_used']} 模型 ---\n"
            combined_summary += f"判断: {result['direction']} (置信度: {result['confidence']:.0%})\n"
            combined_summary += result['summary'][:500] + "...\n" if len(result['summary']) > 500 else result['summary'] + "\n"
        
        return AIAnalysisResult(
            summary=combined_summary,
            confidence=consensus.confidence,
            model_used='multi_model_consensus',
            details={
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
        
        # 综合置信度（考虑一致性）
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
    
    def _get_multi_timeframe_data(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame
    ) -> Optional[Dict]:
        """
        获取多时间框架数据
        
        Returns:
            Dict: 包含日线、周线趋势信息
        """
        try:
            symbol = stock_data.get('symbol', '')
            
            # 从当前数据计算趋势
            if hist is None or hist.empty:
                return None
            
            # 短期趋势 (5日)
            short_term = self._calculate_trend(hist.tail(10))
            
            # 中期趋势 (20日)
            mid_term = self._calculate_trend(hist.tail(30))
            
            # 长期趋势 (60日)
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
        
        # 价格变化
        price_change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
        
        # 移动平均位置
        ma5 = close.tail(5).mean()
        ma_all = close.mean()
        
        # 趋势方向
        if price_change > 3:
            direction = '上升'
        elif price_change < -3:
            direction = '下降'
        else:
            direction = '震荡'
        
        # 趋势强度 (基于 R-squared)
        x = np.arange(len(close))
        y = close.values
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            strength = abs(correlation)
        except:
            strength = 0.5
        
        return {
            'direction': direction,
            'strength': round(strength, 2),
            'price_change': round(price_change, 2),
            'above_ma': close.iloc[-1] > ma_all
        }
    
    def _get_market_sentiment(self) -> Optional[Dict]:
        """
        获取市场情绪指标
        
        Returns:
            Dict: 包含 VIX、市场状态等信息
        """
        try:
            # 检查缓存（每小时更新一次）
            if (self._sentiment_last_update and 
                (datetime.now() - self._sentiment_last_update).seconds < 3600):
                return self._market_sentiment_cache
            
            # 尝试获取 VIX 数据
            vix_value = self._fetch_vix()
            
            # 解析市场状态
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
        except Exception:
            pass
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
        
        # 提取方向
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
    
    def _record_prediction(self, stock_data: Dict, result: AIAnalysisResult, model: str):
        """记录预测结果"""
        try:
            symbol = stock_data.get('symbol', '')
            direction_str = result.details.get('direction', '中性') if result.details else '中性'
            
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
                target_price=None,  # 可从分析中提取
                stop_loss=None,
                model_used=model,
                key_factors=[]
            )
            
            self.prediction_tracker.record_prediction(record)
            
        except Exception as e:
            self.logger.error(f"记录预测失败: {e}")
    
    def _get_cache_key(
        self, stock_data: Dict, hist: pd.DataFrame, interval: str, model: str
    ) -> str:
        """
        生成 AI 分析结果的缓存键
        
        使用数据哈希而非日期，确保数据不变时缓存可复用
        """
        # 计算历史数据的哈希（只使用最后100条数据）
        hist_hash = None
        if hist is not None and not hist.empty:
            recent_hist = hist.tail(100)
            hist_hash = OptimizedCache.compute_data_hash(
                recent_hist[['Close', 'Volume']].to_dict()
            )
        
        # 计算关键基本面信息的哈希
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
            model=model,
            hist_hash=hist_hash,
            info_hash=info_hash
        )
    
    def _build_analysis_prompt(self, stock_data: Dict, hist: pd.DataFrame) -> str:
        """构建用于股票分析的优化提示词（兼容旧版调用）"""
        symbol = stock_data.get('symbol', 'N/A')
        info = stock_data.get('info', {})
        strategies = stock_data.get('strategies', [])
        
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

        short_ratio = info.get('shortRatio')
        short_ratio_str = f"{short_ratio:.2f}%" if isinstance(short_ratio, (int, float)) and short_ratio > 0 else "N/A"

        # 尝试从预计算的技术指标中获取数据
        technical_indicators = self._get_technical_indicators(hist)
        hist_summary = self._get_hist_summary(hist)
        
        # 获取市场情绪
        market_sentiment = self._get_market_sentiment()
        market_sentiment_str = self._format_market_sentiment(market_sentiment) if market_sentiment else ""
        
        # 构建完整提示词
        prompt = f"""你是一位专业的短期股票分析师。请基于以下信息对股票进行结构化分析，并给出短期投资建议（1-4周内）。

{self.FEW_SHOT_EXAMPLES}

════════════════════════════════════════════════════════════

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

{market_sentiment_str}

【技术面分析】
- 符合的交易策略: {', '.join(strategies) if strategies else '无'}
{technical_indicators}

【最近 20 天历史数据】
{hist_summary}

════════════════════════════════════════════════════════════

【分析要求】

请参考历史成功案例，给出结构化分析：

1. 综合评分（必填）：
   - 技术面评分（1-10分）
   - 基本面评分（1-10分）
   - 综合评分

2. 短期走势判断（必填）：
   - 方向: 看涨 / 看跌 / 中性
   - 置信度: 高 / 中 / 低
   - 主要驱动因素

3. 投资建议（必填）：
   - 明确建议：强烈买入/买入/持有/卖出/强烈卖出
   - 建议理由（不超过200字）
   - 如果是买入，必须给出：买入价位、目标价位、止损价位

4. 风险提示：
   - 最大风险点
   - 建议仓位

请严格按照上述格式输出，控制在800字以内。"""

        return prompt
    
    def _get_technical_indicators(self, hist: pd.DataFrame) -> str:
        """
        获取技术指标摘要
        
        优先使用预计算的指标，避免重复计算
        """
        if hist is None or hist.empty:
            return "无技术指标数据"
        
        recent_data = hist.tail(100)
        
        def format_value(val, decimals=2):
            if isinstance(val, (int, float)):
                return f"{val:.{decimals}f}"
            return str(val)
        
        # 尝试使用预计算的指标
        if 'RSI_14' in hist.columns:
            latest_rsi = hist['RSI_14'].iloc[-1]
        else:
            # 后备计算
            delta = recent_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            latest_rsi = (100 - (100 / (1 + rs))).iloc[-1] if not rs.empty else "N/A"
        
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
        
        if 'ATR_14' in hist.columns:
            latest_atr = hist['ATR_14'].iloc[-1]
        else:
            high_low = recent_data['High'] - recent_data['Low']
            high_close = abs(recent_data['High'] - recent_data['Close'].shift())
            low_close = abs(recent_data['Low'] - recent_data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            latest_atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1] if not tr.empty else "N/A"
        
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
        
        # 威廉指标
        if 'WilliamsR_14' in hist.columns:
            latest_williams_r = hist['WilliamsR_14'].iloc[-1]
        else:
            highest_high = recent_data['High'].rolling(window=14, min_periods=1).max()
            lowest_low = recent_data['Low'].rolling(window=14, min_periods=1).min()
            williams_r = ((highest_high - recent_data['Close']) / (highest_high - lowest_low)) * -100
            latest_williams_r = williams_r.iloc[-1] if not williams_r.empty else "N/A"
        
        # 成交量均线
        if 'Volume_MA_20' in hist.columns:
            latest_volume_sma = hist['Volume_MA_20'].iloc[-1]
        else:
            latest_volume_sma = recent_data['Volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
        
        return f"""
【技术指标】
- RSI (14): {format_value(latest_rsi)}
- MACD: {format_value(latest_macd)} / Signal: {format_value(latest_signal)}
- ATR (14): {format_value(latest_atr)}
- 布林带上轨: {format_value(latest_upper)} / 下轨: {format_value(latest_lower)} / 当前价: {format_value(latest_close)}
- 移动平均线: MA20: {format_value(ma_values[20])}, MA50: {format_value(ma_values[50])}, MA200: {format_value(ma_values[200])}
- 威廉指标 (14): {format_value(latest_williams_r)}
- 平均成交量 (20日): {format_value(latest_volume_sma, 0)}
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
    
    def _call_iflow_api(self, prompt: str, model_name: str, max_retries: int = 3) -> tuple[Optional[str], Optional[str]]:
        """
        调用心流 API，带有自动重试机制

        Args:
            prompt: 分析提示词
            model_name: 要使用的模型名称
            max_retries: 最大重试次数

        Returns:
            (API 返回的文本内容, 实际使用的模型名称) 的元组，如果调用失败返回 (None, None)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
            "max_tokens": None,
            "temperature": 0.0,
            "top_p": 0.7
        }

        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                response.raise_for_status()
                result = response.json()

                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    model = result.get('model', model_name)
                    return content, model
                else:
                    return None, None

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    # 指数退避等待
                    import time
                    wait_time = 2 ** attempt
                    self.logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{max_retries})，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                continue
            except (json.JSONDecodeError, KeyError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    self.logger.warning(f"解析响应失败 (尝试 {attempt + 1}/{max_retries})，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                continue

        self.logger.error(f"API 调用最终失败 (共 {max_retries} 次尝试): {last_error}")
        return None, None