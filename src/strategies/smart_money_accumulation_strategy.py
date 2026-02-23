"""
主力建仓策略 v2.0

更准确地定位"建仓接近尾声"的阶段，从"正在建仓"过渡到"即将拉升"

核心优化：
1. 抗跌性量化：引入贝塔系数（Beta）分析
2. 建仓尾声确认：价平量缩 + 筹码锁定性分析
3. "挖坑"形态确认：站上坑沿 + 温和放量 + OBV创新高
4. 资金流向升级：引入柴金资金流（CMF）
"""

from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
from src.strategies.strategy_config import SmartMoneyAccumulationConfig, strategy_config_manager
from src.utils.logger import get_analysis_logger
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


class SmartMoneyAccumulationStrategy(BaseStrategy):
    """
    主力建仓策略 v2.0
    
    识别主力资金建仓接近尾声、即将启动的股票：
    - 抗跌性分析（Beta系数）
    - 建仓尾声确认（价平量缩）
    - 挖坑形态验证
    - CMF 资金流分析
    """
    
    def __init__(self, config: SmartMoneyAccumulationConfig = None):
        self._config = config or strategy_config_manager.get_config('smart_money_accumulation')
        if not isinstance(self._config, SmartMoneyAccumulationConfig):
            self._config = SmartMoneyAccumulationConfig()
        
        if not self._config.validate():
            self._config = SmartMoneyAccumulationConfig()
        
        self._logger = get_analysis_logger()
    
    @property
    def name(self) -> str:
        return "主力建仓策略"
    
    @property
    def category(self) -> str:
        return "早期信号策略"
    
    @property
    def config(self) -> SmartMoneyAccumulationConfig:
        return self._config
    
    def execute(self, context: StrategyContext) -> StrategyResult:
        """执行主力建仓策略检查"""
        hist = context.hist
        benchmark = context.benchmark  # 大盘数据
        
        if hist is None or len(hist) < self._config.min_data_points:
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"数据不足，需要至少 {self._config.min_data_points} 天"}
            )
        
        try:
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            if pd.isna(current_price) or pd.isna(current_volume) or current_price <= 0:
                return StrategyResult(
                    passed=False,
                    confidence=0.0,
                    details={"reason": "价格或成交量数据无效"}
                )
            
            # === 1. 抗跌性评分（Beta分析）===
            resilience_result = self._score_resilience_with_beta(hist, benchmark, current_price)
            
            # === 2. 建仓尾声评分（价平量缩）===
            ending_result = self._score_accumulation_ending(hist, current_price, current_volume)
            
            # === 3. 成交量模式评分 ===
            volume_result = self._score_volume_pattern(hist, current_volume)
            
            # === 4. 挖坑形态评分（增强验证）===
            pattern_result = self._score_digging_pattern_enhanced(hist, current_price, current_volume)
            
            # === 5. 资金流向评分（CMF升级）===
            flow_result = self._score_money_flow_enhanced(hist, current_price, current_volume)
            
            # === 综合评分 ===
            # 建仓尾声是关键信号，给予更高权重
            total_score = (
                resilience_result['score'] * 0.20 +      # 抗跌性 20%
                ending_result['score'] * 0.25 +          # 建仓尾声 25%
                volume_result['score'] * 0.15 +          # 成交量模式 15%
                pattern_result['score'] * 0.20 +         # 挖坑形态 20%
                flow_result['score'] * 0.20              # 资金流向 20%
            )
            
            # 建仓尾声信号的乘数效应
            ending_multiplier = self._calculate_ending_multiplier(ending_result)
            total_score *= ending_multiplier
            
            # 通过条件
            passed = total_score >= self._config.min_score
            
            # 置信度计算
            confidence = self._calculate_confidence(
                total_score, resilience_result, ending_result, flow_result
            )
            
            # 确定建仓阶段
            phase = self._determine_accumulation_phase(total_score, ending_result)
            
            # 预警等级
            alert_level = self._determine_alert_level(total_score, ending_result, flow_result)
            
            if passed:
                self._logger.info(
                    f"[{self.name}] {context.info.get('symbol', 'Unknown')} 通过筛选 - "
                    f"总分: {total_score:.1f}, 阶段: {phase}, "
                    f"Beta: {resilience_result.get('beta', 'N/A')}, "
                    f"尾声信号: {ending_result.get('signal', 'N/A')}, "
                    f"CMF: {flow_result.get('cmf', 'N/A')}"
                )
            
            return StrategyResult(
                passed=passed,
                confidence=confidence,
                details={
                    "total_score": round(total_score, 1),
                    "ending_multiplier": round(ending_multiplier, 2),
                    "accumulation_phase": phase,
                    "alert_level": alert_level,
                    "scores": {
                        "resilience": resilience_result['score'],
                        "ending_signal": ending_result['score'],
                        "volume_pattern": volume_result['score'],
                        "digging_pattern": pattern_result['score'],
                        "money_flow": flow_result['score']
                    },
                    "resilience_details": resilience_result,
                    "ending_details": ending_result,
                    "volume_details": volume_result,
                    "pattern_details": pattern_result,
                    "flow_details": flow_result
                }
            )
            
        except Exception as e:
            self._logger.error(f"[{self.name}] 执行策略时出错: {e}")
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"策略执行错误: {str(e)}"}
            )
    
    def _score_resilience_with_beta(
        self, 
        hist: pd.DataFrame, 
        benchmark: Optional[pd.DataFrame],
        current_price: float
    ) -> Dict[str, Any]:
        """
        抗跌性评分（Beta系数分析）
        
        Beta < 1 且呈下降趋势 = 股票比市场更稳定（抗跌）
        
        Returns:
            包含 score, beta, details 的字典
        """
        try:
            period = self._config.accumulation_period
            score = 0
            details = {}
            
            # 计算价格变化
            recent_close = hist['Close'].iloc[-period:]
            price_change = (recent_close.iloc[-1] / recent_close.iloc[0] - 1)
            price_volatility = recent_close.std() / recent_close.mean()
            
            details['price_change_pct'] = round(price_change * 100, 2)
            details['price_volatility'] = round(price_volatility * 100, 2)
            
            # === Beta 分析 ===
            beta = None
            if benchmark is not None and len(benchmark) >= period:
                beta, beta_trend = self._calculate_beta(hist, benchmark, period)
            
            if beta is not None:
                details['beta'] = round(beta, 3)
                beta_trend_val = beta_trend if beta_trend is not None else 0
                details['beta_trend'] = round(beta_trend_val, 3)
                
                # Beta 评分
                if beta < 0.8:
                    score += 10
                    details['beta_status'] = "低Beta，强抗跌"
                elif beta < 1.0:
                    score += 7
                    details['beta_status'] = "Beta<1，抗跌"
                elif beta < 1.2:
                    score += 4
                    details['beta_status'] = "Beta正常"
                else:
                    details['beta_status'] = "高Beta"
                
                # Beta 趋势（下降 = 越来越抗跌）
                if beta_trend_val < -0.1:
                    score += 5
                    details['beta_trend_status'] = "抗跌性增强"
                elif beta_trend_val < 0:
                    score += 3
                    details['beta_trend_status'] = "趋于稳定"
            else:
                # 无大盘数据，使用内部稳定性分析
                details['beta'] = "无大盘数据"
                
                # 波动率替代分析
                if price_volatility < 0.02:
                    score += 8
                    details['volatility_status'] = "极低波动"
                elif price_volatility < 0.03:
                    score += 5
                    details['volatility_status'] = "低波动"
            
            # === 价格稳定性评分 ===
            if -0.05 <= price_change <= 0.03:
                score += 8
                details['price_trend'] = "横盘整理"
            elif -0.10 <= price_change < -0.05:
                score += 5
                details['price_trend'] = "小幅回调"
            elif 0.03 < price_change <= 0.08:
                score += 4
                details['price_trend'] = "小幅上涨"
            else:
                details['price_trend'] = "大幅波动"
            
            return {
                "score": min(score, 25),
                "beta": beta,
                **details
            }
            
        except Exception as e:
            return {"score": 0, "error": str(e)}
    
    def _calculate_beta(
        self, 
        hist: pd.DataFrame, 
        benchmark: pd.DataFrame, 
        period: int
    ) -> Tuple[float, float]:
        """
        计算个股相对大盘的Beta系数及其趋势
        
        Beta = Cov(R_stock, R_market) / Var(R_market)
        
        Returns:
            (当前Beta, Beta趋势)
        """
        try:
            # 计算收益率
            stock_returns = hist['Close'].iloc[-period:].pct_change().dropna()
            market_returns = benchmark['Close'].iloc[-period:].pct_change().dropna()
            
            # 对齐长度
            min_len = min(len(stock_returns), len(market_returns))
            stock_returns = stock_returns.iloc[-min_len:]
            market_returns = market_returns.iloc[-min_len:]
            
            # 计算当前Beta
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            current_beta = covariance / market_variance if market_variance > 0 else 1.0
            
            # 计算Beta趋势（前半段vs后半段）
            half = min_len // 2
            if half >= 10:
                early_stock = stock_returns.iloc[:half]
                early_market = market_returns.iloc[:half]
                late_stock = stock_returns.iloc[half:]
                late_market = market_returns.iloc[half:]
                
                early_cov = np.cov(early_stock, early_market)[0, 1]
                early_var = np.var(early_market)
                early_beta = early_cov / early_var if early_var > 0 else 1.0
                
                late_cov = np.cov(late_stock, late_market)[0, 1]
                late_var = np.var(late_market)
                late_beta = late_cov / late_var if late_var > 0 else 1.0
                
                beta_trend = late_beta - early_beta
            else:
                beta_trend = 0
            
            return current_beta, beta_trend
            
        except Exception:
            return 1.0, 0.0
    
    def _score_accumulation_ending(
        self, 
        hist: pd.DataFrame, 
        current_price: float,
        current_volume: float
    ) -> Dict[str, Any]:
        """
        建仓尾声评分（价平量缩分析）
        
        核心特征：
        - 近10日成交量占30日比重下降
        - 价格仍维持在区间上沿
        - 筹码锁定性好
        
        Returns:
            包含 score, signal, details 的字典
        """
        try:
            period = self._config.accumulation_period
            score = 0
            signal = ""
            details = {}
            
            # === 1. 价平量缩分析 ===
            vol_30d = hist['Volume'].iloc[-period:].mean()
            vol_10d = hist['Volume'].iloc[-10:].mean()
            
            vol_ratio = vol_10d / vol_30d if vol_30d > 0 else 1
            details['vol_10d_30d_ratio'] = round(vol_ratio, 2)
            
            # 价格位置
            period_high = hist['High'].iloc[-period:].max()
            period_low = hist['Low'].iloc[-period:].min()
            price_position = (current_price - period_low) / (period_high - period_low) if period_high != period_low else 0.5
            details['price_position'] = round(price_position, 3)
            
            # 价平量缩评分
            if vol_ratio < 0.7 and price_position > 0.7:
                # 成交量萎缩 + 价格在上沿 = 典型建仓尾声
                score += 15
                signal = "价平量缩，筹码锁定"
                details['ending_status'] = "典型建仓尾声"
            elif vol_ratio < 0.8 and price_position > 0.5:
                score += 10
                signal = "量缩价稳"
                details['ending_status'] = "接近尾声"
            elif vol_ratio < 0.9:
                score += 5
                signal = "成交量下降"
                details['ending_status'] = "建仓中期"
            else:
                details['ending_status'] = "不明显"
            
            # === 2. 筹码锁定性分析 ===
            # 成交量递减率
            vol_series = hist['Volume'].iloc[-20:]
            vol_slope = self._calculate_slope(vol_series.values)
            details['vol_slope'] = round(vol_slope, 4)
            
            if vol_slope < -0.02:  # 成交量持续下降
                score += 5
                details['chip_lock'] = "筹码锁定良好"
            elif vol_slope < 0:
                details['chip_lock'] = "筹码趋于集中"
            
            # === 3. 地量确认 ===
            min_vol = hist['Volume'].iloc[-period:].min()
            min_vol_ratio = min_vol / vol_30d if vol_30d > 0 else 1
            details['min_vol_ratio'] = round(min_vol_ratio, 2)
            
            if min_vol_ratio < 0.4:
                score += 5
                details['ground_volume'] = True
            
            return {
                "score": min(score, 25),
                "signal": signal if signal else "无明显信号",
                **details
            }
            
        except Exception as e:
            return {"score": 0, "signal": "分析失败", "error": str(e)}
    
    def _score_volume_pattern(
        self, 
        hist: pd.DataFrame, 
        current_volume: float
    ) -> Dict[str, Any]:
        """成交量模式评分"""
        try:
            period = self._config.accumulation_period
            volumes = hist['Volume'].iloc[-period:]
            vol_ma = volumes.mean()
            
            score = 0
            details = {'vol_ma': int(vol_ma)}
            
            # 间歇性放量模式
            vol_ratios = volumes / vol_ma
            high_vol_days = (vol_ratios > 2.0).sum()
            low_vol_days = (vol_ratios < 0.7).sum()
            
            details['high_vol_days'] = int(high_vol_days)
            details['low_vol_days'] = int(low_vol_days)
            
            if 3 <= high_vol_days <= 8 and low_vol_days >= period * 0.4:
                score += 10
                details['volume_pattern'] = "间歇性放量"
            elif high_vol_days > 0 and low_vol_days >= period * 0.3:
                score += 6
                details['volume_pattern'] = "有放量痕迹"
            else:
                details['volume_pattern'] = "无明显模式"
            
            # 成交量趋势
            half = period // 2
            vol_first = volumes.iloc[:half].mean()
            vol_second = volumes.iloc[half:].mean()
            vol_trend = (vol_second / vol_first - 1) if vol_first > 0 else 0
            details['vol_trend_pct'] = round(vol_trend * 100, 2)
            
            if vol_trend > 0.15:
                score += 5
                details['vol_trend_status'] = "成交量递增"
            elif vol_trend > 0:
                score += 3
                details['vol_trend_status'] = "成交量微增"
            
            return {
                "score": min(score, 18),
                **details
            }
            
        except Exception as e:
            return {"score": 0, "error": str(e)}
    
    def _score_digging_pattern_enhanced(
        self, 
        hist: pd.DataFrame, 
        current_price: float,
        current_volume: float
    ) -> Dict[str, Any]:
        """
        挖坑形态评分（增强验证）
        
        验证条件：
        1. 价格站上"坑沿"（前期高点）
        2. 成交量温和放大
        3. OBV 同步创出新高
        """
        try:
            period = self._config.accumulation_period
            score = 0
            details = {}
            
            if len(hist) < period + 10:
                return {"score": 0, "pattern": "数据不足"}
            
            # === 1. 检测挖坑形态 ===
            # 找到前期高点（坑沿）
            prev_period = hist.iloc[-period-10:-period]
            pit_period = hist.iloc[-period:]
            
            prev_high = prev_period['High'].max()
            pit_low = pit_period['Low'].min()
            pit_high = pit_period['High'].max()
            
            # 坑深度
            pit_depth = (prev_high - pit_low) / prev_high if prev_high > 0 else 0
            details['pit_depth_pct'] = round(pit_depth * 100, 2)
            
            # 是否形成挖坑
            has_pit = pit_depth > 0.05 and current_price > pit_low * 1.03
            details['has_pit'] = has_pit
            
            if has_pit:
                score += 5
                details['pattern'] = "挖坑形态"
                
                # === 2. 验证是否站上坑沿 ===
                if current_price >= prev_high:
                    score += 5
                    details['above_rim'] = True
                    details['rim_status'] = "已站上坑沿"
                elif current_price >= prev_high * 0.97:
                    score += 3
                    details['rim_status'] = "接近坑沿"
                else:
                    details['rim_status'] = "仍在坑中"
                
                # === 3. 成交量温和放大 ===
                vol_ma = hist['Volume'].iloc[-period:].mean()
                recent_vol = hist['Volume'].iloc[-5:].mean()
                vol_ratio = recent_vol / vol_ma if vol_ma > 0 else 1
                details['recent_vol_ratio'] = round(vol_ratio, 2)
                
                if 1.0 <= vol_ratio <= 1.5:
                    score += 5
                    details['vol_confirmation'] = "温和放量确认"
                elif vol_ratio > 1.5:
                    score += 2
                    details['vol_confirmation'] = "放量突破"
                
                # === 4. OBV 验证 ===
                obv = self._calculate_obv(hist)
                current_obv = obv.iloc[-1]
                
                # 检查OBV是否创新高
                obv_high = obv.iloc[-period:-1].max()
                details['obv_new_high'] = current_obv > obv_high
                
                if current_obv > obv_high:
                    score += 5
                    details['obv_status'] = "OBV同步新高"
                else:
                    details['obv_status'] = "OBV未确认"
            else:
                # 检查其他形态
                price_range = (pit_high - pit_low) / pit_low if pit_low > 0 else 0
                details['price_range_pct'] = round(price_range * 100, 2)
                
                if price_range < 0.10:
                    score += 5
                    details['pattern'] = "箱体震荡"
                else:
                    details['pattern'] = "无明显形态"
            
            return {
                "score": min(score, 25),
                **details
            }
            
        except Exception as e:
            return {"score": 0, "pattern": "分析失败", "error": str(e)}
    
    def _score_money_flow_enhanced(
        self, 
        hist: pd.DataFrame, 
        current_price: float,
        current_volume: float
    ) -> Dict[str, Any]:
        """
        资金流向评分（CMF升级）
        
        Chaikin Money Flow (CMF) = Σ(MFV) / Σ(Volume)
        MFV = ((C-L) - (H-C)) / (H-L) * Volume
        
        CMF 上穿0轴 = 强烈的建仓结束信号
        """
        try:
            score = 0
            details = {}
            
            period = 20
            
            # === 1. 计算 CMF ===
            cmf_series = self._calculate_cmf(hist, period)
            current_cmf = cmf_series.iloc[-1]
            prev_cmf = cmf_series.iloc[-2] if len(cmf_series) >= 2 else current_cmf
            
            # 处理 NaN 值
            if pd.isna(current_cmf):
                current_cmf = 0.0
            if pd.isna(prev_cmf):
                prev_cmf = 0.0
            
            details['cmf'] = round(float(current_cmf), 3)
            details['cmf_prev'] = round(float(prev_cmf), 3)
            
            # CMF 评分
            if current_cmf > 0.1:
                score += 8
                details['cmf_status'] = "强势资金流入"
            elif current_cmf > 0:
                score += 5
                details['cmf_status'] = "资金流入"
            elif current_cmf > -0.1:
                score += 2
                details['cmf_status'] = "资金平衡"
            else:
                details['cmf_status'] = "资金流出"
            
            # CMF 上穿0轴信号
            if prev_cmf < 0 and current_cmf > 0:
                score += 7
                details['cmf_cross_zero'] = True
                details['cmf_signal'] = "CMF上穿0轴，建仓结束信号"
            elif prev_cmf < 0.05 and current_cmf > 0.05:
                score += 4
                details['cmf_signal'] = "CMF走强"
            
            # === 2. OBV 分析 ===
            obv = self._calculate_obv(hist)
            obv_current = obv.iloc[-1]
            obv_prev = obv.iloc[-10]
            
            obv_trend = (obv_current / obv_prev - 1) if obv_prev > 0 else 0
            details['obv_trend_pct'] = round(obv_trend * 100, 2)
            
            if obv_trend > 0.05:
                score += 5
                details['obv_status'] = "资金持续流入"
            elif obv_trend > 0:
                score += 3
                details['obv_status'] = "资金流入"
            
            # === 3. 量价配合 ===
            recent_close = hist['Close'].iloc[-5:]
            recent_vol = hist['Volume'].iloc[-5:]
            
            up_days = recent_close.diff() > 0
            down_days = recent_close.diff() < 0
            
            up_vol = recent_vol[up_days].mean() if up_days.any() else 0
            down_vol = recent_vol[down_days].mean() if down_days.any() else 0
            
            if up_vol > 0 and down_vol > 0:
                vol_ratio = up_vol / down_vol
                details['up_down_vol_ratio'] = round(vol_ratio, 2)
                
                if vol_ratio > 1.3:
                    score += 3
                    details['vp_status'] = "量价配合良好"
            
            return {
                "score": min(score, 25),
                **details
            }
            
        except Exception as e:
            return {"score": 0, "error": str(e)}
    
    def _calculate_cmf(self, hist: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        计算 Chaikin Money Flow (CMF)
        
        CMF = Σ(MFV) / Σ(Volume)
        MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume
        """
        try:
            # 计算 Money Flow Volume
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
            volume = hist['Volume']
            
            # 防止除零
            hl_range = high - low
            hl_range = hl_range.replace(0, np.nan)
            
            # CLV (Close Location Value)
            clv = ((close - low) - (high - close)) / hl_range
            clv = clv.fillna(0)
            
            # MFV
            mfv = clv * volume
            
            # CMF = 滚动求和
            cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            
            return cmf
            
        except Exception:
            return pd.Series([0] * len(hist), index=hist.index)
    
    def _calculate_obv(self, hist: pd.DataFrame) -> pd.Series:
        """计算 OBV 指标"""
        obv = [0]
        for i in range(1, len(hist)):
            if hist['Close'].iloc[i] > hist['Close'].iloc[i-1]:
                obv.append(obv[-1] + hist['Volume'].iloc[i])
            elif hist['Close'].iloc[i] < hist['Close'].iloc[i-1]:
                obv.append(obv[-1] - hist['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=hist.index)
    
    def _calculate_slope(self, series: np.ndarray) -> float:
        """计算序列斜率"""
        if len(series) < 2:
            return 0
        
        x = np.arange(len(series))
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(series)
        
        numerator = np.sum((x - x_mean) * (series - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator / y_mean if y_mean != 0 else 0
    
    def _calculate_ending_multiplier(self, ending_result: Dict) -> float:
        """计算建仓尾声信号乘数"""
        score = ending_result.get('score', 0)
        
        if score >= 20:
            return 1.2
        elif score >= 15:
            return 1.1
        elif score >= 10:
            return 1.0
        else:
            return 0.9
    
    def _calculate_confidence(
        self,
        total_score: float,
        resilience_result: Dict,
        ending_result: Dict,
        flow_result: Dict
    ) -> float:
        """计算置信度"""
        base_confidence = total_score / 100.0
        
        # Beta 抗跌加成（确保是数值类型）
        beta = resilience_result.get('beta')
        if beta is not None and isinstance(beta, (int, float)) and beta < 0.8:
            base_confidence += 0.05
        
        # CMF 信号加成
        if flow_result.get('cmf_cross_zero', False):
            base_confidence += 0.08
        
        # 建仓尾声信号加成
        if ending_result.get('ground_volume', False):
            base_confidence += 0.05
        
        return round(min(base_confidence, 0.95), 2)
    
    def _determine_accumulation_phase(self, total_score: float, ending_result: Dict) -> str:
        """确定建仓阶段"""
        if total_score >= 65:
            if ending_result.get('score', 0) >= 15:
                return "建仓尾声，即将启动"
            return "建仓后期"
        elif total_score >= 50:
            return "建仓中期"
        elif total_score >= self._config.min_score:
            return "建仓初期"
        else:
            return "无明显建仓"
    
    def _determine_alert_level(
        self,
        total_score: float,
        ending_result: Dict,
        flow_result: Dict
    ) -> str:
        """确定预警等级"""
        if total_score >= 60 and ending_result.get('score', 0) >= 15:
            if flow_result.get('cmf_cross_zero', False):
                return "高预警"
            return "中预警"
        elif total_score >= 50:
            return "低预警"
        else:
            return "无预警"


def quick_smart_money_check(
    hist: pd.DataFrame, 
    benchmark: pd.DataFrame = None,
    market_return: float = 0
) -> Dict:
    """便捷函数：快速检查主力建仓信号"""
    strategy = SmartMoneyAccumulationStrategy()
    context = StrategyContext(
        hist=hist, 
        info={}, 
        benchmark=benchmark,
        is_market_healthy=True,
        market_return=market_return
    )
    result = strategy.execute(context)
    
    return {
        "passed": result.passed,
        "confidence": result.confidence,
        "total_score": result.details.get('total_score', 0),
        "accumulation_phase": result.details.get('accumulation_phase', '未知'),
        "alert_level": result.details.get('alert_level', '无预警'),
        "details": result.details
    }