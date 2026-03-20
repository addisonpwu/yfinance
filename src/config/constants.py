"""
配置常量中心 - 所有默认值的唯一来源

原则：
1. 所有默认值只在这里定义一次
2. 其他模块通过导入引用这些常量
3. config.json 是运行时配置，这些是代码级默认值
"""

# ============================================
# API 配置
# ============================================
API_BASE_DELAY = 0.5
API_MAX_DELAY = 3.0
API_MIN_DELAY = 0.2
API_RETRY_ATTEMPTS = 3
API_MAX_WORKERS = 4

# ============================================
# 数据配置
# ============================================
DATA_MAX_CACHE_DAYS = 7
DATA_FLOAT_DTYPE = "float32"

# ============================================
# 分析配置
# ============================================
ANALYSIS_MIN_VOLUME_THRESHOLD = 100000
ANALYSIS_MIN_DATA_POINTS = 20

# ============================================
# 技术指标参数
# ============================================
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD_DEV = 2.0
ATR_PERIOD = 14
MA_PERIODS = [5, 10, 20, 50, 120, 200]
CMO_PERIOD = 14
WILLIAMS_R_PERIOD = 14
STOCHASTIC_PERIOD = 14
STOCHASTIC_SMOOTH_PERIOD = 3
VOLUME_Z_SCORE_PERIOD = 20

# ============================================
# 新增技术指标参数 (2026-03-04)
# ============================================
ADX_PERIOD = 14              # ADX 趋势强度周期
CMF_PERIOD = 20              # CMF 资金流量周期
VWAP_ANCHOR = 'D'           # VWAP 锚点 (D=日线)
STOCH_RSI_PERIOD = 14       # Stochastic RSI 周期
STOCH_RSI_SMOOTH_K = 3      # Stochastic RSI %K 平滑周期
STOCH_RSI_SMOOTH_D = 3      # Stochastic RSI %D 平滑周期

# ============================================
# 新增技术指标参数 (2026-03-18) - PRD升级
# ============================================

# Ichimoku 雲圖參數
ICHIMOKU_TENKAN_PERIOD = 9    # 轉換線週期
ICHIMOKU_KIJUN_PERIOD = 26    # 基準線週期
ICHIMOKU_SPAN_B_PERIOD = 52   # 遲行帶週期
ICHIMOKU_DISPLACEMENT = 26    # 投射週期

# 斐波那契回調位
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]  # 標準回調位
FIBONACCI_EXTENSIONS = [1.236, 1.382, 1.618, 2.0]     # 擴展位
FIBONACCI_LOOKBACK_PERIOD = 60                         # 默認回看天數

# 背離信號檢測參數
DIVERGENCE_LOOKBACK = 20     # 背離回看天數
DIVERGENCE_MIN_CHANGE = 0.03 # 最小價格變化（3%）

# 多時間框架參數
MULTI_TIMEFRAME_ENABLED = True
TIMEFRAMES = ['1d', '1w', '1m']  # 日線、週線、月線

# VaR 風險計算參數
VAR_CONFIDENCE_95 = 0.95     # 95% 置信度
VAR_CONFIDENCE_99 = 0.99     # 99% 置信度
VAR_HOLDING_PERIOD = 1       # 持有期（天）
VAR_LOOKBACK_DAYS = 252      # 回看天數（一年）

# 蒙特卡羅模擬參數
MONTE_CARLO_SIMULATIONS = 1000  # 模擬路徑數
MONTE_CARLO_DAYS = 30           # 模擬天數

# ============================================
# VIX 阈值（市场情绪）
# ============================================
VIX_LOW = 15.0          # < 15: 低波动/乐观
VIX_NORMAL = 20.0       # 15-20: 正常波动
VIX_HIGH = 30.0         # 20-30: 高波动/谨慎
VIX_PANIC = 40.0        # > 30: 极度恐慌

# ============================================
# 新闻配置
# ============================================
NEWS_TIMEOUT = 60000
NEWS_MAX_ITEMS = 10
NEWS_DAYS_BACK = 14
NEWS_CACHE_TTL_HOURS = 6

# ============================================
# 股票列表配置
# ============================================
STOCK_LIST_JSON_PATH = ""          # 股票列表 JSON 文件路径
STOCK_LIST_ENABLED = False         # 是否启用股票列表模式

# ============================================
# AI 配置
# ============================================
AI_API_TIMEOUT = 30
AI_MAX_DATA_POINTS = 100

# AI 提供商默认模型配置
DEFAULT_AI_PROVIDERS = {
    "iflow": {
        "default_model": "deepseek-v3.2",
        "available_models": ["deepseek-v3.2", "qwen3-max", "tstars2.0", "iflow-rome-30ba3b", "qwen3-coder-plus"]
    },
    "nvidia": {
        "default_model": "z-ai/glm5",
        "available_models": ["z-ai/glm5", "deepseek-ai/deepseek-v3.2", "qwen/qwen3.5-397b-a17b", "moonshotai/kimi-k2.5"]
    },
    "gemini": {
        "default_model": "gemini-2.5-flash",
        "available_models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-3-flash-preview", "gemini-3-pro-preview"]
    },
    "opencode": {
        "default_model": "glm-5",
        "available_models": ["glm-5"]
    }
}

# ============================================
# 回测配置
# ============================================
BACKTEST_INITIAL_CAPITAL = 100000.0
BACKTEST_COMMISSION = 0.001
BACKTEST_SLIPPAGE = 0.0005
BACKTEST_RISK_FREE_RATE = 0.02
BACKTEST_MAX_POSITION = 0.2

# ============================================
# 仓位管理配置
# ============================================
POSITION_MAX_POSITION = 0.15
POSITION_MAX_SECTOR = 0.30
POSITION_MAX_TOTAL = 0.80
POSITION_MIN_POSITION = 0.02
POSITION_KELLY_FRACTION = 0.5
POSITION_RISK_PER_TRADE = 0.02

# ============================================
# 策略默认参数
# ============================================
# 动量爆发策略
MOMENTUM_PRICE_BREAKOUT_THRESHOLD = 1.01
MOMENTUM_VOLUME_BURST_MULTIPLIER = 2.0
MOMENTUM_5D_THRESHOLD = 0.03
MOMENTUM_20D_THRESHOLD = 0.05
MOMENTUM_MIN_DATA_POINTS = 21
MOMENTUM_CONFIDENCE = 0.8

# 吸筹加速策略 v2.0 (增强版)
ACCUMULATION_PERIOD = 30
ACCUMULATION_VOLATILITY_THRESHOLD = 0.15
ACCUMULATION_VOLUME_TREND_PERIOD = 30
ACCUMULATION_BREAKOUT_THRESHOLD = 1.015
ACCUMULATION_VOLUME_RATIO_THRESHOLD = 2.5
ACCUMULATION_RSI_PREV_MIN = 40
ACCUMULATION_RSI_PREV_MAX = 60
ACCUMULATION_RSI_CURRENT_THRESHOLD = 65
ACCUMULATION_CONFIDENCE = 0.8

# 吸筹期增强参数
ACCUMULATION_PRICE_STABILITY_WEIGHT = 0.30      # 价格稳定性权重
ACCUMULATION_VOLUME_PATTERN_WEIGHT = 0.40       # 成交量分布权重
ACCUMULATION_VWAP_WEIGHT = 0.30                 # VWAP 权重

# 资金流确认参数
ACCUMULATION_MFI_PERIOD = 14                    # MFI 周期
ACCUMULATION_MFI_THRESHOLD = 50                 # MFI 阈值
ACCUMULATION_AD_LOOKBACK = 20                   # A/D Line 回看期

# RSI 斜率参数
ACCUMULATION_RSI_SLOPE_WINDOW = 3               # RSI 斜率计算窗口
ACCUMULATION_RSI_SLOPE_MIN = 1.0                # RSI 斜率最小阈值

# 加速信号评分权重
ACCUMULATION_BREAKOUT_WEIGHT = 0.25             # 价格突破权重
ACCUMULATION_VOLUME_WEIGHT = 0.25               # 量比权重
ACCUMULATION_MFI_WEIGHT = 0.25                  # MFI 权重
ACCUMULATION_AD_WEIGHT = 0.25                   # A/D Line 权重

# 波动率压缩策略
VOLATILITY_BB_PERIOD = 20
VOLATILITY_SQUEEZE_LOOKBACK = 100
VOLATILITY_SQUEEZE_PERCENTILE = 0.1
VOLATILITY_BREAKOUT_THRESHOLD = 0.02
VOLATILITY_VOLUME_MULTIPLIER = 1.5
VOLATILITY_VOLUME_PERIOD = 50
VOLATILITY_CONFIDENCE = 0.8

# 信号评分器
SCORER_TREND_WEIGHT = 0.25
SCORER_MOMENTUM_WEIGHT = 0.20
SCORER_VOLUME_WEIGHT = 0.15
SCORER_MARKET_WEIGHT = 0.20
SCORER_SECTOR_WEIGHT = 0.20
SCORER_PASS_THRESHOLD = 0.7
SCORER_MIN_DATA_POINTS = 50

# RSI 区间阈值
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_LOW = 40
RSI_HIGH = 60
RSI_STRONG = 80

# 底部反转策略 v2.0 (评分制)
BOTTOM_RSI_OVERSOLD_THRESHOLD = 30.0  # RSI超卖阈值
BOTTOM_RSI_RECOVERY_THRESHOLD = 3.0   # RSI回升幅度阈值
BOTTOM_SUPPORT_DISTANCE_THRESHOLD = 0.05  # 价格距离支撑位阈值(5%)
BOTTOM_VOLUME_CONTRACTION_THRESHOLD = 0.6  # 成交量萎缩阈值(60%)
BOTTOM_VOLUME_EXPANSION_MIN = 1.0     # 成交量温和放大最小值
BOTTOM_VOLUME_EXPANSION_MAX = 2.5     # 成交量温和放大最大值
BOTTOM_MIN_DATA_POINTS = 30           # 最小数据点数
BOTTOM_CONFIDENCE = 0.75              # 策略置信度
BOTTOM_MIN_SCORE = 50.0               # 最低通过分数

# ============================================
# 新增策略参数 (v2.0 - 早期信号策略)
# ============================================

# 启动前兆策略 v2.0 (增强版)
BREAKOUT_MA_CONVERGENCE_THRESHOLD = 0.03   # 均线粘合阈值(3%)
BREAKOUT_MA_CONVERGENCE_PERIOD = 5          # 粘合持续天数
BREAKOUT_PRICE_NEAR_HIGH_THRESHOLD = 0.95   # 接近期高点比例
BREAKOUT_CONSOLIDATION_PERIOD = 20          # 整理期天数
BREAKOUT_VOLUME_CONTRACTION_THRESHOLD = 0.6 # 成交量萎缩阈值
BREAKOUT_VOLUME_EXPANSION_MIN = 1.2         # 放量最小比例
BREAKOUT_VOLUME_EXPANSION_MAX = 2.5         # 放量最大比例
BREAKOUT_RSI_NEUTRAL_LOW = 45               # RSI中性区间下限
BREAKOUT_RSI_NEUTRAL_HIGH = 65              # RSI中性区间上限
BREAKOUT_MACD_THRESHOLD = 0.02              # MACD接近零轴阈值
BREAKOUT_BB_SQUEEZE_THRESHOLD = 0.15        # 布林带挤压阈值
BREAKOUT_MIN_SCORE = 50.0                   # 最低通过分数
BREAKOUT_STRONG_SIGNAL_SCORE = 70.0         # 强信号分数
BREAKOUT_MIN_DATA_POINTS = 30               # 最小数据点数

# TTM Squeeze 参数
BREAKOUT_TTM_BB_PERIOD = 20                 # 布林带周期
BREAKOUT_TTM_BB_STD_DEV = 2.0               # 布林带标准差倍数
BREAKOUT_TTM_KC_MULTIPLIER = 1.5            # 肯特纳通道ATR倍数
BREAKOUT_TTM_ATR_PERIOD = 14                # ATR 周期

# 形态识别参数
BREAKOUT_PATTERN_PERIOD = 20                # 形态识别周期
BREAKOUT_CHANNEL_CONVERGENCE_THRESHOLD = 0.7  # 通道收敛阈值
BREAKOUT_TRIANGLE_THRESHOLD = 0.3           # 三角形趋势强度阈值
BREAKOUT_FLAG_THRESHOLD = 0.08              # 旗形前期涨跌幅阈值

# 相对强度参数
BREAKOUT_RS_STRONG_THRESHOLD = 1.10         # 强势RS阈值
BREAKOUT_RS_WEAK_THRESHOLD = 0.95           # 弱势RS阈值
BREAKOUT_RS_LOOKBACK = 20                   # RS回看期

# 评分权重
BREAKOUT_MA_WEIGHT = 0.20                   # 均线粘合权重
BREAKOUT_PATTERN_WEIGHT = 0.25              # 蓄势形态权重
BREAKOUT_VOLUME_WEIGHT = 0.15               # 成交量权重
BREAKOUT_VOLATILITY_WEIGHT = 0.20           # 波动率压缩权重
BREAKOUT_TECHNICAL_WEIGHT = 0.10            # 技术指标权重
BREAKOUT_RS_WEIGHT = 0.10                   # 弱转强权重

# 主力建仓策略 v2.0 (增强版)
SMART_MONEY_ACCUMULATION_PERIOD = 30                # 分析周期
SMART_MONEY_PRICE_STABILITY_THRESHOLD = 0.05        # 价格稳定性阈值
SMART_MONEY_RELATIVE_STRENGTH_THRESHOLD = 0.0       # 相对强度阈值
SMART_MONEY_VOLUME_PATTERN_THRESHOLD = 0.3          # 成交量模式阈值
SMART_MONEY_INTERMITTENT_VOLUME_RATIO = 2.0         # 间歇性放量比例
SMART_MONEY_TURNOVER_RATE_LOW = 0.02                # 低换手率阈值
SMART_MONEY_TURNOVER_RATE_HIGH = 0.08               # 高换手率阈值
SMART_MONEY_MIN_SCORE = 45.0                        # 最低通过分数
SMART_MONEY_MIN_DATA_POINTS = 30                    # 最小数据点数

# Beta 分析参数
SMART_MONEY_BETA_LOW_THRESHOLD = 0.8                # 低Beta阈值
SMART_MONEY_BETA_HIGH_THRESHOLD = 1.2               # 高Beta阈值
SMART_MONEY_BETA_TREND_THRESHOLD = -0.1             # Beta下降趋势阈值

# 建仓尾声参数
SMART_MONEY_VOL_RATIO_THRESHOLD = 0.7               # 量缩比例阈值
SMART_MONEY_PRICE_POSITION_THRESHOLD = 0.7          # 价格位置阈值
SMART_MONEY_GROUND_VOLUME_RATIO = 0.4               # 地量比例阈值

# CMF 参数
SMART_MONEY_CMF_PERIOD = 20                         # CMF 周期
SMART_MONEY_CMF_STRONG_THRESHOLD = 0.1              # CMF 强势阈值
SMART_MONEY_CMF_WEAK_THRESHOLD = -0.1               # CMF 弱势阈值

# 挖坑形态参数
SMART_MONEY_PIT_DEPTH_THRESHOLD = 0.05              # 坑深度阈值
SMART_MONEY_PIT_VOL_RATIO_MIN = 1.0                 # 挖坑后量比最小值
SMART_MONEY_PIT_VOL_RATIO_MAX = 1.5                 # 挖坑后量比最大值

# 评分权重
SMART_MONEY_RESILIENCE_WEIGHT = 0.20                # 抗跌性权重
SMART_MONEY_ENDING_WEIGHT = 0.25                    # 建仓尾声权重
SMART_MONEY_VOLUME_WEIGHT = 0.15                    # 成交量权重
SMART_MONEY_PATTERN_WEIGHT = 0.20                   # 挖坑形态权重
SMART_MONEY_FLOW_WEIGHT = 0.20                      # 资金流向权重

# 技术共振策略
RESONANCE_MIN_COUNT = 3                 # 最少共振指标数
RESONANCE_STRONG_COUNT = 5              # 强共振指标数
RESONANCE_RSI_OVERSOLD = 30.0           # RSI 超卖阈值
RESONANCE_RSI_RECOVERY = 35.0           # RSI 回升阈值
RESONANCE_MACD_CROSS_THRESHOLD = 0.02   # MACD 接近金叉阈值
RESONANCE_BB_LOWER_POSITION = 0.2       # 布林带下轨位置阈值
RESONANCE_VOLUME_INCREASE_MIN = 1.2     # 成交量增加最小比例
RESONANCE_KDJ_OVERSOLD = 20.0           # KDJ 超卖阈值
RESONANCE_MIN_SCORE = 50.0              # 最低通过分数
RESONANCE_MIN_DATA_POINTS = 30          # 最小数据点数

# ============================================
# 启动捕捉策略 v1.0 (整合版)
# 整合启动前兆策略 + 主力建仓策略的核心优势
# ============================================

# 基础参数
LAUNCH_CAPTURE_PERIOD = 30                      # 分析周期
LAUNCH_MIN_SCORE = 60.0                         # 最低通过分数(提高门槛)
LAUNCH_STRONG_SIGNAL_SCORE = 75.0               # 强信号分数
LAUNCH_MIN_DATA_POINTS = 30                     # 最小数据点数

# 均线粘合参数
LAUNCH_MA_CONVERGENCE_THRESHOLD = 0.06          # 均线粘合阈值(6%) - 放宽以提高筛选通过率
LAUNCH_MA_SPREAD_EXTREME = 0.015                # 极度粘合(1.5%)
LAUNCH_MA_SPREAD_STRONG = 0.025                 # 强粘合(2.5%)
LAUNCH_MA_SPREAD_MODERATE = 0.04                # 粘合(4%)
LAUNCH_MA_SPREAD_WEAK = 0.06                    # 收敛(6%)

# 形态识别参数
LAUNCH_PATTERN_PERIOD = 20                      # 形态识别周期
LAUNCH_CHANNEL_CONVERGENCE = 0.7                # 通道收敛阈值
LAUNCH_TRIANGLE_THRESHOLD = 0.3                 # 三角形识别阈值
LAUNCH_PIT_DEPTH_MIN = 0.05                     # 挖坑深度最小值
LAUNCH_PIT_DEPTH_MAX = 0.20                     # 挖坑深度最大值

# 波动率压缩参数 (TTM Squeeze)
LAUNCH_TTM_BB_PERIOD = 20                       # 布林带周期
LAUNCH_TTM_BB_STD_DEV = 2.0                     # 布林带标准差
LAUNCH_TTM_KC_MULTIPLIER = 1.5                  # 肯特纳通道ATR倍数
LAUNCH_TTM_ATR_PERIOD = 14                      # ATR周期

# 资金信号参数
LAUNCH_CMF_PERIOD = 20                          # CMF周期
LAUNCH_CMF_STRONG_THRESHOLD = 0.1               # CMF强势阈值
LAUNCH_CMF_WEAK_THRESHOLD = -0.1                # CMF弱势阈值
LAUNCH_VOL_RATIO_THRESHOLD = 0.7                # 量缩比例阈值
LAUNCH_PRICE_POSITION_THRESHOLD = 0.7           # 价格位置阈值
LAUNCH_GROUND_VOLUME_RATIO = 0.4                # 地量比例阈值

# 抗跌特征参数 (Beta分析)
LAUNCH_BETA_LOW_THRESHOLD = 0.8                 # 低Beta阈值
LAUNCH_BETA_HIGH_THRESHOLD = 1.2                # 高Beta阈值
LAUNCH_RS_LOOKBACK = 20                         # 相对强度回看期
LAUNCH_RS_STRONG = 1.05                         # 相对强势阈值

# 技术指标参数
LAUNCH_RSI_NEUTRAL_LOW = 45                     # RSI中性区间下限
LAUNCH_RSI_NEUTRAL_HIGH = 65                    # RSI中性区间上限
LAUNCH_MACD_THRESHOLD = 0.02                    # MACD接近零轴阈值

# 成交量参数
LAUNCH_VOLUME_EXPANSION_MIN = 1.2               # 放量最小比例
LAUNCH_VOLUME_EXPANSION_MAX = 2.5               # 放量最大比例
LAUNCH_VOLUME_CONTRACTION = 0.6                 # 量缩阈值

# 评分权重 (7大维度)
LAUNCH_MA_WEIGHT = 0.15                         # 均线粘合权重
LAUNCH_PATTERN_WEIGHT = 0.20                    # 形态识别权重
LAUNCH_VOLATILITY_WEIGHT = 0.15                 # 波动率压缩权重
LAUNCH_MONEY_FLOW_WEIGHT = 0.20                 # 资金信号权重
LAUNCH_RESILIENCE_WEIGHT = 0.15                 # 抗跌特征权重
LAUNCH_TECHNICAL_WEIGHT = 0.10                  # 技术指标权重
LAUNCH_VOLUME_WEIGHT = 0.05                     # 成交量模式权重

# ============================================
# OBV 底背离 + BOLL 超卖策略参数 v1.0 (旧版)
# ============================================
OBV_BOLL_LLV_PERIOD = 20                        # LLV 周期（最低价回看）
OBV_BOLL_OBV_LOOKBACK = 5                       # OBV 对比回看天数
OBV_BOLL_VOLUME_RATIO_MIN = 0.4                 # 量比下限
OBV_BOLL_VOLUME_RATIO_MAX = 1.0                 # 量比上限
OBV_BOLL_MA_LONG_PERIOD = 120                   # 长期均线周期
OBV_BOLL_MIN_DATA_POINTS = 125                  # 最小数据点数(120+5)
OBV_BOLL_CONFIDENCE = 0.80                      # 策略置信度

# ============================================
# OBV 底背离 + BOLL 超卖策略参数 v2.0 (评分制)
# ============================================

# CMF 资金流向参数
OBV_BOLL_CMF_PERIOD = 20             # CMF 资金流向周期

# 评分权重
OBV_BOLL_OBV_WEIGHT = 30             # OBV底背离权重
OBV_BOLL_BOLL_WEIGHT = 25            # 布林带超卖权重
OBV_BOLL_VOLUME_WEIGHT = 15          # 量比权重
OBV_BOLL_TREND_WEIGHT = 15           # 长期趋势权重
OBV_BOLL_CMF_WEIGHT = 15             # CMF资金流向权重

# 评分阈值
OBV_BOLL_MIN_PASS_SCORE = 60         # 最低通过分数
OBV_BOLL_STRONG_SIGNAL_SCORE = 75    # 强信号分数

# OBV 底背离增强参数
OBV_BOLL_OBV_SHORT_PERIOD = 5        # 短区间回看天数
OBV_BOLL_OBV_LONG_PERIOD = 10        # 长区间回看天数
OBV_BOLL_OBV_SLOPE_MIN = 0.02        # OBV斜率最小阈值

# 布林带超卖细化参数
OBV_BOLL_BB_DEEP_OVERSOLD = 0.02     # 深度超卖阈值(跌破2%)
OBV_BOLL_BB_MID_OVERSOLD = 0.01      # 中度超卖阈值(跌破1%)
OBV_BOLL_BB_WIDTH_LOOKBACK = 100     # 带宽回看天数
OBV_BOLL_BB_SQUEEZE_PERCENTILE = 0.25 # 带宽压缩百分位

# 标准量比参数
OBV_BOLL_STD_VOLUME_PERIOD = 5       # 标准量比计算周期

# 止盈止损参数
OBV_BOLL_STOP_LOSS_ATR_MULT = 2.0    # 止损ATR倍数
OBV_BOLL_TAKE_PROFIT_ATR_MULT_1 = 3.0 # 止盈1 ATR倍数
OBV_BOLL_TAKE_PROFIT_ATR_MULT_2 = 5.0 # 止盈2 ATR倍数

# 市场环境调整
OBV_BOLL_UNHEALTHY_THRESHOLD_ADD = 10 # 市场不健康时阈值提高
OBV_BOLL_UNHEALTHY_CONFIDENCE_PENALTY = 0.85 # 市场不健康时置信度惩罚

# 评分系数 (v2.0 优化)
OBV_BOLL_OBV_STRONG_MULTIPLIER = 1.0       # 强背离系数
OBV_BOLL_OBV_MODERATE_MULTIPLIER = 0.67    # 中等背离系数
OBV_BOLL_OBV_WEAK_MULTIPLIER = 0.33        # 弱背离系数
OBV_BOLL_BOLL_DEEP_MULTIPLIER = 1.0        # 深度超卖系数
OBV_BOLL_BOLL_MID_MULTIPLIER = 0.72        # 中度超卖系数
OBV_BOLL_BOLL_LIGHT_MULTIPLIER = 0.4       # 轻度超卖系数
OBV_BOLL_BOLL_SQUEEZE_BONUS = 3.0          # 带宽压缩加分