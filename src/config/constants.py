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
MA_PERIODS = [5, 10, 20, 50, 200]
CMO_PERIOD = 14
WILLIAMS_R_PERIOD = 14
STOCHASTIC_PERIOD = 14
STOCHASTIC_SMOOTH_PERIOD = 3
VOLUME_Z_SCORE_PERIOD = 20

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

# 吸筹加速策略
ACCUMULATION_PERIOD = 30
ACCUMULATION_VOLATILITY_THRESHOLD = 0.15
ACCUMULATION_VOLUME_TREND_PERIOD = 30
ACCUMULATION_BREAKOUT_THRESHOLD = 1.015
ACCUMULATION_VOLUME_RATIO_THRESHOLD = 2.5
ACCUMULATION_RSI_PREV_MIN = 40
ACCUMULATION_RSI_PREV_MAX = 60
ACCUMULATION_RSI_CURRENT_THRESHOLD = 65
ACCUMULATION_CONFIDENCE = 0.8

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
