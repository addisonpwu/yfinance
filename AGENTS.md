# 股票筛选与分析系统 - Agent 指南

## 项目概述

这是一个基于 Python 的股票筛选与分析系统，采用模块化设计，旨在根据多种可扩展的、经过专业交易逻辑强化的策略，从美国和香港股市中找出符合特定条件的股票。

### 核心特性
- **多市场支持**：支持美股（US）和港股（HK）市场
- **策略模块化**：动态加载策略，易于扩展
- **智能缓存系统**：高效的数据缓存机制，支持 TTL
- **AI 分析集成**：支持三大 AI 提供商（iFlow、NVIDIA NIM、Google Gemini）
- **多时间框架**：支持日线、小时线、分钟线数据
- **多模型AI分析**：支持多种AI模型进行股票分析，支持多模型投票共识
- **新闻整合**：自动获取 Yahoo Finance 新闻，整合到 AI 分析
- **HTML/PDF 报告**：现代化报告格式，支持浏览器打印，包含新闻展示
- **回测引擎**：完整的策略回测功能，支持蒙特卡洛验证
- **仓位管理**：动态仓位计算，支持 Kelly 公式和风险平价
- **市场环境识别**：自动识别趋势/震荡/高波动市场
- **宏观指标整合**：VIX、美债收益率、美元指数等宏观分析
- **速度模式**：支持快速/平衡/安全三种运行模式
- **信号评分器**：多维度信号综合评分系统
- **配置验证**：Pydantic 配置验证，环境变量安全管理
- **复权数据处理**：自动处理除权除息，确保价格连续性

---

## 完整数据处理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           系统启动验证                                        │
│  1. 加载 config.json 配置文件                                                │
│  2. Pydantic 配置验证（类型、范围、关系检查）                                  │
│  3. 加载 .env 环境变量（API 密钥等敏感信息）                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据获取阶段                                        │
│  1. 检查缓存版本 (version.txt)                                               │
│  2. 增量更新：只下载缺失/过期的数据                                            │
│  3. 获取股票列表 (港股: hkex.com.hk / 美股: Finviz)                          │
│  4. 获取大盘数据 (^HSI / ^GSPC)                                              │
│  5. 获取宏观指标 (VIX, TNX, FVX, DXY)                                        │
│  6. 获取新闻数据 (Yahoo Finance RSS, 支持缓存)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         数据预处理阶段                                        │
│  1. 复权处理 (auto_adjust=True)                                              │
│  2. 成交量阈值过滤 (min_volume_threshold)                                     │
│  3. 数据点数量检查 (min_data_points_threshold)                               │
│  4. 预计算技术指标:                                                           │
│     - 移动平均线 (MA_5, MA_10, MA_20, MA_50, MA_200)                         │
│     - RSI (RSI_14)                                                          │
│     - MACD (MACD, MACD_Signal, MACD_Hist)                                   │
│     - 布林带 (BB_Upper, BB_Middle, BB_Lower, BBP)                           │
│     - ATR (ATR_14)                                                          │
│     - 成交量均线 (Volume_MA_20)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          市场环境分析                                         │
│  MarketRegimeStrategy (整合宏观指标):                                        │
│  ├── 技术指标分析                                                            │
│  │   ├── 趋势强度 (近似 ADX)                                                 │
│  │   ├── 波动率水平 (年化波动率)                                              │
│  │   ├── 趋势方向 (价格与均线位置)                                            │
│  │   └── 市场健康得分                                                        │
│  └── 宏观指标分析                                                            │
│      ├── VIX 恐慌指数 (市场恐慌程度)                                          │
│      ├── 10年期美债收益率 (利率环境)                                          │
│      ├── 收益率曲线利差 (经济周期信号)                                         │
│      └── 美元指数 (资金流向)                                                  │
│                                                                              │
│  输出: regime (trending/mean_reverting/volatile)                            │
│       is_healthy, health_score, macro_risk_score                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          策略筛选阶段                                         │
│  对每只股票执行综合策略分析:                                                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ LaunchCaptureStrategy (启动捕捉策略) v1.0 (整合版)                    │   │
│  │                                                                      │   │
│  │ 整合"启动前兆策略"和"主力建仓策略"核心优势，捕捉即将启动的股票:        │   │
│  │                                                                      │   │
│  │ 七大维度评分:                                                         │   │
│  │ 1. 均线粘合 (15%)  - 多周期均线收敛，乘法前提                         │   │
│  │ 2. 形态识别 (20%)  - 三角形/旗形/楔形/挖坑                            │   │
│  │ 3. 波动率压缩 (15%) - TTM Squeeze 布林带收窄                         │   │
│  │ 4. 资金信号 (20%)  - CMF资金流 + 价平量缩 + OBV                       │   │
│  │ 5. 抗跌特征 (15%)  - Beta系数 + 相对强度                              │   │
│  │ 6. 技术指标 (10%)  - RSI + MACD 辅助确认                              │   │
│  │ 7. 成交量模式 (5%)  - 缩量蓄势/温和放量                               │   │
│  │                                                                      │   │
│  │ 通过条件: 总分 ≥ 60分 且 均线粘合通过                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  策略通过 → 进入 AI 分析阶段                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI 分析阶段                                         │
│  支持三大 AI 提供商 (通过 --provider 参数选择):                                │
│  - iFlow: 心流 AI (默认)                                                     │
│  - nvidia: NVIDIA NIM API                                                    │
│  - gemini: Google Gemini API                                                 │
│                                                                              │
│  分析流程:                                                                    │
│  1. 构建分析提示词 (价格数据 + 技术指标 + 基本面 + 新闻)                      │
│  2. AI 模型分析 (8维度技术分析):                                             │
│     - 趋势分析                                                               │
│     - 支撑与阻力                                                             │
│     - 动能指标分析                                                           │
│     - 成交量分析                                                             │
│     - 形态识别                                                               │
│     - 短期走势预测                                                           │
│     - 风险评估                                                               │
│     - 投资建议 (含价位建议)                                                  │
│  3. 缓存 AI 分析结果 (按数据哈希 + 模型 + 新闻哈希)                           │
│  4. 多模型投票共识 (可选，通过 --model all 启用)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          报告生成阶段                                         │
│  1. 实时写入 TXT 文件 (符合条件的股票)                                        │
│  2. 生成 HTML 报告:                                                          │
│     - 统计卡片 (筛选数量、策略命中、耗时)                                     │
│     - 股票卡片 (基本信息、评分、AI分析、近期新闻)                             │
│     - 策略标签 (各策略命中统计)                                              │
│  3. 可选: 生成 PDF (需安装 weasyprint)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 策略详细说明

### 1. MomentumBreakoutStrategy (动量爆发策略)

**策略类型**: 动量策略  
**适用场景**: 趋势市场，捕捉强势突破行情

#### 核心逻辑

```
价格突破 AND 量能爆发 AND 动量强度 → 通过
```

#### 检查条件

| 条件 | 计算方式 | 默认阈值 |
|------|----------|----------|
| **价格突破** | 当前价 > 20日最高价 × 阈值 | 1.02 (突破2%) |
| **量能爆发** | 当日成交量 > 20日均量 × 倍数 | 1.5倍 |
| **动量强度** | 5日涨幅 > 3% 且 20日涨幅 > 5% | 5日3%, 20日5% |

#### 代码示例

```python
from src.strategies.momentum_breakout_strategy import MomentumBreakoutStrategy
from src.core.strategies.strategy import StrategyContext

strategy = MomentumBreakoutStrategy()
context = StrategyContext(hist=hist, info=info, is_market_healthy=True)
result = strategy.execute(context)

if result.passed:
    print(f"通过! 置信度: {result.confidence}")
    print(f"价格突破: {result.details['price_breakout_details']}")
    print(f"量能爆发: {result.details['volume_burst_details']}")
```

#### 置信度计算

| 满足条件 | 置信度 |
|----------|--------|
| 价格 + 量能 + 动量 | 0.85 |
| 价格 + 量能 | 0.60 |
| 价格 + 动量 | 0.50 |
| 量能 + 动量 | 0.40 |
| 仅价格 | 0.25 |

---

### 2. VolatilitySqueezeStrategy (波动率压缩策略)

**策略类型**: 波动率策略  
**适用场景**: 横盘整理后的突破行情

#### 核心逻辑

```
布林带挤压 AND 突破确认 AND 量能配合 AND 市场健康 → 通过
```

#### 检查条件

| 条件 | 计算方式 | 默认阈值 |
|------|----------|----------|
| **布林带挤压** | BB宽度 < 近100日10分位数 | 百分位10% |
| **突破确认** | 价格 > MA20 且 当日涨幅 > 阈值 | 涨幅1.5% |
| **量能配合** | 当日成交量 > 50日均量 × 倍数 | 1.2倍 |
| **市场健康** | 大盘环境健康 | True |

#### 布林带宽度计算

```
BB_Width = (BB_Upper - BB_Lower) / BB_Middle
```

#### 代码示例

```python
from src.strategies.volatility_squeeze_strategy import VolatilitySqueezeStrategy

strategy = VolatilitySqueezeStrategy()
result = strategy.execute(context)

if result.passed:
    print(f"布林带宽度百分位: {result.details['squeeze_details']['bb_width_percentile']}")
    print(f"突破涨幅: {result.details['breakout_details']['daily_change_pct']}%")
```

#### 置信度计算

| 满足条件 | 置信度 |
|----------|--------|
| 挤压 + 突破 + 量能 + 市场OK | 0.85 |
| 挤压 + 突破 + 量能 | 0.70 |
| 挤压 + 突破 | 0.50 |
| 突破 + 量能 | 0.40 |

---

### 3. AccumulationAccelerationStrategy (主力吸筹加速策略)

**策略类型**: 吸筹策略  
**适用场景**: 主力资金建仓后的启动阶段

#### 核心逻辑

```
吸筹期 AND 量能趋势上升 AND 加速信号 AND RSI动态上穿 → 通过
```

#### 检查条件

| 条件 | 计算方式 | 默认阈值 |
|------|----------|----------|
| **吸筹期** | 近30日价格波动幅度 < 阈值 | 15% |
| **量能趋势** | 近15日均量 > 前15日均量 | 上升 |
| **加速信号** | 当前价 > 区间高点 × 1.02 且 量比 > 1.5 | 价2%, 量1.5倍 |
| **RSI动态上穿** | 前日RSI在40-60区间，当日RSI > 60 | 40-60→60+ |

#### 吸筹期识别

```
volatility = (High_max - Low_min) / Avg_Price
is_accumulating = volatility < 0.15
```

#### 代码示例

```python
from src.strategies.accumulation_acceleration_strategy import AccumulationAccelerationStrategy

strategy = AccumulationAccelerationStrategy()
result = strategy.execute(context)

if result.passed:
    print(f"吸筹期波动率: {result.details['accumulation_period_details']['volatility_pct']}%")
    print(f"RSI: {result.details['rsi_details']['rsi_current']}")
```

#### 置信度计算

| 满足条件 | 置信度 |
|----------|--------|
| 吸筹 + 量能趋势 + 加速 + RSI | 0.85 |
| 吸筹 + 加速 + RSI | 0.70 |
| 吸筹 + 量能趋势 + 加速 | 0.60 |
| 加速 + RSI | 0.50 |
| 吸筹 + 加速 | 0.40 |

---

### 4. SignalScorer (信号评分器)

**策略类型**: 综合评分  
**适用场景**: 多维度信号综合评估

#### 核心逻辑

```
加权综合得分 = Σ(维度得分 × 权重)
通过条件: 综合得分 >= 阈值 (默认 0.7)
```

#### 五大维度

| 维度 | 权重 | 评估内容 |
|------|------|----------|
| **趋势跟踪** | 25% | 价格与均线位置、均线排列、趋势持续性 |
| **动量突破** | 20% | 价格突破、RSI、MACD金叉 |
| **量能确认** | 15% | 成交量突破、量价配合、成交量趋势 |
| **市场回调** | 20% | 大盘健康状态、相对表现、市场环境 |
| **行业强度** | 20% | 行业信息、技术形态、布林带突破 |

#### 各维度评分细则

**趋势跟踪 (25%)**
```
- 价格在均线上方数量 (40%)
- 均线多头排列程度 (30%)
- 近20日涨幅持续性 (30%)
```

**动量突破 (20%)**
```
- 价格突破20日高点 (40%)
- RSI 在 50-70 区间 (30%)
- MACD 金叉 (30%)
```

**量能确认 (15%)**
```
- 成交量突破程度 (40%)
- 量价配合情况 (30%)
- 成交量上升趋势 (30%)
```

**市场回调 (20%)**
```
- 大盘健康状态 (30%)
- 相对大盘表现 (50%)
- 市场环境适配 (20%)
```

**行业强度 (20%)**
```
- 行业信息完整性 (30%)
- 技术形态强度 (70%)
```

#### 代码示例

```python
from src.strategies.signal_scorer import SignalScorer, quick_score

# 方式1: 完整使用
scorer = SignalScorer()
result = scorer.execute(context)
print(f"综合得分: {result.details['final_score']}")
print(f"各维度得分: {result.details['scores']}")

# 方式2: 快速评分
result = quick_score(hist, info, market_healthy=True)
print(f"通过: {result['passed']}, 得分: {result['score']}")
print(f"优势: {result['breakdown']['strengths']}")
print(f"劣势: {result['breakdown']['weaknesses']}")
```

#### 输出示例

```python
{
    "passed": True,
    "score": 0.72,
    "scores": {
        "trend_following": 0.80,
        "momentum_breakout": 0.65,
        "volume_confirmation": 0.55,
        "market_correction": 0.70,
        "sector_strength": 0.75
    },
    "breakdown": {
        "strengths": ["趋势: 80%", "形态: 75%"],
        "weaknesses": ["量能: 55%"],
        "recommendation": "温和看多 - 主要信号积极"
    }
}
```

---

### 5. MarketRegimeStrategy (市场环境识别策略)

**策略类型**: 市场分析  
**适用场景**: 判断大盘环境，指导策略选择

#### 核心逻辑

```
综合技术指标 + 宏观指标 → 市场环境判断
```

#### 输出分类

| 市场类型 | 特征 | 适合策略 |
|----------|------|----------|
| **trending** | 趋势强度 > 0.3，波动率适中 | 动量策略、趋势跟踪 |
| **mean_reverting** | 趋势强度弱，波动率低 | 均值回归、网格交易 |
| **volatile** | 波动率 > 40% | 减仓观望、期权策略 |

#### 技术指标分析

| 指标 | 计算方式 | 作用 |
|------|----------|------|
| **趋势强度** | 价格位置 + 均线斜率 | 判断趋势方向 |
| **ADX** | 平均趋向指数 | 判断趋势强度 |
| **波动率** | 收益率标准差 × √252 | 判断波动程度 |
| **夏普比率** | (收益-无风险利率) / 波动率 | 风险调整收益 |

#### 宏观指标整合

| 指标 | 代码 | 影响权重 |
|------|------|----------|
| **VIX 恐慌指数** | ^VIX | ±25分 |
| **10年期美债收益率** | ^TNX | ±10分 |
| **收益率曲线** | 10Y-5Y利差 | ±15分 |
| **美元指数** | DX-Y.NYB | ±10分 |

#### VIX 区间解读

| VIX 值 | 状态 | 市场含义 | 操作建议 |
|--------|------|----------|----------|
| < 12 | 自满 | 市场过度乐观 | 谨慎，可能回调 |
| 12-20 | 正常 | 波动正常 | 正常操作 |
| 20-30 | 升高 | 市场担忧增加 | 关注买入机会 |
| > 30 | 恐慌 | 市场恐慌 | 可能逆向买入 |

#### 代码示例

```python
from src.strategies.market_regime_strategy import MarketRegimeStrategy, get_market_regime

# 方式1: 完整分析
strategy = MarketRegimeStrategy(use_macro=True)
result = strategy.execute(context)

# 方式2: 快速获取
regime = get_market_regime(hist)
print(f"市场类型: {regime['regime']}")
print(f"健康得分: {regime['health_score']}")
print(f"宏观风险: {regime.get('macro_risk_score', 'N/A')}")
print(f"推荐策略: {regime.get('recommended_strategy', 'N/A')}")
```

#### 输出示例

```python
{
    "regime": "trending",
    "trend_strength": 0.45,
    "trend_direction": "up",
    "volatility_level": "medium",
    "volatility_pct": 22.5,
    "is_healthy": True,
    "health_score": 0.72,
    "confidence": 0.85,
    "macro_risk_score": 35,
    "macro_sentiment": "neutral",
    "macro_indicators": {
        "vix": {"current_value": 18.5, "trend": "down"},
        "treasury_yield": {"current_value": 4.25, "trend": "up"},
        "yield_curve": {"current_value": 0.45, "trend": "flat"},
        "dxy": {"current_value": 103.5, "trend": "neutral"}
    },
    "recommended_strategy": "balanced"
}
```

---

## 项目架构

```
yfinace/
├── main.py                 # 主入口点
├── config.json             # 配置文件
├── .env                    # 环境变量（敏感信息）
├── .env.example            # 环境变量模板
├── pyproject.toml          # 项目配置
├── setup.py                # 安装配置
├── requirements.txt        # 依赖文件
├── src/                    # 源代码目录
│   ├── ai/                 # AI分析模块
│   │   └── analyzer/       # AI分析器
│   │       ├── iflow_analyzer.py    # iFlow API 分析器
│   │       ├── nvidia_analyzer.py   # NVIDIA NIM API 分析器
│   │       ├── gemini_analyzer.py   # Google Gemini API 分析器
│   │       └── service.py           # AI 服务统一接口
│   ├── analysis/           # 分析模块
│   │   └── news_analyzer.py
│   ├── backtest/           # 回测引擎
│   │   ├── engine.py       # 回测引擎核心
│   │   └── metrics.py      # 回测指标计算
│   ├── config/             # 配置管理
│   │   ├── settings.py     # 配置类定义
│   │   └── config_validator.py  # 配置验证与密钥管理
│   ├── core/               # 核心模块
│   │   ├── models/         # 数据模型
│   │   │   └── entities.py
│   │   ├── services/       # 服务层
│   │   │   ├── analysis_service.py
│   │   │   ├── cache_version_manager.py
│   │   │   ├── market_data_service.py
│   │   │   ├── progress_tracker.py
│   │   │   ├── report_writer.py
│   │   │   └── stock_analyzer.py
│   │   └── strategies/     # 策略核心
│   │       ├── loader.py   # 策略加载器
│   │       └── strategy.py # 策略基类
│   ├── data/               # 数据处理
│   │   ├── cache/          # 缓存服务
│   │   │   └── cache_service.py
│   │   ├── external/       # 外部数据源
│   │   │   ├── stock_repository.py
│   │   │   └── macro_indicators.py
│   │   └── loaders/        # 数据加载器
│   │       ├── finviz_loader.py
│   │       ├── hk_loader.py
│   │       ├── us_loader.py
│   │       └── yahoo_loader.py
│   ├── risk/               # 风险管理
│   │   └── position_sizer.py
│   ├── strategies/         # 策略模块
│   │   ├── accumulation_acceleration_strategy.py
│   │   ├── market_regime_strategy.py
│   │   ├── momentum_breakout_strategy.py
│   │   ├── signal_scorer.py
│   │   ├── strategy_config.py
│   │   └── volatility_squeeze_strategy.py
│   └── utils/              # 工具模块
│       ├── exceptions.py
│       └── logger.py
├── data_cache/             # 数据缓存目录
│   ├── ai_analysis/
│   ├── HK/
│   └── US/
└── logs/                   # 日志文件目录
```

---

## 新闻功能说明

### 新闻获取

系统通过 Yahoo Finance RSS 自动获取股票相关新闻：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_news_items` | 每只股票最大新闻数量 | 10 |
| `days_back` | 回溯天数 | 14 |
| `cache_ttl_hours` | 缓存有效期（小时） | 6 |

### 新闻数据流

```
┌───────────────────────────────────────────────────────────────┐
│  YahooFinanceRepository.get_news(symbol, market)              │
│      ↓                                                         │
│  1. 检查缓存 (news_{symbol}_{date})                           │
│      ↓ (未命中)                                                │
│  2. RSS 获取 (feedparser)                                      │
│      ↓                                                         │
│  3. 写入缓存 (TTL: 6小时)                                      │
│      ↓                                                         │
│  返回: [{title, link, published, summary, publisher}]         │
└───────────────────────────────────────────────────────────────┘
                               ↓
┌───────────────────────────────────────────────────────────────┐
│  AI 分析整合                                                   │
│      ↓                                                         │
│  IFlowAIAnalyzer._format_news() → 加入 prompt                 │
│      ↓                                                         │
│  AI 模型参考新闻内容进行分析判断                                │
└───────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.data.loaders.yahoo_loader import YahooFinanceRepository

repo = YahooFinanceRepository()

# 获取港股新闻
news_hk = repo.get_news('0700', 'HK', days_back=14, max_items=10)
for item in news_hk:
    print(f"[{item['published']}] {item['title']}")

# 获取美股新闻
news_us = repo.get_news('AAPL', 'US', days_back=7, max_items=5)
```

### 新闻数据格式

```python
{
    "title": "新闻标题",
    "link": "https://hk.finance.yahoo.com/news/...",
    "published": "2026-02-19 06:24",
    "summary": "摘要内容...",
    "publisher": "Yahoo Finance",
    "source": "yahoo_rss"
}
```

---

## 配置参数

系统支持多种配置选项，通过 `config.json` 文件进行设置：

```json
{
  "speed_mode": "balanced",
  "api": {
    "base_delay": 0.4,
    "max_delay": 1.0,
    "min_delay": 0.4,
    "retry_attempts": 3,
    "max_workers": 2
  },
  "data": {
    "max_cache_days": 7,
    "float_dtype": "float32",
    "enable_cache": false,
    "enable_finviz": true
  },
  "analysis": {
    "enable_data_preprocessing": true,
    "min_volume_threshold": 100000,
    "min_data_points_threshold": 20
  },
  "technical_indicators": {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std_dev": 2,
    "atr_period": 14,
    "ma_periods": [5, 10, 20, 50, 200]
  },
  "strategies": {
    "momentum_breakout": {
      "price_breakout_threshold": 1.02,
      "volume_burst_multiplier": 1.5,
      "momentum_5d_threshold": 0.03,
      "momentum_20d_threshold": 0.05
    },
    "volatility_squeeze": {
      "bb_period": 20,
      "squeeze_percentile": 0.10,
      "breakout_change_threshold": 0.015
    },
    "accumulation_acceleration": {
      "accumulation_volatility_threshold": 0.15,
      "rsi_breakout_threshold": 60
    },
    "signal_scorer": {
      "weights": {
        "trend_following": 0.25,
        "momentum_breakout": 0.20,
        "volume_confirmation": 0.15,
        "market_correction": 0.20,
        "sector_strength": 0.20
      },
      "pass_threshold": 0.7
    },
    "market_regime": {
      "health_score_threshold": 0.6,
      "trend_strength_threshold": 0.3,
      "use_macro_indicators": true,
      "macro_weight": 0.3
    }
  },
  "news": {
    "max_news_items": 10,
    "days_back": 14,
    "cache_ttl_hours": 6
  },
  "ai": {
    "api_timeout": 30,
    "model": "deepseek-v3.2",
    "max_data_points": 100,
    "providers": {
      "iflow": {
        "default_model": "deepseek-v3.2",
        "available_models": ["deepseek-v3.2", "qwen3-max", "tstars2.0"]
      },
      "nvidia": {
        "default_model": "z-ai/glm5",
        "available_models": ["z-ai/glm5", "deepseek-ai/deepseek-v3.2"]
      },
      "gemini": {
        "default_model": "gemini-2.5-flash",
        "available_models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"]
      }
    }
  }
}
```

### 速度模式预设

| 模式 | 并行数 | 延迟 | 适用场景 |
|------|--------|------|----------|
| `fast` | 8 | 0.2s | 快速测试 |
| `balanced` | 4 | 0.5s | 日常使用（推荐） |
| `safe` | 2 | 1.0s | 避免 API 限流 |

---

## AI 提供商详细说明

系统支持三大 AI 提供商，可通过 `--provider` 参数选择：

### 提供商对比

| 提供商 | SDK | 默认模型 | 特点 |
|--------|-----|----------|------|
| **iFlow** | 自研 API | deepseek-v3.2 | 心流 AI，支持多模型投票 |
| **NVIDIA** | OpenAI SDK | z-ai/glm5 | NVIDIA NIM，支持 reasoning_content |
| **Gemini** | Google GenAI | gemini-2.5-flash | Google 最新模型，100万 token 上下文 |

### 多提供商支持

支持同时使用多个 AI 提供商进行联合分析：

```bash
# 使用多个提供商（逗号分隔）
python3 main.py --market HK --provider iflow,nvidia,gemini

# 多提供商 + 指定模型
python3 main.py --market HK --provider iflow,gemini --model deepseek-v3.2
```

多提供商分析时会：
1. 并行调用每个提供商的 AI 分析
2. 合并所有分析结果到一个报告中
3. 显示每个提供商的置信度和分析摘要

### iFlow 提供商

**支持模型**：
- `deepseek-v3.2` - 默认，DeepSeek 最新版本
- `qwen3-max` - 通义千问
- `tstars2.0` - TStars 模型
- `qwen3-coder-plus` - 代码增强版

**使用示例**：
```python
from src.ai.analyzer.service import AIAnalysisService

# 使用 iFlow
service = AIAnalysisService(provider='iflow')
result = service.analyze_stock(stock_data, hist, model='deepseek-v3.2')
```

### NVIDIA 提供商

**支持模型**：
- `z-ai/glm5` - 智谱 GLM-5
- `deepseek-ai/deepseek-v3.2` - DeepSeek on NVIDIA
- `qwen/qwen3.5-397b-a17b` - 通义千问
- `moonshotai/kimi-k2.5` - Moonshot Kimi

**特色功能**：
- 支持 `reasoning_content`（思考过程）展示
- 流式响应支持

**使用示例**：
```python
# 使用 NVIDIA
service = AIAnalysisService(provider='nvidia', enable_streaming=True)
result = service.analyze_stock(stock_data, hist, model='z-ai/glm5')
```

### Gemini 提供商

**支持模型**：
| 模型 | 描述 | 上下文 | 适用场景 |
|------|------|--------|----------|
| `gemini-2.5-flash` | **推荐默认**，最佳性价比 | 100万 tokens | 日常分析 |
| `gemini-2.5-pro` | 高级思考模型 | 100万 tokens | 复杂分析 |
| `gemini-2.5-flash-lite` | 极速轻量 | 100万 tokens | 快速筛选 |
| `gemini-2.0-flash` | 二代快速模型 | 100万 tokens | 通用场景 |
| `gemini-3-flash-preview` | 最新预览版 | 100万 tokens | 实验功能 |
| `gemini-3-pro-preview` | 最强模型预览版 | 100万 tokens | 复杂推理 |

**使用示例**：
```python
# 使用 Gemini
from src.ai.analyzer.gemini_analyzer import GeminiAIAnalyzer

analyzer = GeminiAIAnalyzer(enable_cache=True, enable_streaming=True)
result = analyzer.analyze(stock_data, hist, model='gemini-2.5-flash')

# 多模型投票共识
result = analyzer.analyze(stock_data, hist, model='all')
```

**API 密钥获取**：
1. 访问 [Google AI Studio](https://ai.google.dev/)
2. 点击 "Get API Key" 创建密钥
3. 将密钥添加到 `.env` 文件：
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### 多模型投票共识

所有提供商都支持使用 `model='all'` 启用多模型投票：

```bash
# 使用所有可用模型进行分析
python3 main.py --market HK --provider gemini --model all --symbol 0001.HK
```

共识机制会：
1. 并行调用多个模型
2. 统计看涨/看跌/中性票数
3. 计算一致率和综合置信度
4. 返回共识分析结果

---

## 配置与安全管理

### 环境变量配置

敏感信息通过 `.env` 文件管理：

```bash
# 复制模板
cp .env.example .env

# 编辑填写 API 密钥
IFLOW_API_KEY=your_actual_api_key_here
```

### .env 文件内容

```bash
# AI API 配置（必需）
IFLOW_API_KEY=your_iflow_api_key_here
IFLOW_API_BASE_URL=https://api.iflow.com/v1

# NVIDIA NIM API 密钥（用于 NVIDIA AI 股票分析）
# 获取地址: https://build.nvidia.com/
NVIDIA_API_KEY=nvapi-your_key_here

# Google Gemini API 密钥（用于 Gemini AI 股票分析）
# 获取地址: https://ai.google.dev/
GEMINI_API_KEY=your_gemini_api_key_here

# 日志级别
LOG_LEVEL=INFO

# 开发模式
DEV_MODE=false

# 代理配置（可选）
# HTTP_PROXY=http://127.0.0.1:7890
```

### 配置验证

```python
from src.config.config_validator import validate_startup, get_secrets_manager

# 启动验证
if not validate_startup():
    exit(1)

# 获取 API 密钥
secrets = get_secrets_manager()
api_key = secrets.get_iflow_api_key()
```

---

## 使用方法

### 基本命令

```bash
# 筛选港股
python3 main.py --market HK

# 使用特定AI模型
python3 main.py --market HK --model qwen3-max

# 速度模式
python3 main.py --market HK --speed fast

# 多提供商模式（同时使用多个AI进行分析）
python3 main.py --market HK --provider iflow,nvidia,gemini

# 多提供商 + 指定模型（使用首个模型）
python3 main.py --market HK --provider iflow,gemini --model deepseek-v3.2
```

### AI 提供商选择

```bash
# 使用 iFlow AI (默认)
python3 main.py --market HK --provider iflow

# 使用 NVIDIA NIM API
python3 main.py --market HK --provider nvidia --model z-ai/glm5

# 使用 Google Gemini API
python3 main.py --market HK --provider gemini --model gemini-2.5-flash

# 多模型投票共识
python3 main.py --market HK --provider gemini --model all
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--market` | 必需，市场代码 (US/HK) |
| `--provider` | AI 提供商 (iflow/nvidia/gemini)，可用逗号分隔多选，如 `iflow,nvidia,gemini` |
| `--model` | AI 模型选择，或使用 `all` 启用多模型投票 |
| `--no-cache-update` | 跳过缓存更新 |
| `--skip-strategies` | 跳过策略筛选 |
| `--symbol` | 指定单一股票代码 |
| `--interval` | 数据时段 (1d/1h/1m) |
| `--speed` | 速度模式 (fast/balanced/safe) |

---

## 安装与部署

### 环境要求
- Python 3.8+
- Windows / macOS / Linux

### 安装步骤

```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate  # macOS/Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填写 API 密钥

# 5. 安装项目
pip install -e .
```

### 依赖项

```
yfinance
pandas
requests
lxml
openpyxl
playwright
tenacity
pydantic>=2.0       # 配置验证（可选）
python-dotenv>=1.0  # 环境变量管理（可选）
feedparser>=6.0.0   # 新闻获取
openai>=1.0.0       # NVIDIA NIM API
google-genai>=0.3.0 # Google Gemini API
```

---

## 故障排除

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| API 限制 | 使用 `--speed safe` 或调整 `base_delay` |
| 数据缺失 | 删除缓存重新下载 |
| AI分析失败 | 检查 `.env` 中对应的 API 密钥（IFLOW_API_KEY / NVIDIA_API_KEY / GEMINI_API_KEY） |
| 内存不足 | 使用 `--speed safe` 减少并行数 |
| 配置验证失败 | 检查 `config.json` 格式和 `.env` 文件 |
| Gemini SDK 未安装 | 运行 `pip install google-genai` |
| NVIDIA SDK 未安装 | 运行 `pip install openai` |

### 调试方法

- 使用 `--symbol` 分析特定股票
- 查看 `logs/` 目录日志
- 使用 `--skip-strategies` 测试 AI 分析
- 使用 `--speed fast` 快速测试

---

## 回测引擎详细说明

### 核心配置 (BacktestConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_capital` | float | 100000 | 初始资金 |
| `commission` | float | 0.001 | 手续费率 (0.1%) |
| `slippage` | float | 0.0005 | 滑点 (0.05%) |
| `risk_free_rate` | float | 0.02 | 无风险利率 |
| `max_position` | float | 0.2 | 最大仓位 (20%) |
| `stop_loss` | float | None | 止损比例 |
| `take_profit` | float | None | 止盈比例 |
| `position_sizing` | str | "equal" | 仓位管理方式 |

### 交易记录 (Trade)

```python
@dataclass
class Trade:
    entry_date: datetime      # 入场时间
    entry_price: float        # 入场价格
    exit_date: datetime       # 出场时间
    exit_price: float         # 出场价格
    direction: str            # "long" | "short"
    quantity: float           # 交易数量
    pnl: float                # 盈亏金额
    pnl_pct: float            # 盈亏比例
    commission: float         # 手续费
    reason: str               # 平仓原因
```

### 回测指标 (BacktestMetrics)

#### 收益指标

| 指标 | 计算方式 |
|------|----------|
| `total_return` | (期末净值 / 期初净值) - 1 |
| `annualized_return` | (总收益 + 1)^(365/天数) - 1 |
| `monthly_returns` | 月度收益率序列 |

#### 风险指标

| 指标 | 计算方式 |
|------|----------|
| `annualized_volatility` | 日波动率 × √252 |
| `max_drawdown` | max((净值 - 累计最大净值) / 累计最大净值) |
| `max_drawdown_duration` | 最大回撤持续天数 |

#### 风险调整收益

| 指标 | 公式 |
|------|------|
| `sharpe_ratio` | (年化收益 - 无风险利率) / 年化波动率 |
| `sortino_ratio` | (年化收益 - 无风险利率) / 下行波动率 |
| `calmar_ratio` | 年化收益 / |最大回撤| |

#### 交易统计

| 指标 | 说明 |
|------|------|
| `win_rate` | 盈利交易占比 |
| `profit_factor` | 总盈利 / 总亏损 |
| `expectancy` | 胜率×平均盈利 - (1-胜率)×平均亏损 |

### 使用示例

```python
from src.backtest.engine import BacktestEngine, BacktestConfig, backtest, print_result
from src.backtest.metrics import calculate_metrics, print_metrics

# 定义策略函数
def my_strategy(data):
    """返回: 1=买入, -1=卖出, 0=持有"""
    if data['Close'].iloc[-1] > data['Close'].iloc[-20:].mean():
        return 1
    elif data['Close'].iloc[-1] < data['Close'].iloc[-20:].mean() * 0.95:
        return -1
    return 0

# 方式1: 快速回测
result = backtest(data, my_strategy, initial_capital=100000)
print_result(result)

# 方式2: 高级配置
config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    stop_loss=0.08,
    take_profit=0.15,
    position_sizing="volatility"
)
engine = BacktestEngine(config)
result = engine.run(data, my_strategy)

# 查看交易详情
for trade in result.trades[:5]:
    print(f"{trade.direction}: {trade.entry_date} @ {trade.entry_price:.2f} -> {trade.exit_date} @ {trade.exit_price:.2f}, PnL: {trade.pnl:.2f}")

# 计算指标
metrics = calculate_metrics(result.equity_curve)
print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")
```

### 蒙特卡洛测试

```python
# 评估策略稳健性
mc_result = engine.monte_carlo_test(result, n_runs=1000)

print(f"正收益概率: {mc_result['prob_positive']:.2%}")
print(f"夏普>1概率: {mc_result['prob_sharpe_gt_1']:.2%}")
print(f"回撤<10%概率: {mc_result['prob_max_dd_lt_10']:.2%}")
print(f"收益95%置信区间: [{mc_result['total_return']['ci_low']:.2%}, {mc_result['total_return']['ci_high']:.2%}]")
```

### Walk-Forward 验证

```python
# 滑动窗口样本外测试
wf_results = engine.walk_forward_validation(
    data, 
    my_strategy,
    train_window=252,  # 1年训练
    test_window=63,    # 3个月测试
    step=21            # 每月滑动
)

for i, result in enumerate(wf_results):
    print(f"窗口 {i+1}: 收益 {result.metrics['total_return']:.2%}")
```

---

## 仓位管理详细说明

### 核心配置 (PositionConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_position` | float | 0.15 | 单只股票最大仓位 (15%) |
| `max_sector` | float | 0.30 | 单个行业最大仓位 (30%) |
| `max_total` | float | 0.80 | 总仓位上限 (80%) |
| `min_position` | float | 0.02 | 最小仓位 (2%) |
| `kelly_fraction` | float | 0.5 | Kelly 分数 (减半) |
| `risk_per_trade` | float | 0.02 | 每笔交易风险 (2%) |

### 仓位计算方法

#### 1. Kelly 公式

```
Kelly % = W - (1-W) / R

其中:
W = 胜率
R = 平均盈利 / 平均亏损

建议使用 Kelly/2 降低风险
```

```python
from src.risk.position_sizer import kelly_sizer

# 从历史收益率计算
kelly_pct, analysis = kelly_sizer(returns_series)
print(f"建议仓位: {kelly_pct:.1%}")
print(f"胜率: {analysis['win_rate']:.2%}")
print(f"盈亏比: {analysis['wl_ratio']:.2f}")
print(f"风险提示: {analysis['risk_warning']}")
```

#### 2. 波动率调整仓位

```python
from src.risk.position_sizer import calc_position

# 根据信心度和波动率计算
result = calc_position(
    confidence=0.75,      # 信号信心度
    price=100.0,          # 当前价格
    volatility=0.25,      # 年化波动率
    account_balance=100000
)

print(f"仓位金额: {result['position_size']}")
print(f"仓位比例: {result['position_pct']:.1%}")
print(f"股数: {result['shares']}")
```

#### 3. 风险平价

```python
from src.risk.position_sizer import PositionSizer

sizer = PositionSizer()

# 各资产波动率
volatilities = {
    'AAPL': 0.25,
    'MSFT': 0.22,
    'GOOGL': 0.28
}

# 风险平价分配
weights = sizer.volatility_parity_position(volatilities, total_target_risk=0.15)
# {'AAPL': 0.32, 'MSFT': 0.36, 'GOOGL': 0.29}
```

#### 4. 信心加权仓位

```python
# 根据信号信心度分配
confidences = {
    'AAPL': 0.85,
    'MSFT': 0.70,
    'GOOGL': 0.60
}

weights = sizer.confidence_weighted_position(confidences)
# 信心高的股票获得更大仓位
```

### 风险评估

```python
from src.risk.position_sizer import risk_summary

positions = {
    'AAPL': 15000,
    'MSFT': 12000,
    'GOOGL': 8000
}

summary = risk_summary(positions, account_balance=100000)
print(f"总仓位: {summary['total_exposure_pct']:.1%}")
print(f"持仓数量: {summary['num_positions']}")
print(f"现金比例: {summary['cash_pct']:.1%}")
print(f"风险等级: {summary['risk_level']}")
```

---

## 技术指标计算公式

### 移动平均线 (MA)

```
SMA(n) = Σ(Close_i) / n    i = 1 to n

EMA(n) = Close × k + EMA_prev × (1-k)
k = 2 / (n + 1)
```

### 相对强弱指数 (RSI)

```
RS = 平均涨幅 / 平均跌幅
RSI = 100 - (100 / (1 + RS))

默认周期: 14
超买区: > 70
超卖区: < 30
```

### MACD

```
DIF = EMA(12) - EMA(26)
DEA = EMA(DIF, 9)
MACD柱 = (DIF - DEA) × 2

金叉: DIF上穿DEA (买入信号)
死叉: DIF下穿DEA (卖出信号)
```

### 布林带 (Bollinger Bands)

```
中轨 = SMA(20)
上轨 = 中轨 + 2 × Std(20)
下轨 = 中轨 - 2 × Std(20)
带宽 = (上轨 - 下轨) / 中轨

挤压: 带宽 < 历史10分位数
突破: 价格突破上轨
```

### ATR (平均真实波幅)

```
TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
ATR(14) = SMA(TR, 14)

用途: 止损距离 = 入场价 - 2×ATR
```

### ADX (平均趋向指数)

```
+DM = max(High - PrevHigh, 0)
-DM = max(PrevLow - Low, 0)

+DI = 100 × SMA(+DM, 14) / SMA(TR, 14)
-DI = 100 × SMA(-DM, 14) / SMA(TR, 14)

DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = SMA(DX, 14)

趋势强度:
ADX < 20: 无趋势
ADX 20-40: 趋势形成
ADX > 40: 强趋势
```

### CMO (钱德动量摆动指标)

```
CMO = 100 × (Su - Sd) / (Su + Sd)

Su = 上涨幅度之和
Sd = 下跌幅度之和

超买: > 50
超卖: < -50
```

---

## HTML 报告格式说明

### 报告结构

```
┌─────────────────────────────────────┐
│           页面头部                   │
│  标题、市场、日期、耗时              │
├─────────────────────────────────────┤
│           统计卡片                   │
│  筛选数量 | 策略数 | 涨跌 | 耗时     │
├─────────────────────────────────────┤
│           策略标签                   │
│  各策略命中统计                      │
├─────────────────────────────────────┤
│           股票卡片列表               │
│  ┌─────────────────────────────┐    │
│  │ 代码 | 名称 | 行业          │    │
│  │ 市值 | PE | 涨跌幅          │    │
│  │ 策略标签                    │    │
│  │ 评分显示                    │    │
│  │ AI 分析摘要                 │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### 样式特性

- **响应式设计**: 适配不同屏幕宽度
- **打印优化**: 浏览器 Cmd+P 导出 PDF
- **深色主题**: 支持系统深色模式
- **卡片阴影**: 现代化视觉效果

### 自定义报告

```python
from src.core.services.report_writer import ReportWriter

# 创建报告写入器
writer = ReportWriter(
    filename="custom_report",  # 自动处理扩展名
    market="HK",
    output_format='both'       # 'txt', 'html', 'both'
)

writer.initialize()

# 写入单个结果
writer.write_stock_result({
    'symbol': '0001.HK',
    'name': '长和',
    'strategies': ['动量爆发策略', '信号评分器'],
    'score': 0.75,
    'ai_analysis': '...'
})

# 写入摘要
writer.write_summary(results, 'HK')
```

---

## API 限流处理策略

### 延迟策略

```python
# config.json 中的 API 配置
{
  "api": {
    "base_delay": 0.5,    # 基础延迟
    "max_delay": 2.0,     # 最大延迟
    "min_delay": 0.1,     # 最小延迟
    "retry_attempts": 3   # 重试次数
  }
}
```

### 动态延迟调整

```python
# 根据响应自动调整延迟
def adjust_delay(current_delay, response_status):
    if response_status == 429:  # Too Many Requests
        return min(current_delay * 2, max_delay)
    elif response_status == 200:
        return max(current_delay * 0.9, min_delay)
    return current_delay
```

### 速度模式

| 模式 | 并行数 | 延迟 | 每分钟请求 |
|------|--------|------|-----------|
| `fast` | 8 | 0.2s | ~480 |
| `balanced` | 4 | 0.5s | ~240 |
| `safe` | 2 | 1.0s | ~120 |

### 错误处理

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def fetch_with_retry(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

---

## 日志系统说明

### 日志级别

| 级别 | 用途 |
|------|------|
| DEBUG | 详细调试信息 |
| INFO | 正常运行信息 |
| WARNING | 警告信息 |
| ERROR | 错误信息 |

### 日志文件位置

```
logs/
├── analysis_YYYY-MM-DD.log    # 分析日志
├── error_YYYY-MM-DD.log       # 错误日志
└── performance.log            # 性能日志
```

### 日志格式

```
2026-02-19 10:30:15,123 - INFO - [动量爆发策略] 0001.HK 通过筛选
2026-02-19 10:30:16,456 - WARNING - API 请求延迟增加
2026-02-19 10:30:17,789 - ERROR - 获取 0002.HK 数据失败: Timeout
```

### 使用日志

```python
from src.utils.logger import get_analysis_logger

logger = get_analysis_logger()

logger.info("开始分析股票")
logger.warning("数据质量较差")
logger.error("处理失败", exc_info=True)
```

---

## 常见使用场景示例

### 场景1: 日常选股

```bash
# 筛选港股，使用平衡模式
python3 main.py --market HK --speed balanced
```

### 场景2: 快速测试新策略

```bash
# 使用快速模式测试
python3 main.py --market HK --speed fast --symbol 0001.HK
```

### 场景3: AI 分析特定股票

```bash
# 跳过策略，直接 AI 分析
python3 main.py --market HK --skip-strategies --symbol 0001.HK
```

### 场景4: 多模型对比分析

```bash
# 使用所有 AI 模型分析
python3 main.py --market HK --model all --symbol 0001.HK
```

### 场景5: 使用 Gemini API 分析

```bash
# 使用 Gemini 快速模型
python3 main.py --market HK --provider gemini --symbol 0001.HK

# 使用 Gemini Pro 高级模型
python3 main.py --market HK --provider gemini --model gemini-2.5-pro
```

### 场景6: 短线交易分析

```bash
# 使用小时线数据
python3 main.py --market HK --interval 1h
```

### 场景7: 自定义回测

```python
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategies.momentum_breakout_strategy import MomentumBreakoutStrategy

# 定义策略
def momentum_backtest(data):
    strategy = MomentumBreakoutStrategy()
    # 简化返回信号
    if data['Close'].iloc[-1] > data['High'].iloc[-20:].max() * 1.02:
        return 1
    return 0

# 运行回测
config = BacktestConfig(
    initial_capital=100000,
    stop_loss=0.08,
    take_profit=0.15
)
engine = BacktestEngine(config)
result = engine.run(data, momentum_backtest)

# 输出结果
print(f"总收益: {result.metrics['total_return']:.2%}")
print(f"夏普比率: {result.metrics['sharpe_ratio']:.3f}")
print(f"最大回撤: {result.metrics['max_drawdown']:.2%}")
```

### 场景8: 动态仓位计算

```python
from src.risk.position_sizer import PositionSizer, kelly_sizer
from src.strategies.signal_scorer import quick_score

# 计算信号得分
score_result = quick_score(hist, info, market_healthy=True)

# 根据得分计算仓位
sizer = PositionSizer()
position = sizer.calculate_position(
    confidence=score_result['score'],
    price=hist['Close'].iloc[-1],
    volatility=hist['Close'].pct_change().std() * (252**0.5),
    account_balance=500000
)

print(f"建议买入: {position['shares']} 股")
print(f"仓位比例: {position['position_pct']:.1%}")
```

### 场景9: 宏观环境判断

```python
from src.data.external.macro_indicators import get_macro_analysis, is_high_risk_environment

# 获取宏观分析
analysis = get_macro_analysis()

print(f"风险评分: {analysis['risk_score']}/100")
print(f"市场情绪: {analysis['sentiment']}")
print(f"推荐策略: {analysis['recommended_strategy']}")

# 检查是否高风险
if is_high_risk_environment():
    print("⚠️ 当前为高风险环境，建议降低仓位")
```

---

## 近期更新日志

### 2026-03-04
- ✨ 多提供商支持升级
  - `--provider` 参数支持逗号分隔多选 (如 `iflow,nvidia,gemini`)
  - 多提供商分析时并行调用并合并结果
- ✨ AI 提示词优化
  - 所有分析器添加增强版四维度分析框架
  - 市场环境评估 (30%)、技术面分析 (40%)、新闻影响分析 (20%)、风险评估 (10%)
  - 技术信号详细指引：RSI、MACD、均线、布林带、量价配合
  - 所有分析器添加新闻分析支持（格式化 + 分析指引）
- ✨ 新闻分析增强
  - 按时间倒序排列，最新新闻在前
  - 添加新闻摘要内容
  - 添加利好/利空判断标准指引

### 2026-02-21
- ✨ 新增 Google Gemini API 分析器 (`gemini_analyzer.py`)
  - 支持 Google GenAI SDK 调用 Gemini API
  - 支持 6 种 Gemini 模型（gemini-2.5-flash、gemini-2.5-pro 等）
  - 支持流式响应和多模型投票共识
  - 支持预测追踪和缓存
- ✨ AI 服务架构升级为三提供商支持
  - iFlow (心流 AI) - 默认
  - NVIDIA NIM API
  - Google Gemini API
- 🏗️ 更新配置系统支持 Gemini 提供商
  - `config.json` 添加 `ai.providers.gemini` 配置
  - `constants.py` 添加 Gemini 默认模型列表
  - `settings.py` 的 `AIProvidersConfig` 支持 gemini 属性
- ✨ 环境变量支持新增 `GEMINI_API_KEY`
- 📦 依赖更新：添加 `google-genai>=0.3.0`
- 📝 更新 AGENTS.md 文档

### 2026-02-19
- ✨ 新增 NVIDIA API 分析器 (`nvidia_analyzer.py`)
  - 支持 OpenAI SDK 调用 NVIDIA NIM API
  - 支持多模型投票分析
  - 支持预测追踪和缓存
- ✨ 新增 `--provider` 命令行参数 (iflow/nvidia)
- 🏗️ 创建配置常量中心 (`src/config/constants.py`)
  - 统一管理所有默认值，单一来源原则
  - 包含 API、技术指标、VIX 阈值、策略参数等
- 🏗️ 重构配置管理
  - `settings.py` 从 constants.py 导入默认值
  - `strategy_config.py` 从 constants.py 导入策略参数
  - 消除多处重复定义的默认值
- 🐛 修复 `news.days_back` 和 `news.cache_ttl_hours` 配置未使用问题
- 🎨 VIX 阈值 (15/20/30) 抽取为常量，替代硬编码
- ✨ 新增新闻获取功能 (Yahoo Finance RSS)
- ✨ 新闻数据整合到 AI 分析 prompt
- ✨ HTML 报告新增"近期新闻"展示区域
- ✨ 新闻数据缓存支持 (TTL 可配置)
- 🐛 修复 AI 缓存键未包含新闻哈希的问题
- 🐛 移除 AI 分析结果的字符截断限制
- ✨ 新增宏观指标服务 (`macro_indicators.py`)
- ✨ 市场环境识别整合宏观指标
- ✨ 配置验证模块支持 Pydantic 和降级模式
- ✨ 环境变量安全管理 (`.env` 支持)
- ✨ 缓存服务支持 TTL 参数
- 🎨 HTML 报告可视化优化：评分仪表盘、方向指示器、可折叠AI分析
- 🎨 HTML 报告添加策略颜色标签、股票头部颜色根据方向变化
- 🎨 TXT 报告结构化展示：Unicode分隔线、图标、AI摘要提取
- 🎨 TXT 报告摘要列表添加详细版本和风险提示
- ⚙️ 禁用 AI 分析缓存（每次重新分析）
- ⚙️ 禁用新闻缓存（每次重新获取）
- 📝 更新 AGENTS.md 文档

### 2026-02-18
- ✨ 复权数据处理增强
- ✨ 添加 `get_adjustment_info()` 方法
- ✨ 技术指标计算时验证复权状态