# 股票筛选与分析系统 - Agent 指南

## 项目概述

这是一个基于 Python 的股票筛选与分析系统，采用模块化设计，旨在根据多种可扩展的、经过专业交易逻辑强化的策略，从美国和香港股市中找出符合特定条件的股票。

### 核心特性
- **多市场支持**：支持美股（US）和港股（HK）市场
- **策略模块化**：动态加载策略，易于扩展
- **智能缓存系统**：高效的数据缓存机制，提升重复运行效率
- **AI 分析集成**：集成 iFlow API 进行技术分析
- **多时间框架**：支持日线、小时线、分钟线数据
- **多模型AI分析**：支持多种AI模型进行股票分析
- **HTML/PDF 报告**：现代化报告格式，支持浏览器打印
- **回测引擎**：完整的策略回测功能，支持蒙特卡洛验证
- **仓位管理**：动态仓位计算，支持 Kelly 公式和风险平价
- **市场环境识别**：自动识别趋势/震荡/高波动市场
- **速度模式**：支持快速/平衡/安全三种运行模式
- **信号评分器**：多维度信号综合评分系统

## 项目架构

```
yfinace/
├── main.py                 # 主入口点
├── config.json             # 配置文件
├── pyproject.toml          # 项目配置
├── setup.py                # 安装配置
├── requirements.txt        # 依赖文件
├── src/                    # 源代码目录
│   ├── ai/                 # AI分析模块
│   │   └── analyzer/       # AI分析器
│   │       ├── iflow_analyzer.py
│   │       └── service.py
│   ├── analysis/           # 分析模块
│   │   └── news_analyzer.py
│   ├── backtest/           # 回测引擎
│   │   ├── engine.py       # 回测引擎核心
│   │   └── metrics.py      # 回测指标计算
│   ├── config/             # 配置管理
│   │   └── settings.py     # 配置类定义
│   ├── core/               # 核心模块
│   │   ├── models/         # 数据模型
│   │   │   └── entities.py
│   │   ├── services/       # 服务层
│   │   │   ├── analysis_service.py
│   │   │   ├── cache_version_manager.py
│   │   │   ├── market_data_service.py
│   │   │   ├── progress_tracker.py
│   │   │   ├── report_writer.py   # 支持 HTML/PDF
│   │   │   └── stock_analyzer.py
│   │   └── strategies/     # 策略核心
│   │       ├── loader.py   # 策略加载器
│   │       └── strategy.py # 策略基类
│   ├── data/               # 数据处理
│   │   ├── cache/          # 缓存服务
│   │   │   └── cache_service.py
│   │   ├── external/       # 外部数据源
│   │   │   └── stock_repository.py
│   │   └── loaders/        # 数据加载器
│   │       ├── finviz_loader.py   # Finviz 数据加载
│   │       ├── hk_loader.py       # 港股列表加载
│   │       ├── us_loader.py       # 美股列表加载
│   │       └── yahoo_loader.py    # Yahoo 数据加载
│   ├── risk/               # 风险管理
│   │   └── position_sizer.py  # 动态仓位管理
│   ├── strategies/         # 策略模块
│   │   ├── accumulation_acceleration_strategy.py  # 主力吸筹加速策略
│   │   ├── market_regime_strategy.py     # 市场环境识别策略
│   │   ├── momentum_breakout_strategy.py # 动量突破策略
│   │   ├── signal_scorer.py              # 信号评分器
│   │   ├── strategy_config.py            # 策略配置管理
│   │   └── volatility_squeeze_strategy.py # 波动率压缩策略
│   └── utils/              # 工具模块
│       ├── exceptions.py
│       └── logger.py
├── data_cache/             # 数据缓存目录
│   ├── ai_analysis/        # AI分析结果缓存
│   ├── HK/                 # 港股数据缓存
│   └── US/                 # 美股数据缓存
└── logs/                   # 日志文件目录
```

## 配置参数

系统支持多种配置选项，通过 `config.json` 文件进行设置：

```json
{
  "speed_mode": "balanced",     // 速度模式: fast/balanced/safe
  "api": {
    "base_delay": 0.4,          // API 调用基础延迟
    "max_delay": 1.0,           // API 调用最大延迟
    "min_delay": 0.4,           // API 调用最小延迟
    "retry_attempts": 3,        // 重试次数
    "max_workers": 2            // 最大并行工作线程数
  },
  "data": {
    "max_cache_days": 7,        // 缓存数据最大天数
    "float_dtype": "float32",
    "enable_cache": false,
    "enable_finviz": true       // 启用 Finviz 数据
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
    "ma_periods": [5, 10, 20, 50, 200],
    "cmo_period": 14,              // CMO 指标
    "williams_r_period": 14,       // Williams %R
    "stochastic_period": 14,       // 随机指标
    "volume_z_score_period": 20    // 成交量 Z-Score
  },
  "strategies": {
    "momentum_breakout": { ... },
    "accumulation_acceleration": { ... },
    "volatility_squeeze": { ... },
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
      "trend_strength_threshold": 0.3
    }
  },
  "ai": {
    "api_timeout": 30,
    "model": "deepseek-v3.2",
    "max_data_points": 100
  }
}
```

### 速度模式预设

| 模式 | 并行数 | 延迟 | 适用场景 |
|------|--------|------|----------|
| `fast` | 8 | 0.2s | 快速测试 |
| `balanced` | 4 | 0.5s | 日常使用（推荐） |
| `safe` | 2 | 1.0s | 避免 API 限流 |

## 主要功能模块

### 1. 策略系统

策略系统采用动态加载机制，自动从 `strategies/` 目录加载所有继承自 `BaseStrategy` 的类。

#### 现有策略

| 策略名称 | 文件 | 说明 |
|---------|------|------|
| **MomentumBreakoutStrategy** | momentum_breakout_strategy.py | 动量突破策略，检测价格突破、量能爆发、动量强度 |
| **VolatilitySqueezeStrategy** | volatility_squeeze_strategy.py | 波动率压缩策略，检测布林带压缩、突破确认、量能配合 |
| **AccumulationAccelerationStrategy** | accumulation_acceleration_strategy.py | 主力吸筹加速策略，检测横盘幅度、量能趋势、突破信号 |
| **SignalScorer** | signal_scorer.py | 多维度信号评分器，综合趋势/动量/量能/市场环境评分 |
| **MarketRegimeStrategy** | market_regime_strategy.py | 市场环境识别，判断趋势/震荡/高波动市场 |

#### 策略基类使用示例
```python
from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult

class MyNewStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "我的新策略"

    @property
    def category(self) -> str:
        return "趋势跟踪"

    def execute(self, context: StrategyContext) -> StrategyResult:
        hist = context.hist
        info = context.info
        is_market_healthy = context.is_market_healthy
        
        # 实现策略逻辑
        passed = self._check_conditions(hist)
        
        return StrategyResult(
            passed=passed,
            confidence=0.8,
            details={"reason": "条件满足"}
        )
```

### 2. 信号评分器 (SignalScorer)

多维度信号综合评分系统：

| 维度 | 权重 | 说明 |
|------|------|------|
| 趋势跟踪 | 25% | 价格与均线位置、均线排列、趋势持续性 |
| 动量突破 | 20% | 价格突破、RSI、MACD 金叉 |
| 量能确认 | 15% | 成交量突破、量价配合、成交量趋势 |
| 市场回调 | 20% | 大盘健康状态、相对表现、市场环境 |
| 行业强度 | 20% | 行业信息、技术形态、布林带突破 |

```python
from src.strategies.signal_scorer import quick_score

# 快速评分
result = quick_score(hist, info, market_healthy=True)
# {'passed': True, 'score': 0.75, 'scores': {...}, 'breakdown': {...}}
```

### 3. 市场环境识别 (MarketRegimeStrategy)

自动识别市场类型：
- **trending**: 趋势市场，适合动量策略
- **mean_reverting**: 震荡市场，适合均值回归策略  
- **volatile**: 高波动市场，需谨慎操作

```python
from src.strategies.market_regime_strategy import get_market_regime

regime = get_market_regime(hist)
# {'regime': 'trending', 'is_healthy': True, 'health_score': 0.72, ...}
```

### 4. 回测引擎

完整的策略回测功能：

```python
from src.backtest.engine import BacktestEngine, BacktestConfig, backtest, print_result

# 快速回测
def my_strategy(data):
    # 返回 1=买入, -1=卖出, 0=持有
    if data['Close'].iloc[-1] > data['Close'].iloc[-20:].mean():
        return 1
    return 0

result = backtest(data, my_strategy, initial_capital=100000)
print_result(result)

# 高级配置
config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,        # 0.1% 手续费
    slippage=0.0005,         # 0.05% 滑点
    stop_loss=0.08,          # 8% 止损
    take_profit=0.15,        # 15% 止盈
    position_sizing="volatility"
)
engine = BacktestEngine(config)
result = engine.run(data, my_strategy)

# 蒙特卡洛测试
mc_result = engine.monte_carlo_test(result, n_runs=1000)
```

### 5. 仓位管理

动态仓位计算，支持多种方法：

```python
from src.risk.position_sizer import PositionSizer, calc_position, kelly_sizer

# 基于信心度计算仓位
position = calc_position(
    confidence=0.75,
    price=100.0,
    volatility=0.25,
    account_balance=100000
)
# {'position_size': 15000, 'shares': 150, 'position_pct': 0.15}

# Kelly 公式
kelly_pct, analysis = kelly_sizer(returns_series)
print(f"建议仓位: {kelly_pct:.1%}")
print(analysis['risk_warning'])

# 风险平价
sizer = PositionSizer()
weights = sizer.volatility_parity_position({
    'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.30
})
```

### 6. HTML/PDF 报告生成

现代化报告格式：

```python
from src.core.services.report_writer import ReportWriter

# 创建报告写入器
writer = ReportWriter(
    filename="hk_stocks_2026-02-18",
    market="HK",
    output_format='both'  # 'txt', 'html', 'both'
)

writer.initialize()
writer.write_stock_result(result)
writer.write_summary(results, 'HK')

# 输出文件:
# - hk_stocks_2026-02-18.txt  (纯文本)
# - hk_stocks_2026-02-18.html (HTML报告)
# - hk_stocks_2026-02-18.pdf  (可选，需安装依赖)
```

### 7. 数据缓存系统

- **按市场分离**：US 和 HK 市场数据分别存储
- **按时间框架分离**：不同时间框架（1d/1h/1m）使用不同缓存文件
- **版本控制**：`version.txt` 记录最后同步日期
- **增量更新**：只下载缺失或过期的数据

### 8. 多模型 AI 分析

支持的 AI 模型：
- `iflow-rome-30ba3b`
- `qwen3-max`
- `tstars2.0`
- `deepseek-v3.2` (默认)
- `qwen3-coder-plus`
- `all` (运行所有模型并合并结果)

## 使用方法

### 基本命令
```bash
# 筛选美股
python3 main.py --market US

# 筛选港股
python3 main.py --market HK

# 快速模式（跳过缓存更新）
python3 main.py --market HK --no-cache-update

# 分析指定股票
python3 main.py --market HK --symbol 0017.HK

# 使用小时线数据
python3 main.py --market HK --interval 1h

# 跳过策略筛选，对所有股票进行AI分析
python3 main.py --market HK --skip-strategies

# 使用特定AI模型
python3 main.py --market HK --model qwen3-max

# 使用所有AI模型分析
python3 main.py --market HK --model all

# 使用快速模式运行
python3 main.py --market HK --speed fast

# 使用安全模式运行（避免API限流）
python3 main.py --market HK --speed safe
```

### 参数说明
| 参数 | 说明 |
|------|------|
| `--market` | 必需，市场代码 (US/HK) |
| `--no-cache-update` | 跳过缓存更新 |
| `--skip-strategies` | 跳过策略筛选 |
| `--symbol` | 指定单一股票代码 |
| `--interval` | 数据时段 (1d/1h/1m) |
| `--model` | AI模型选择 |
| `--speed` | 速度模式 (fast/balanced/safe) |

## 安装与部署

### 环境要求
- Python 3.8+
- 系统兼容性：Windows, macOS, Linux

### 安装步骤
```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate  # macOS/Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装项目
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

# 可选：PDF 报告生成
# weasyprint>=60.0
# pdfkit>=1.0.0
```

### 启用 PDF 生成
```bash
# 方式一：weasyprint (推荐)
pip install weasyprint

# 方式二：pdfkit
brew install wkhtmltopdf
pip install pdfkit
```

## 开发指南

### 添加新策略
1. 在 `src/strategies/` 目录创建新文件
2. 继承 `BaseStrategy` 类
3. 实现 `name`、`category` 属性和 `execute` 方法
4. 系统自动检测并加载

### 策略配置管理
使用 `strategy_config.py` 中的配置类管理策略参数：

```python
from src.strategies.strategy_config import strategy_config_manager

# 获取策略配置
config = strategy_config_manager.get_config('signal_scorer')
print(config.weights)
print(config.pass_threshold)
```

### 数据处理流程
1. **数据获取**：从 Yahoo Finance / Finviz 获取数据
2. **缓存管理**：增量同步或快速加载
3. **策略执行**：运行所有策略筛选
4. **AI 分析**：对符合条件的股票进行 AI 分析
5. **报告生成**：输出 TXT/HTML/PDF 报告

### 性能优化
- 预计算技术指标避免重复计算
- AI 分析结果缓存
- 配置缓存避免重复加载
- 并行处理股票分析
- 内存优化 (float32)
- 速度模式预设，一键调整性能参数

## 输出格式

### 文件输出
```
hk_stocks_2026-02-18.txt   # 纯文本摘要
hk_stocks_2026-02-18.html  # HTML 详细报告
hk_stocks_2026-02-18.pdf   # PDF (需安装依赖)
```

### HTML 报告特性
- 统计卡片：筛选数量、策略数、耗时
- 策略标签：各策略命中统计
- 股票卡片：名称、代码、行业、市值、PE
- 评分显示：技术面/基本面/综合评分
- AI 分析：完整分析结果展示
- 打印友好：浏览器 Cmd+P 导出 PDF

## 故障排除

### 常见问题
1. **API 限制**：使用 `--speed safe` 或手动调整 `base_delay`
2. **数据缺失**：删除缓存重新下载
3. **AI分析失败**：检查 `IFLOW_API_KEY` 环境变量
4. **内存不足**：使用 `--speed safe` 减少并行数

### 调试方法
- 使用 `--symbol` 分析特定股票
- 查看 `logs/` 目录下的日志文件
- 使用 `--skip-strategies` 测试 AI 分析流程
- 使用 `--speed fast` 快速测试

## 回测指标说明

### 核心指标
| 指标 | 说明 |
|------|------|
| `total_return` | 总收益率 |
| `sharpe_ratio` | 夏普比率 |
| `max_drawdown` | 最大回撤 |
| `win_rate` | 胜率 |
| `profit_factor` | 盈亏比 |

### 蒙特卡洛测试
通过打乱收益率序列评估策略稳健性，输出置信区间：
- `prob_positive`: 正收益概率
- `prob_sharpe_gt_1`: 夏普比率 > 1 的概率
- `prob_max_dd_lt_10`: 最大回撤 < 10% 的概率