# 股票筛选与分析系统 - Agent 指南

## 项目概述

这是一个基于 Python 的股票筛选与分析系统，采用模块化设计，旨在根据多种可扩展的、经过专业交易逻辑强化的策略，从美国和香港股市中找出符合特定条件的股票。

### 核心特性
- **多市场支持**：支持美股（US）和港股（HK）市场
- **策略模块化**：动态加载策略，易于扩展
- **智能缓存系统**：高效的数据缓存机制，提升重复运行效率
- **AI 分析集成**：集成 iFlow API 进行技术分析
- **新闻情感分析**：自动抓取并分析股票新闻情感
- **多时间框架**：支持日线、小时线、分钟线数据
- **配置缓存机制**：避免重复加载配置文件
- **智能API延迟策略**：根据请求结果动态调整延迟时间
- **AI分析结果缓存**：避免重复的API调用
- **预计算技术指标**：避免重复计算技术指标
- **改进的错误处理和日志记录**：提供详细的运行日志
- **跳过策略功能**：支持跳过策略筛选，对所有股票进行AI分析
- **多模型AI分析**：支持多种AI模型进行股票分析，包括`iflow-rome-30ba3b`、`qwen3-max`、`tstars2.0`、`deepseek-v3.2`、`qwen3-coder-plus`及`all`选项
- **高级技术指标**：支持RSI、MACD、布林带、ATR等技术指标分析
- **数据预处理优化**：包含成交量阈值、数据点数量等预处理优化
- **实时输出**：支持符合条件股票的实时输出到文件

## 项目架构

```
yfinace/
├── main.py                 # 主入口点
├── config.json             # 配置文件
├── pyproject.toml          # 项目配置
├── setup.py                # 安装配置
├── README.md               # 项目说明
├── requirements.txt        # 依赖文件
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── ai/                 # AI分析模块
│   │   ├── __init__.py
│   │   ├── analyzer/       # AI分析器
│   │   │   ├── __init__.py
│   │   │   ├── ai_analyzer.py
│   │   │   ├── iflow_analyzer.py
│   │   │   ├── service.py
│   │   │   └── __pycache__/
│   │   └── models/
│   │       └── __init__.py
│   ├── analysis/           # 分析引擎
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   └── __pycache__/
│   ├── config/             # 配置管理
│   │   ├── __init__.py
│   │   ├── settings.py     # 配置类定义
│   │   └── __pycache__/
│   ├── core/               # 核心模块
│   │   ├── __init__.py
│   │   ├── models/         # 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── entities.py
│   │   │   └── __pycache__/
│   │   ├── services/       # 服务层
│   │   │   ├── __init__.py
│   │   │   ├── analysis_service.py
│   │   │   └── __pycache__/
│   │   └── strategies/     # 策略核心
│   │       ├── __init__.py
│   │       ├── loader.py   # 策略加载器
│   │       ├── strategy.py # 策略执行器
│   │       └── __pycache__/
│   ├── data/               # 数据处理
│   │   ├── __init__.py
│   │   ├── cache/          # 缓存服务
│   │   │   ├── __init__.py
│   │   │   ├── cache_service.py
│   │   │   └── __pycache__/
│   │   ├── external/       # 外部数据源
│   │   │   ├── __init__.py
│   │   │   ├── stock_repository.py
│   │   │   └── __pycache__/
│   │   └── loaders/        # 数据加载器
│   │       ├── __init__.py
│   │       ├── yahoo_loader.py
│   │       └── __pycache__/
│   ├── data_loader/        # 股票列表加载器
│   │   ├── __init__.py
│   │   ├── get_yahoo_news.py
│   │   ├── hk_loader.py    # 港股加载器
│   │   ├── us_loader.py    # 美股加载器
│   │   └── __pycache__/
│   ├── strategies/         # 策略模块
│   │   ├── __init__.py
│   │   ├── accumulation_strategy.py    # 主力吸筹策略
│   │   ├── base_strategy.py            # 策略基类
│   │   ├── bollinger_squeeze_strategy.py  # 布林带挤压突破策略
│   │   ├── golden_cross_strategy.py    # 黄金交叉策略
│   │   ├── inside_day_strategy.py      # 内部日反转策略
│   │   ├── mean_reversion_strategy.py  # 均值回归策略
│   │   ├── turnover_momentum_breakout_strategy.py  # 换手率动量突破策略
│   │   ├── vcp_pocket_pivot_strategy.py  # VCP口袋支点策略
│   │   ├── volume_anomaly_strategy.py  # 成交量异常放大策略
│   │   └── __pycache__/
│   ├── utils/              # 工具模块
│   │   ├── __init__.py
│   │   ├── exceptions.py
│   │   ├── logger.py
│   │   └── __pycache__/
│   └── news_analyzer.py    # 新闻分析模块
├── data_cache/             # 数据缓存目录
│   ├── ai_analysis/        # AI分析结果缓存
│   ├── HK/                 # 港股数据缓存
│   └── US/                 # 美股数据缓存
├── logs/                   # 日志文件目录
└── venv/                   # 虚拟环境
```

## 配置参数

系统支持多种配置选项，通过 `config.json` 文件进行设置：

```json
{
  "api": {
    "base_delay": 0.5,      // API 调用基础延迟
    "max_delay": 2.0,       // API 调用最大延迟
    "min_delay": 0.1,       // API 调用最小延迟
    "retry_attempts": 3,    // 重试次数
    "max_workers": 4        // 最大并行工作线程数
  },
  "data": {
    "max_cache_days": 7,    // 缓存数据最大天数
    "float_dtype": "float32", // 浮点数数据类型
    "data_download_period": { // 不同时间框架的数据下载周期
      "1m": "7d",           // 1分钟线：最近7天
      "1h": "730d",         // 1小时线：最近2年
      "1d": "max"           // 1日线：全部历史
    },
    "enable_cache": false   // 是否启用缓存
  },
  "analysis": {
    "enable_realtime_output": true,    // 启用实时输出
    "enable_data_preprocessing": true, // 启用数据预处理
    "min_volume_threshold": 100000,    // 最小成交量阈值
    "min_data_points_threshold": 20    // 最小数据点阈值
  },
  "technical_indicators": {
    "rsi_period": 14,       // RSI周期
    "macd_fast": 12,        // MACD快线周期
    "macd_slow": 26,        // MACD慢线周期
    "macd_signal": 9,       // MACD信号线周期
    "bb_period": 20,        // 布林带周期
    "bb_std_dev": 2,        // 布林带标准差倍数
    "atr_period": 14,       // ATR周期
    "ma_periods": [5, 10, 20, 50, 200] // 移动平均线周期
  },
  "strategies": {
    "vcp_pocket_pivot": {
      "ma_periods": [50, 150, 200],      // VCP策略移动平均线周期
      "volatility_windows": [50, 20, 10], // VCP策略波动率窗口
      "volume_avg_period": 50,            // VCP策略成交量平均周期
      "pp_lookback_period": 10,           // VCP策略口袋支点回看周期
      "pp_max_bias_ratio": 0.08           // VCP策略最大偏差比例
    },
    "bollinger_squeeze": {
      "bb_period": 20,                    // 布林带周期
      "squeeze_lookback": 100,            // 挤压回看周期
      "squeeze_percentile": 0.10,         // 挤压百分位
      "prolonged_squeeze_period": 5,      // 延长挤压周期
      "long_trend_period": 200,           // 长期趋势周期
      "ma_slope_period": 5,               // 移动平均斜率周期
      "volume_period": 50                 // 成交量周期
    }
  },
  "news": {
    "timeout": 60000,       // 新闻获取超时时间(毫秒)
    "max_news_items": 5     // 最大新闻条数
  },
  "ai": {
    "api_timeout": 30,      // AI API超时时间
    "model": "deepseek-v3.2", // AI模型
    "max_data_points": 100  // AI分析最大数据点数
  }
}
```

## 主要功能模块

### 1. 策略系统

策略系统采用动态加载机制，自动从 `strategies/` 目录加载所有继承自 `BaseStrategy` 的类。

#### 现有策略
- **MomentumBreakoutStrategy**: 動量爆發策略，僅實現價格突破、量能爆發、動量強度邏輯 (momentum_breakout_strategy.py)
- **VolatilitySqueezeStrategy**: 波動率壓縮策略，僅實現布林帶壓縮、突破確認、量能配合邏輯 (volatility_squeeze_strategy.py)
- **AccumulationAccelerationStrategy**: 主力吸籌加速策略，僅實現橫盤幅度、量能趨勢、突破信號、RSI動態上穿邏輯 (accumulation_acceleration_strategy.py)

#### 策略基类 (BaseStrategy)
```python
from .base_strategy import BaseStrategy

class MyNewStrategy(BaseStrategy):
    @property
    def name(self):
        return "我的新策略"

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # 实现策略逻辑
        # hist: 股票历史数据
        # kwargs: 包含 'info' (基本面信息), 'market_return' (市场回报), 'is_market_healthy' (市场健康状况) 等
        pass
```

### 2. 数据缓存系统

智能缓存系统包含以下特性：
- **按市场分离**：US 和 HK 市场数据分别存储
- **按时间框架分离**：不同时间框架（1d/1h/1m）使用不同缓存文件
- **版本控制**：每个市场目录下有 `version.txt` 文件记录最后同步日期
- **增量更新**：只下载缺失或过期的数据
- **自动预处理**：优化数据类型以节省内存
- **配置驱动**：通过配置文件控制缓存行为

### 3. AI 分析系统

集成 iFlow API 提供 8 维度技术分析：
- 趋势分析
- 支撑与阻力
- 动能指标分析
- 成交量分析
- 形态识别
- 短期走势预测
- 风险评估
- 投资建议（含价位建议）

#### AI分析结果缓存机制
- 基于股票数据和历史数据的哈希值生成缓存键
- 按天缓存，缓存有效期为7天
- 缓存路径在 `data_cache/ai_analysis/` 目录下

#### 多模型AI分析系统
- **模型选择**：支持`iflow-rome-30ba3b`、`qwen3-max`、`tstars2.0`、`deepseek-v3.2`、`qwen3-coder-plus`等多种AI模型
- **全模型分析**：使用`--model all`参数可同时运行所有模型并合并分析结果
- **模型特定缓存**：不同模型的分析结果独立缓存，避免混淆
- **智能缓存键**：缓存键包含模型信息，确保模型变更时重新分析

### 4. 跳过策略功能

新增的跳过策略功能允许用户跳过所有选股策略，直接对所有股票进行AI分析：
- **全股票AI分析**：所有股票都绕过策略筛选，直接进入AI分析流程
- **保持数据质量检查**：仍然执行基本的数据质量验证（成交量、价格有效性等）
- **灵活组合使用**：可与其他参数（如 `--no-cache-update`, `--interval`）组合使用

### 5. 多时间框架支持

系统支持多种时间框架的数据分析：
- **日线（1d）**：默认时间框架，下载全部历史数据
- **小时线（1h）**：下载最近2年数据，适用于短期分析
- **分钟线（1m）**：下载最近7天数据，适用于超短期分析
- **独立缓存**：不同时间框架使用独立的缓存文件，互不干扰

### 6. 实时输出功能

系统支持符合条件股票的实时输出：
- **实时文件更新**：在分析过程中实时将符合条件的股票写入文件
- **便于监控**：方便用户在长时间分析过程中监控进度
- **格式化输出**：输出包含股票代码、符合策略和AI分析摘要

## 使用方法

### 基本命令
```bash
# 筛选美股
python3 main.py --market US

# 筛选港股
python3 main.py --market HK

# 强制快速模式（跳过缓存更新）
python3 main.py --market HK --no-cache-update

# 分析指定单一股票
python3 main.py --market HK --symbol 0017.HK

# 使用小时线数据进行分析
python3 main.py --market HK --interval 1h

# 使用分钟线数据进行分析
python3 main.py --market HK --interval 1m

# 跳过策略筛选，所有股票都进行AI分析
python3 main.py --market HK --skip-strategies

# 使用特定AI模型进行分析
python3 main.py --market HK --model qwen3-max

# 使用所有AI模型进行分析
python3 main.py --market HK --model all

# 组合使用：使用小时线数据分析指定股票
python3 main.py --market HK --symbol 0017.HK --interval 1h
```

### 参数说明
- `--market`: 必需参数，指定要分析的市场 (`US` 或 `HK`)
- `--no-cache-update`: 可选参数，跳过缓存更新，直接使用现有缓存数据
- `--skip-strategies`: 可选参数，跳过策略筛选，所有股票都进行AI分析
- `--symbol`: 可选参数，指定分析单一股票代码
- `--interval`: 可选参数，指定数据时段类型（`1d`/`1h`/`1m`）
- `--model`: 可选参数，指定AI分析模型（`iflow-rome-30ba3b`/`qwen3-max`/`tstars2.0`/`deepseek-v3.2`/`qwen3-coder-plus`/`all`）

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
# 或 .\venv\Scripts\activate  # Windows

# 3. 安装 PyTorch (根据系统架构安装)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 或者使用 pyproject.toml 安装
pip install -e .
```

### 依赖项
```
yfinance>=0.2.18
pandas>=1.5.0
requests>=2.28.0
lxml>=4.9.0
openpyxl>=3.0.0
playwright>=1.30.0
numpy>=1.21.0
```

## 开发指南

### 添加新策略
1. 在 `src/strategies/` 目录中创建新的 Python 文件
2. 创建继承自 `BaseStrategy` 的类
3. 实现 `name` 属性和 `execute` 方法
4. 系统将自动检测并加载新策略

### 数据处理流程
1. **数据获取**：从 Yahoo Finance 获取股票数据
2. **缓存管理**：根据模式选择快速加载或增量同步
3. **策略执行**：对每只股票运行所有策略（如果未启用 `--skip-strategies` 参数）
4. **AI 分析**：对符合条件的股票进行 AI 技术分析（或对所有股票进行AI分析，如果启用 `--skip-strategies` 参数）
5. **结果输出**：生成详细报告和摘要列表

### 性能优化
- **配置缓存机制**：避免在多线程环境中重复加载配置文件
- **依赖注入**：通过参数传递配置对象而非内部加载
- **配置验证**：确保配置值在合理范围内
- **智能API延迟策略**：根据请求结果动态调整延迟时间
- **AI分析结果缓存**：避免重复的API调用
- **预计算技术指标**：在数据加载时预计算技术指标，避免重复计算
- **改进错误处理**：更详细异常捕获和处理
- **日志记录**：提供详细的运行日志，便于调试和监控
- **内存优化**：使用 float32 替代 float64，减少内存占用
- **并行处理**：使用 ThreadPoolExecutor 并行分析股票
- **数据预处理**：基础数据质量检查，过滤低质量股票
- **缓存机制**：智能缓存减少重复网络请求
- **跳过策略优化**：支持跳过策略筛选，对所有股票进行AI分析
- **多模型优化**：支持多种AI模型分析，提供更全面的分析结果

### 多模型功能开发
- **参数传递**：从 main.py 的命令行参数到 ai_analyzer.py 的函数调用
- **模型选择逻辑**：在 analyze_stock 函数中实现多模型选择逻辑
- **缓存机制**：为不同模型创建独立的缓存键，避免结果混淆
- **结果合并**：当使用 `all` 模型选项时，合并所有模型的分析结果

## 输出格式

### 详细报告
生成 `..._details.txt` 文件，包含：
- 股票基本信息（名称、代码、行业等）
- 符合的策略列表
- AI 分析摘要
- 基本面数据（市值、PE等）
- 新闻情感分析结果（如果启用）

### 摘要列表
生成 `..._stocks_YYYY-MM-DD.txt` 文件，包含：
- 格式化的股票代码列表
- 便于复制到其他软件使用
- 当启用 `--skip-strategies` 参数时，列表将包含所有经过AI分析的股票

## 故障排除

### 常见问题
1. **API 限制**：增加 API 调用延迟或减少并行线程数
2. **数据缺失**：检查缓存文件完整性，必要时删除缓存重新下载
3. **内存不足**：减少并行线程数或使用快速模式
4. **AI分析失败**：检查 `IFLOW_API_KEY` 环境变量是否设置
5. **配置验证错误**：检查 config.json 中的配置值是否在合理范围内
6. **跳过策略功能异常**：启用 `--skip-strategies` 参数后，确保系统有足够的资源处理所有股票的AI分析
7. **函数返回值错误**：修复了 `analyze_single_stock` 函数中的缩进问题，确保在跳过策略模式下正确返回结果
8. **多模型功能问题**：确保环境变量 `IFLOW_API_KEY` 已设置以使用AI分析功能

### 调试方法
- 使用 `--symbol` 参数分析特定股票进行调试
- 检查 `config.json` 中的配置参数
- 查看缓存文件是否正确生成
- 监控系统资源使用情况
- 查看日志文件 (位于 logs/ 目录下) 以获取详细运行信息
- 使用 `--skip-strategies` 参数测试AI分析流程
- 使用 `--symbol` 和 `--skip-strategies` 组合进行功能验证
- 使用 `--model` 参数测试多模型AI分析功能
- 设置 `--interval` 参数测试不同时间框架的数据分析