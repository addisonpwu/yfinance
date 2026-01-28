# 股票筛选与分析系统 - Agent 指南

## 项目概述

这是一个基于 Python 的股票筛选与分析系统，采用模块化设计，旨在根据多种可扩展的、经过专业交易逻辑强化的策略，从美国和香港股市中找出符合特定条件的股票。

### 核心特性
- **多市场支持**：支持美股（US）和港股（HK）市场
- **策略模块化**：动态加载策略，易于扩展
- **智能缓存系统**：高效的数据缓存机制，提升重复运行效率
- **AI 分析集成**：集成 iFlow API 进行技术分析
- **AI 预测模型**：整合 Kronos Transformer 模型进行股价预测
- **新闻情感分析**：自动抓取并分析股票新闻情感
- **多时间框架**：支持日线、小时线、分钟线数据
- **配置缓存机制**：避免重复加载配置文件
- **智能API延迟策略**：根据请求结果动态调整延迟时间
- **AI分析结果缓存**：避免重复的API调用
- **预计算技术指标**：避免重复计算技术指标
- **改进的错误处理和日志记录**：提供详细的运行日志

## 项目架构

```
yfinace/
├── main.py                 # 主入口点
├── config.json             # 配置文件
├── ai_analyzer.py          # AI 分析模块
├── news_analyzer.py        # 新闻分析模块
├── analysis/               # 分析引擎
│   └── analyzer.py
├── data_loader/            # 数据加载器
│   ├── us_loader.py        # 美股加载器
│   └── hk_loader.py        # 港股加载器
├── strategies/             # 策略模块
│   ├── base_strategy.py    # 策略基类
│   └── [具体策略文件]
├── Kronos/                 # AI 预测模型
├── data_cache/             # 数据缓存目录
├── logs/                   # 日志文件目录
└── requirements.txt        # 依赖文件
```

## 配置参数

系统支持多种配置选项，通过 `config.json` 文件进行设置：

```json
{
  "api": {
    "base_delay": 0.5,      // API 调用基础延迟
    "max_delay": 0.9,       // API 调用最大延迟
    "min_delay": 0.5,       // API 调用最小延迟
    "retry_attempts": 3,    // 重试次数
    "max_workers": 2        // 最大并行工作线程数
  },
  "data": {
    "max_cache_days": 7,    // 缓存数据最大天数
    "float_dtype": "float32" // 浮点数数据类型
  },
  "analysis": {
    "enable_realtime_output": true,    // 启用实时输出
    "enable_data_preprocessing": true, // 启用数据预处理
    "min_volume_threshold": 100000     // 最小成交量阈值
  }
}
```

## 主要功能模块

### 1. 策略系统

策略系统采用动态加载机制，自动从 `strategies/` 目录加载所有继承自 `BaseStrategy` 的类。

#### 现有策略
- **VCP_PocketPivotStrategy**: VCP口袋支点策略，寻找"口袋支点"作为早期介入信号
- **MainForceAccumulationStrategy**: 主力吸筹策略，捕捉机构投资者吸筹阶段
- **GoldenCrossStrategy**: 黄金交叉策略，经典长线趋势反转策略
- **MeanReversionStrategy**: 均值回归策略，捕捉超跌反弹机会
- **InsideDayStrategy**: 内部日反转策略，捕捉下降趋势中的转折点
- **BollingerSqueezeStrategy**: 布林带挤压突破策略，捕捉波动率压缩后的突破
- **TurnoverMomentumBreakoutStrategy**: 换手率动量突破策略，寻找换手率激增的股票

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

### 4. Kronos AI 预测系统

为港股提供 AI 预测模型：
- **Transformer 架构**：基于自回归模型
- **全球支持**：支持全球 45+ 交易所数据
- **概率预测**：提供未来 10 天的上升/下跌概率

## 使用方法

### 基本命令
```bash
# 筛选美股
python3 main.py --market US

# 筛选港股
python3 main.py --market HK

# 强制快速模式（跳过缓存更新）
python3 main.py --market HK --no-cache-update

# 跳过 Kronos 预测（仅适用于港股）
python3 main.py --market HK --no-kronos

# 分析指定单一股票
python3 main.py --market HK --symbol 0017.HK

# 使用小时线数据进行分析
python3 main.py --market HK --interval 1h

# 使用分钟线数据进行分析
python3 main.py --market HK --interval 1m
```

### 参数说明
- `--market`: 必需参数，指定要分析的市场 (`US` 或 `HK`)
- `--no-cache-update`: 可选参数，跳过缓存更新，直接使用现有缓存数据
- `--no-kronos`: 可选参数，跳过 Kronos 预测（仅适用于港股）
- `--symbol`: 可选参数，指定分析单一股票代码
- `--interval`: 可选参数，指定数据时段类型（`1d`/`1h`/`1m`）

## 安装与部署

### 环境要求
- Python 3.12+
- PyTorch (根据系统架构安装)

### 安装步骤
```bash
# 1. 创建虚拟环境
python3.12 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate  # macOS/Linux

# 3. 安装 PyTorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# 4. 安装其他依赖
pip install -r requirements.txt
pip install -r Kronos/requirements.txt
```

### 依赖项
```
yfinance
pandas
requests
lxml
openpyxl
playwright
```

## 开发指南

### 添加新策略
1. 在 `strategies/` 目录中创建新的 Python 文件
2. 创建继承自 `BaseStrategy` 的类
3. 实现 `name` 属性和 `run` 方法
4. 系统将自动检测并加载新策略

### 数据处理流程
1. **数据获取**：从 Yahoo Finance 获取股票数据
2. **缓存管理**：根据模式选择快速加载或增量同步
3. **策略执行**：对每只股票运行所有策略
4. **AI 分析**：对符合条件的股票进行 AI 技术分析
5. **Kronos 预测**：对港股进行 AI 预测（如启用）
6. **结果筛选**：根据预测结果筛选最终股票列表

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

## 输出格式

### 详细报告
生成 `..._details.txt` 文件，包含：
- 股票基本信息（名称、代码、行业等）
- 符合的策略列表
- Kronos 预测结果
- AI 分析摘要
- 基本面数据（市值、PE等）

### 摘要列表
生成 `..._details.txt` 文件，包含：
- 格式化的股票代码列表
- 便于复制到其他软件使用

## 故障排除

### 常见问题
1. **API 限制**：增加 API 调用延迟或减少并行线程数
2. **数据缺失**：检查缓存文件完整性，必要时删除缓存重新下载
3. **内存不足**：减少并行线程数或使用快速模式
4. **Kronos 预测失败**：检查 Kronos 环境配置和模型路径
5. **配置验证错误**：检查 config.json 中的配置值是否在合理范围内

### 调试方法
- 使用 `--symbol` 参数分析特定股票进行调试
- 检查 `config.json` 中的配置参数
- 查看缓存文件是否正确生成
- 监控系统资源使用情况
- 查看日志文件 (位于 logs/ 目录下) 以获取详细运行信息