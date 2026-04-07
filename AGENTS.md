# 股票筛选与分析系统 - Agent 指南

**Generated:** 2026-04-05 | **Commit:** 9f0acb5 | **Branch:** master

## 项目概述

Python 股票筛选与分析系统，支持美股（US）和港股（HK），集成多策略筛选、多 AI 提供商分析、回测引擎、仓位管理。

### 核心特性
- **多市场**: 美股 + 港股
- **策略模块化**: `src/strategies/` 动态加载，继承 BaseStrategy
- **多 AI 提供商**: iFlow / NVIDIA NIM / Google Gemini
- **多时间框架**: 日线(1d) / 小时线(1h) / 分钟线(1m)
- **数据库**: PostgreSQL + SQLAlchemy 2.0 + FastAPI
- **报告**: HTML/TXT/JSON，支持新闻整合

---

## 架构

```
yfinance/
├── main.py                 # CLI 入口点
├── config.json             # 运行时配置
├── .env                    # API 密钥等敏感信息
├── src/
│   ├── ai/analyzer/        # 多 AI 提供商分析 (iFlow/NVIDIA/Gemini)
│   ├── strategies/         # 策略实现 (obv_boll, strategy_config)
│   ├── core/
│   │   ├── strategies/     # 策略基类 + 动态加载引擎
│   │   ├── services/       # 核心业务服务 (分析/报告/图表)
│   │   └── models/         # 数据类 (StrategyResult, StockAnalysisResult)
│   ├── data/
│   │   ├── loaders/        # 数据获取层 (Yahoo/Finviz/HK/US)
│   │   ├── cache/          # 缓存服务 (OptimizedCache)
│   │   └── external/       # 外部数据源 (宏观指标)
│   ├── backtest/           # 回测引擎 + 指标计算
│   ├── risk/               # 仓位管理 + VaR
│   ├── db/                 # PostgreSQL ORM (Stock, News, AIAnalysis)
│   ├── repositories/       # 数据访问层 (StockRepo, NewsRepo, AIAnalysisRepo)
│   ├── api/                # FastAPI REST API
│   ├── config/             # 配置管理 (constants → settings → validator)
│   └── utils/              # 通用工具 (异常/日志)
├── data_cache/             # 运行时缓存 (US/, HK/)
├── reports/                # 输出报告
└── logs/                   # 运行日志
```

---

## WHERE TO LOOK

| 任务 | 位置 | 说明 |
|------|------|------|
| 添加新策略 | `src/strategies/` | 继承 `BaseStrategy`，自动加载 |
| 修改 AI 分析 | `src/ai/analyzer/` | 参见各提供商 AGENTS.md |
| 修改配置默认值 | `src/config/constants.py` | 单一来源 |
| 修改报告格式 | `src/core/services/report_writer.py` | HTML/TXT 生成 |
| 添加新数据源 | `src/data/loaders/` | 继承 YahooFinanceRepository 模式 |
| 数据库修改 | `src/db/models/` | SQLAlchemy 2.0 Async |
| API 端点 | `src/api/routes/` | FastAPI 路由 |
| AI 分析持久化 | `src/core/services/ai_analysis_persister.py` | 分析结果自动存储 |
| AI 分析 API | `src/api/routes/ai_analyses.py` | RESTful 接口 |
| 回测策略 | `src/backtest/engine.py` | BacktestEngine |

---

## CONVENTIONS

### 配置管理
- 所有默认值在 `src/config/constants.py`，不要在其他模块硬编码
- 使用 `config_manager` 单例访问配置
- 速度模式: `fast`(8 workers, 0.2s) / `balanced`(4, 0.5s) / `safe`(2, 1.0s)

### 策略开发
- 继承 `BaseStrategy`，实现 `execute(context: StrategyContext)` 方法
- 返回 `StrategyResult(passed, confidence, details)`
- 策略文件放在 `src/strategies/`，自动被 `pkgutil` 加载

### AI 分析器
- 所有分析器继承 `IFlowAIAnalyzer`
- 实现 `_call_api()` 方法，复用 `_step_by_step_analysis()` 流程
- 使用 `AIAnalysisService` 门面，不要直接实例化分析器

### 数据获取
- 先检查缓存，再请求外部 API
- 使用 `OptimizedCache` 进行持久化
- 技术指标作为 DataFrame 列添加: `RSI_14`, `MACD`, `BB_Upper`, `ATR_14`, `MA_*`

### 报告生成
- 使用 `ReportWriter.write_stock_result()` 写入，不要直接写文件
- 线程安全: `threading.RLock()`
- 原子写入: 先写临时文件，再 `os.rename()`

### AI 分析持久化
- 分析完成后自动通过 `AIAnalysisPersister` 写入数据库
- 每个 provider 的分析结果作为独立记录存储
- 自动清理旧记录（默认保留最新 30 条）
- Stock 不存在时自动创建
- 使用 `asyncio.new_event_loop()` 桥接同步→异步调用

---

## ANTI-PATTERNS (FORBIDDEN)

### 通用
- **Never** 在模块中硬编码配置值 - 从 `src.config.constants` 导入
- **Never** 跳过缓存检查 - 先调用 `cache_service.get()`
- **Never** 使用 `as any` / `@ts-ignore` / `@ts-expect-error` 抑制类型错误
- **Never** 提交 `.env` 文件 - 它是 gitignored 的

### 数据层
- **Never** 修改输入 DataFrame - 先 `.copy()`
- **Never** 重复计算指标 - 检查 `if 'RSI_14' not in df.columns`
- **Never** 分钟线数据获取全部历史 - 使用 `period='7d'`

### AI 分析
- **Never** 直接实例化分析器 - 使用 `AIAnalysisService`
- **Never** 硬编码 API 端点 - 使用 `config_manager.get_config().ai`
- **Never** 绕过 `_rate_limit_lock` 调用 API

### 报告
- **Never** 直接写入报告文件 - 使用 `ReportWriter.write_stock_result()`
- **Never** 修改 `_results` 而不持有 `self._lock`

### AI 分析持久化
- **Never** 在 StockAnalyzer 外直接调用 Persister - 使用 `AIAnalysisPersister`
- **Never** 绕过 auto_save 配置强制写入 - 通过配置控制
- **Never** 手动管理 analyzed_at 时间戳 - 使用 `datetime.now()`

---

## COMMANDS

```bash
# 筛选港股
python3 main.py --market HK

# 美股 + 快速模式
python3 main.py --market US --speed fast

# 多 AI 提供商分析
python3 main.py --market HK --provider iflow,nvidia,gemini

# 分析单只股票
python3 main.py --market HK --symbol 0001.HK --interval 1h

# 合并报告
python3 merge_stocks.py --verbose

# 数据库 API
uvicorn src.api.main:app --reload

# 查询 AI 分析记录
curl http://localhost:8000/api/v1/ai-analyses/0001.HK/latest

# 查询 AI 分析记录
curl http://localhost:8000/api/v1/ai-analyses/0001.HK/latest
```

---

## NOTES

### 已知问题
- `setup.py` 中项目名拼写为 "yfinace"（应为 yfinance）
- `pyproject.toml` 和 `setup.py` 配置重复（建议只保留 pyproject.toml）
- 无测试文件（pytest 已声明为 dev 依赖但未使用）
- 无 GitHub Actions CI/CD

### 测试状态
- ❌ 无测试文件 (test_*.py, *_test.py)
- ❌ 无 pytest.ini / conftest.py
- ❌ 无 coverage 配置
- ✅ pytest 已在 pyproject.toml dev 依赖中声明

### 项目规模
- 文件: 130 个
- Python 代码: ~19,800 行
- 大文件 (>500 LOC): 13 个
- 最大深度: 5 层
