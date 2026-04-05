# AI Analysis Module

**Generated:** 2026-04-05 | **Commit:** 9f0acb5 | **Branch:** master

## Overview
多 AI 提供商分析系统，支持 iFlow、NVIDIA NIM、Google Gemini、OpenCode，提供股票技术分析、预测跟踪和多模型共识投票。

## Key Files

**iflow_analyzer.py** (1803 LOC) - 基础分析器，定义模板方法: `_step_by_step_analysis()`, `_build_analysis_prompt()`, `_call_api()`。PredictionTracker 缓存分析结果。

**nvidia_analyzer.py** (1210 LOC) - NVIDIA NIM API 实现，支持流式响应和 reasoning_content 提取。

**gemini_analyzer.py** (1030 LOC) - Google Gemini API 集成，支持 model='all' 多模型共识。

**opencode_analyzer.py** (549 LOC) - OpenCode AI 集成。

**service.py** (~100 LOC) - AIAnalysisService 门面，提供提供商路由、模型发现、批量分析编排。

## Conventions

- 所有分析器继承 `IFlowAIAnalyzer`，复用 `_step_by_step_analysis()` 流程
- `PredictionTracker` 通过 `data_cache/prediction_tracker.json` 缓存分析结果
- 多模型共识通过 `model='all'` 参数触发
- 异步流式通过 `enable_streaming=True` 支持 (NVIDIA/Gemini)
- 所有分析器使用 `OptimizedCache` 进行 AI 结果缓存 (TTL-based)
- 速率限制通过 `_last_api_call_time` 类变量和 `MIN_API_INTERVAL`
- API 端点在 `config.json` 中配置，永不硬编码
- 所有提示词通过 `_format_news()` 方法格式化新闻

## Anti-Patterns

- **Never** 直接实例化分析器 - 始终使用 `AIAnalysisService`
- **Never** 修改 `PredictionTracker.predictions` - 使用 `track()` 方法
- **Never** 硬编码 API 端点 - 使用 `config_manager.get_config().ai`
- **Never** 绕过 `_rate_limit_lock` 调用 API
- **Never** 修改 `IFlowAIAnalyzer` 中的 `FEW_SHOT_EXAMPLES`
- **Never** 直接调用 `analyzer.analyze()` - 使用 `service.analyze_stock()`

## Dependencies

- `entities.py`: AIAnalysisResult 数据类，StockData 模型
- `cache_service.py`: OptimizedCache 用于 AI 结果持久化
- `config_manager`: API 密钥、模型列表、速率限制
- `constants.py`: VIX 阈值、默认超时

## Architecture Notes

分析器层次结构形成模板模式 - `IFlowAIAnalyzer` 定义 `_step_by_step_analysis()`, `_build_analysis_prompt()`, `_call_api()` 钩子。子类仅覆盖 `_call_api()` 以实现提供商特定逻辑，同时复用分析管道。

`PredictionTracker` 在 `data_cache/prediction_tracker.json` 中维护预测历史，14 天后自动验证。通过 `get_model_accuracy()` 计算每个模型的准确性指标。
