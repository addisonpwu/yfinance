# AI Analysis Module - Agent Guide

## Overview
Multi-provider AI analysis system supporting iFlow, NVIDIA NIM, Google Gemini, and OpenCode APIs for stock technical analysis with prediction tracking and multi-model consensus voting.

## Key Files

**iflow_analyzer.py** (1803 LOC) - Base analyzer, PredictionTracker, step-by-step analysis pipeline, PredictionDirection enum, MultiModelConsensus dataclass. All other analyzers inherit patterns from here.

**nvidia_analyzer.py** (1210 LOC) - NVIDIA NIM API implementation. Inherits from IFlowAIAnalyzer. Supports streaming responses and reasoning_content extraction.

**gemini_analyzer.py** (1030 LOC) - Google Gemini API integration. Inherits from IFlowAIAnalyzer. Supports multi-model consensus via model='all' parameter.

**opencode_analyzer.py** (549 LOC) - OpenCode AI integration. Inherits from IFlowAIAnalyzer. Minimal implementation following base patterns.

**service.py** (~100 LOC) - Unified AIAnalysisService facade. Provider routing, model discovery, batch analysis orchestration.

## Conventions

- All analyzers inherit from IFlowAIAnalyzer - follow its _step_by_step_analysis pattern
- PredictionTracker for caching analysis results via data_cache/prediction_tracker.json
- Multi-model consensus triggered via model='all' parameter
- Async streaming supported via enable_streaming=True (NVIDIA/Gemini)
- All analyzers use OptimizedCache for AI result caching (TTL-based)
- Rate limiting via _last_api_call_time class variable with MIN_API_INTERVAL
- API endpoints configured in config.json, never hardcoded
- All prompts include formatted news via _format_news() method

## Anti-Patterns

- Never directly instantiate analyzers - always use AIAnalysisService
- Do not modify PredictionTracker.predictions directly - use track() method
- Never hardcode API endpoints - use config_manager.get_config().ai
- Do not bypass _rate_limit_lock for API calls
- Never modify FEW_SHOT_EXAMPLES in IFlowAIAnalyzer
- Do not call analyzer.analyze() directly - use service.analyze_stock()

## Dependencies

- **entities.py**: AIAnalysisResult dataclass, StockData model
- **cache_service.py**: OptimizedCache for AI result persistence
- **config_manager**: API keys, model lists, rate limits from config.json
- **constants.py**: VIX thresholds, default timeouts

## Architecture Notes

The analyzer hierarchy forms a template pattern - IFlowAIAnalyzer defines _step_by_step_analysis(), _build_analysis_prompt(), _call_api() hooks. Subclasses override _call_api() for provider-specific implementations while reusing the analysis pipeline.

PredictionTracker maintains prediction history in data_cache/prediction_tracker.json with automatic verification after 14 days. Accuracy metrics computed per model via get_model_accuracy().
