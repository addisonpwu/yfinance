# Config Directory

**Generated:** 2026-04-05 | **Commit:** 9f0acb5 | **Branch:** master

## Overview
配置管理系统，使用 Pydantic 验证，constants.py 作为单一事实来源，.env 管理敏感信息。

## Key Files

| File | Purpose |
|------|---------|
| `constants.py` | **SINGLE SOURCE OF TRUTH** - 所有默认值定义在此 (453 行) |
| `settings.py` | 16 个 Pydantic 数据类: AppConfig, APIConfig, DataConfig, AnalysisConfig, TechnicalIndicatorsConfig, StrategiesConfig, NewsConfig, AIConfig, BacktestConfig, PositionConfig, SpeedModePreset |
| `config_validator.py` | 运行时配置验证，SecretsManager 管理 .env (483 行) |
| `__init__.py` | 导出 config_manager 单例实例 |

## Conventions

- **所有默认值仅在 constants.py** - 其他模块不硬编码
- 使用 `get_config_manager()` 获取单例 ConfigManager
- 速度模式: `fast`(8 workers, 0.2s) / `balanced`(4 workers, 0.5s) / `safe`(2 workers, 1.0s)
- 密钥在 `.env` 文件中，通过 SecretsManager 验证
- 配置从 `config.json` 加载，通过 Pydantic 验证

## Anti-Patterns

- **NEVER** 在模块中硬编码值 - 从 `src.config.constants` 导入
- **NEVER** 创建新的 `ConfigManager` 实例 - 使用导出的 `config_manager` 单例
- **NEVER** 提交 `.env` 文件 - 它是 gitignored 的
- **NEVER** 绕过 Pydantic 验证 - 启动时调用 `validate_startup()`

## Hierarchy

```
constants.py     (defaults - single source of truth)
       ↓
config.json      (runtime overrides)
       ↓
.env             (secrets: API keys)
       ↓
Pydantic validation  (settings.py classes)
       ↓
config_manager   (singleton runtime access)
```

## Usage

```python
from src.config import config_manager, get_config_manager
from src.config.constants import API_DEFAULT_BASE_DELAY

# Access config via singleton
api_delay = config_manager.config.api.base_delay

# Or use getter
config = get_config_manager().config

# Import constants directly
from src.config.constants import VIX_THRESHOLD_HIGH, SPEED_MODE_BALANCED
```
