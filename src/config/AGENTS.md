# Config Directory

Configuration management for the stock screening system. Uses Pydantic for validation, constants.py as the single source of truth, and .env for secrets.

## Key Files

| File | Purpose |
|------|---------|
| `constants.py` | SINGLE SOURCE OF TRUTH. All default values defined here. 453 lines of constants. |
| `settings.py` | 16 Pydantic dataclasses: AppConfig, APIConfig, DataConfig, AnalysisConfig, TechnicalIndicatorsConfig, StrategiesConfig, NewsConfig, AIConfig, BacktestConfig, PositionConfig, SpeedModePreset |
| `config_validator.py` | Runtime config validation. SecretsManager for .env files. 483 lines |
| `__init__.py` | Exports config_manager singleton instance |

## Conventions

- **ALL defaults in constants.py ONLY** - no hardcoded values in other modules
- Use `get_config_manager()` to access the singleton ConfigManager
- Speed modes: `fast` (8 workers, 0.2s delay) / `balanced` (4 workers, 0.5s delay) / `safe` (2 workers, 1.0s delay)
- Secrets live in `.env` file, validated via SecretsManager
- Config loaded from `config.json` at startup via Pydantic validation

## Anti-Patterns

- **NEVER** hardcode values in modules - import from `src.config.constants`
- Don't create new `ConfigManager` instances - use the exported `config_manager` singleton
- Never commit `.env` file - it's gitignored for a reason
- Don't bypass Pydantic validation - always call `validate_startup()` on boot

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
