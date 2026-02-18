"""
配置验证模块

使用 Pydantic 进行配置验证，确保：
1. 类型安全
2. 数值范围验证
3. 键名规范化（去除空格）
4. 必填字段检查

依赖：
- pydantic>=2.0 (可选，无则使用降级模式)
- python-dotenv (可选)
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

# 尝试导入 pydantic
try:
    from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

# 尝试导入 python-dotenv
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def load_env_file(env_path: str = ".env") -> None:
    """
    加载 .env 文件中的环境变量
    
    Args:
        env_path: .env 文件路径
    """
    if HAS_DOTENV:
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✅ 已加载环境变量文件: {env_path}")
        else:
            print(f"ℹ️  环境变量文件不存在: {env_path}，使用系统环境变量")
    else:
        print("ℹ️  python-dotenv 未安装，使用系统环境变量")


def get_env(key: str, default: str = None, required: bool = False) -> Optional[str]:
    """
    获取环境变量
    
    Args:
        key: 环境变量名
        default: 默认值
        required: 是否必需
        
    Returns:
        环境变量值
    """
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"必需的环境变量 {key} 未设置")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔型环境变量"""
    value = os.environ.get(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """获取整型环境变量"""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """获取浮点型环境变量"""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ==================== 简单配置验证（无 pydantic 时使用）====================

def normalize_keys(raw_config: Dict) -> Dict:
    """
    规范化配置键名（去除空格）
    
    Args:
        raw_config: 原始配置字典
        
    Returns:
        规范化后的配置字典
    """
    if not isinstance(raw_config, dict):
        return raw_config
        
    normalized = {}
    for key, value in raw_config.items():
        clean_key = key.strip()
        
        if isinstance(value, dict):
            normalized[clean_key] = normalize_keys(value)
        elif isinstance(value, list):
            normalized[clean_key] = [
                normalize_keys(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            normalized[clean_key] = value
            
    return normalized


def validate_config_simple(raw_config: Dict) -> Dict:
    """
    简单配置验证（无 pydantic 时使用）
    
    Args:
        raw_config: 原始配置字典
        
    Returns:
        验证后的配置字典
    """
    errors = []
    warnings = []
    
    # 规范化键名
    config = normalize_keys(raw_config)
    
    # 验证 speed_mode
    speed_mode = config.get('speed_mode', 'balanced')
    if speed_mode not in ['fast', 'balanced', 'safe']:
        warnings.append(f"speed_mode '{speed_mode}' 无效，使用 'balanced'")
        config['speed_mode'] = 'balanced'
    
    # 验证 API 配置
    api = config.get('api', {})
    if api:
        if not isinstance(api.get('base_delay'), (int, float)):
            warnings.append("api.base_delay 类型无效")
        elif api.get('base_delay', 0) < 0:
            errors.append("api.base_delay 不能为负数")
        
        if not isinstance(api.get('max_workers'), int):
            warnings.append("api.max_workers 类型无效")
        elif api.get('max_workers', 0) < 1:
            errors.append("api.max_workers 必须大于 0")
    
    # 验证数据配置
    data = config.get('data', {})
    if data:
        if data.get('max_cache_days', 7) < 1:
            warnings.append("data.max_cache_days 必须大于 0")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    if warnings:
        print("⚠️  配置验证警告:")
        for w in warnings:
            print(f"   - {w}")
    
    return config


# ==================== Pydantic 配置模型（可选）====================

if HAS_PYDANTIC:
    # 使用 Pydantic 进行严格验证
    class APIConfigModel(BaseModel):
        model_config = ConfigDict(extra='ignore')
        base_delay: float = Field(default=0.5, ge=0.1, le=10.0)
        max_delay: float = Field(default=3.0, ge=0.5, le=30.0)
        min_delay: float = Field(default=0.2, ge=0.05, le=5.0)
        retry_attempts: int = Field(default=3, ge=1, le=10)
        max_workers: int = Field(default=4, ge=1, le=16)
        
        @model_validator(mode='after')
        def validate_delays(self):
            if self.min_delay > self.base_delay:
                raise ValueError(f"min_delay({self.min_delay}) 不能大于 base_delay({self.base_delay})")
            if self.base_delay > self.max_delay:
                raise ValueError(f"base_delay({self.base_delay}) 不能大于 max_delay({self.max_delay})")
            return self

    class DataConfigModel(BaseModel):
        model_config = ConfigDict(extra='ignore')
        max_cache_days: int = Field(default=7, ge=1, le=30)
        float_dtype: str = Field(default="float32")
        data_download_period: Optional[Dict[str, str]] = Field(default_factory=dict)
        enable_cache: bool = Field(default=True)
        enable_finviz: bool = Field(default=True)
        
        @field_validator('float_dtype')
        @classmethod
        def validate_float_dtype(cls, v):
            if v not in ['float16', 'float32', 'float64']:
                raise ValueError(f"float_dtype 必须是 float16/float32/float64 之一")
            return v

    class AIConfigModel(BaseModel):
        model_config = ConfigDict(extra='ignore')
        api_timeout: int = Field(default=30, ge=10, le=120)
        model: str = Field(default="deepseek-v3.2")
        max_data_points: int = Field(default=100, ge=20, le=500)

    class AppConfigModel(BaseModel):
        model_config = ConfigDict(extra='ignore')
        speed_mode: str = Field(default="balanced")
        api: APIConfigModel = Field(default_factory=APIConfigModel)
        data: DataConfigModel = Field(default_factory=DataConfigModel)
        ai: AIConfigModel = Field(default_factory=AIConfigModel)
        
        @field_validator('speed_mode')
        @classmethod
        def validate_speed_mode(cls, v):
            if v not in ['fast', 'balanced', 'safe']:
                raise ValueError(f"speed_mode 必须是 fast/balanced/safe 之一")
            return v
else:
    # 降级模式：使用简单的 dataclass
    @dataclass
    class APIConfigModel:
        base_delay: float = 0.5
        max_delay: float = 3.0
        min_delay: float = 0.2
        retry_attempts: int = 3
        max_workers: int = 4

    @dataclass
    class DataConfigModel:
        max_cache_days: int = 7
        float_dtype: str = "float32"
        data_download_period: Dict = None
        enable_cache: bool = True
        enable_finviz: bool = True
        
        def __post_init__(self):
            if self.data_download_period is None:
                self.data_download_period = {}

    @dataclass
    class AIConfigModel:
        api_timeout: int = 30
        model: str = "deepseek-v3.2"
        max_data_points: int = 100

    @dataclass
    class AppConfigModel:
        speed_mode: str = "balanced"
        api: APIConfigModel = None
        data: DataConfigModel = None
        ai: AIConfigModel = None
        
        def __post_init__(self):
            if self.api is None:
                self.api = APIConfigModel()
            if self.data is None:
                self.data = DataConfigModel()
            if self.ai is None:
                self.ai = AIConfigModel()


# ==================== 配置验证器 ====================

class ConfigValidator:
    """配置验证器"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._raw_config: Dict = {}
        self._validated_config: Optional[AppConfigModel] = None
        self._validation_errors: List[str] = []
        self._validation_warnings: List[str] = []
    
    def load_and_validate(self) -> AppConfigModel:
        """
        加载并验证配置文件
        
        Returns:
            验证后的配置模型
        """
        self._raw_config = self._load_raw_config()
        
        if HAS_PYDANTIC:
            # 使用 Pydantic 验证
            normalized_config = normalize_keys(self._raw_config)
            try:
                self._validated_config = AppConfigModel(**normalized_config)
                print(f"✅ 配置文件验证通过: {self.config_path}")
            except Exception as e:
                self._validation_errors.append(str(e))
                raise ValueError(f"配置验证失败:\n{e}")
        else:
            # 使用简单验证
            try:
                validated_dict = validate_config_simple(self._raw_config)
                self._validated_config = AppConfigModel(
                    speed_mode=validated_dict.get('speed_mode', 'balanced'),
                    api=APIConfigModel(**validated_dict.get('api', {})),
                    data=DataConfigModel(**validated_dict.get('data', {})),
                    ai=AIConfigModel(**validated_dict.get('ai', {}))
                )
                print(f"✅ 配置文件验证通过 (简化模式): {self.config_path}")
            except Exception as e:
                self._validation_errors.append(str(e))
                raise ValueError(f"配置验证失败:\n{e}")
        
        return self._validated_config
    
    def _load_raw_config(self) -> Dict:
        """加载原始配置文件"""
        if not os.path.exists(self.config_path):
            print(f"⚠️  配置文件不存在: {self.config_path}，将使用默认配置")
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件 JSON 格式错误: {e}")
    
    def get_validated_config(self) -> AppConfigModel:
        """获取已验证的配置"""
        if self._validated_config is None:
            self._validated_config = self.load_and_validate()
        return self._validated_config
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        if self._validated_config is None:
            return {}
        if HAS_PYDANTIC:
            return self._validated_config.model_dump()
        else:
            return {
                'speed_mode': self._validated_config.speed_mode,
                'api': {
                    'base_delay': self._validated_config.api.base_delay,
                    'max_delay': self._validated_config.api.max_delay,
                    'min_delay': self._validated_config.api.min_delay,
                    'retry_attempts': self._validated_config.api.retry_attempts,
                    'max_workers': self._validated_config.api.max_workers,
                },
                'data': {
                    'max_cache_days': self._validated_config.data.max_cache_days,
                    'float_dtype': self._validated_config.data.float_dtype,
                    'enable_cache': self._validated_config.data.enable_cache,
                    'enable_finviz': self._validated_config.data.enable_finviz,
                },
                'ai': {
                    'api_timeout': self._validated_config.ai.api_timeout,
                    'model': self._validated_config.ai.model,
                    'max_data_points': self._validated_config.ai.max_data_points,
                }
            }
    
    @property
    def errors(self) -> List[str]:
        return self._validation_errors
    
    @property
    def warnings(self) -> List[str]:
        return self._validation_warnings


# ==================== 敏感信息管理 ====================

class SecretsManager:
    """敏感信息管理器"""
    
    SENSITIVE_KEYS = [
        'IFLOW_API_KEY',
        'IFLOW_API_BASE_URL',
        'FINVIZ_API_KEY',
        'HTTP_PROXY',
        'HTTPS_PROXY',
    ]
    
    def __init__(self, env_path: str = ".env"):
        self._env_path = env_path
        self._secrets: Dict[str, str] = {}
        self._load_secrets()
    
    def _load_secrets(self) -> None:
        """加载敏感信息"""
        load_env_file(self._env_path)
        for key in self.SENSITIVE_KEYS:
            value = os.environ.get(key)
            if value:
                self._secrets[key] = value
    
    def get(self, key: str, default: str = None, required: bool = False) -> Optional[str]:
        value = self._secrets.get(key, default)
        if required and not value:
            raise ValueError(f"必需的敏感信息 {key} 未设置。请在 .env 文件或环境变量中设置。")
        return value
    
    def get_iflow_api_key(self) -> str:
        return self.get('IFLOW_API_KEY', required=True)
    
    def get_iflow_base_url(self) -> str:
        return self.get('IFLOW_API_BASE_URL', default='https://api.iflow.com/v1')
    
    def is_configured(self, key: str) -> bool:
        return bool(self._secrets.get(key))
    
    @property
    def configured_keys(self) -> List[str]:
        return list(self._secrets.keys())
    
    def validate_required_secrets(self) -> List[str]:
        required_keys = ['IFLOW_API_KEY']
        missing = []
        for key in required_keys:
            if not self.is_configured(key):
                missing.append(key)
        return missing


# ==================== 全局实例 ====================

_secrets_manager: Optional[SecretsManager] = None
_config_validator: Optional[ConfigValidator] = None


def get_secrets_manager(env_path: str = ".env") -> SecretsManager:
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(env_path)
    return _secrets_manager


def get_config_validator(config_path: str = "config.json") -> ConfigValidator:
    global _config_validator
    if _config_validator is None:
        _config_validator = ConfigValidator(config_path)
    return _config_validator


def validate_startup(config_path: str = "config.json", env_path: str = ".env") -> bool:
    """
    启动时验证配置和敏感信息
    """
    print("\n" + "=" * 50)
    print("系统启动验证")
    print("=" * 50)
    
    # 验证配置文件
    try:
        validator = get_config_validator(config_path)
        validator.load_and_validate()
    except ValueError as e:
        print(f"❌ 配置验证失败: {e}")
        return False
    
    # 验证敏感信息
    secrets = get_secrets_manager(env_path)
    missing = secrets.validate_required_secrets()
    
    if missing:
        print(f"⚠️  以下敏感信息未配置: {', '.join(missing)}")
        print(f"   请创建 {env_path} 文件并填写必要的密钥")
        return False
    
    print(f"✅ 敏感信息验证通过")
    print(f"   已配置: {', '.join(secrets.configured_keys) if secrets.configured_keys else '无'}")
    print("=" * 50 + "\n")
    
    return True