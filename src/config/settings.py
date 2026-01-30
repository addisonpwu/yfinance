from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
from pathlib import Path
import os

@dataclass
class APIConfig:
    base_delay: float = 0.5
    max_delay: float = 2.0
    min_delay: float = 0.1
    retry_attempts: int = 3
    max_workers: int = 4

@dataclass
class DataConfig:
    max_cache_days: int = 7
    float_dtype: str = "float32"

@dataclass
class AnalysisConfig:
    enable_realtime_output: bool = True
    enable_data_preprocessing: bool = True
    min_volume_threshold: int = 100000

@dataclass
class AppConfig:
    api: APIConfig
    data: DataConfig
    analysis: AnalysisConfig

class ConfigManager:
    _instance = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: str = "config.json") -> AppConfig:
        if self._config is not None:
            return self._config
            
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            # 如果不存在，创建默认配置
            self._create_default_config(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = json.load(f)
        
        # 使用默认值填充缺失的配置项
        api_config = raw_config.get('api', {})
        data_config = raw_config.get('data', {})
        analysis_config = raw_config.get('analysis', {})
        
        self._config = AppConfig(
            api=APIConfig(
                base_delay=api_config.get('base_delay', 0.5),
                max_delay=api_config.get('max_delay', 2.0),
                min_delay=api_config.get('min_delay', 0.1),
                retry_attempts=api_config.get('retry_attempts', 3),
                max_workers=api_config.get('max_workers', 4)
            ),
            data=DataConfig(
                max_cache_days=data_config.get('max_cache_days', 7),
                float_dtype=data_config.get('float_dtype', 'float32')
            ),
            analysis=AnalysisConfig(
                enable_realtime_output=analysis_config.get('enable_realtime_output', True),
                enable_data_preprocessing=analysis_config.get('enable_data_preprocessing', True),
                min_volume_threshold=analysis_config.get('min_volume_threshold', 100000)
            )
        )
        return self._config
    
    def _create_default_config(self, config_path: str):
        """创建默认配置文件"""
        default_config = {
            "api": {
                "base_delay": 0.5,
                "max_delay": 2.0,
                "min_delay": 0.1,
                "retry_attempts": 3,
                "max_workers": 4
            },
            "data": {
                "max_cache_days": 7,
                "float_dtype": "float32"
            },
            "analysis": {
                "enable_realtime_output": True,
                "enable_data_preprocessing": True,
                "min_volume_threshold": 100000
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    def get_config(self) -> AppConfig:
        if self._config is None:
            self.load_config()
        return self._config

# 全局配置实例
config_manager = ConfigManager()