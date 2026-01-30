from typing import Optional, Any
import json
import pickle
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

class OptimizedCache:
    def __init__(self, cache_dir: str = "data_cache", ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        # 检查过期时间
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(days=self.ttl_days):
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, key: str, value: Any) -> None:
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"缓存写入失败: {e}")
    
    def generate_key(self, *args, **kwargs) -> str:
        """根据输入参数生成缓存键"""
        data = {
            'args': args,
            'kwargs': kwargs
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()