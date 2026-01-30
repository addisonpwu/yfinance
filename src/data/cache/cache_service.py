from typing import Optional, Any
import json
import pickle
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

class OptimizedCache:
    def __init__(self, cache_dir: str = "data_cache", ttl_days: int = 7, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cache"
    
    def _get_ttl_for_key(self, key: str) -> timedelta:
        """根据缓存键确定TTL，对于小时线和分钟线数据使用更短的TTL"""
        # 检查key是否包含小时线或分钟线标识
        if '_1h_' in key or '_1m_' in key:
            # 小时线和分钟线数据TTL为1小时
            return timedelta(hours=1)
        # 其他数据（如日线）使用默认的TTL
        return timedelta(days=self.ttl_days)
    
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
            
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        # 检查过期时间
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        ttl = self._get_ttl_for_key(key)
        if datetime.now() - modified_time > ttl:
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
            
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