from typing import Optional, Any, Dict
import json
import pickle
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path


class OptimizedCache:
    """
    统一的缓存服务，支持 pickle 和 JSON 两种格式
    
    - pickle 格式：用于缓存 Python 对象（如 DataFrame）
    - JSON 格式：用于缓存 AI 分析结果等需要可读性的数据
    """
    
    def __init__(self, cache_dir: str = "data_cache", ttl_days: int = 7, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str, suffix: str = ".cache") -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}{suffix}"
    
    def _get_ttl_for_key(self, key: str) -> timedelta:
        """根据缓存键确定 TTL，对于小时线和分钟线数据使用更短的 TTL"""
        if '_1h_' in key or '_1m_' in key:
            return timedelta(hours=1)
        return timedelta(days=self.ttl_days)
    
    # ==================== Pickle 格式方法 ====================
    
    def get(self, key: str) -> Optional[Any]:
        """获取 pickle 格式的缓存数据"""
        if not self.enabled:
            return None
            
        cache_path = self._get_cache_path(key, ".cache")
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
        """设置 pickle 格式的缓存数据"""
        if not self.enabled:
            return
            
        cache_path = self._get_cache_path(key, ".cache")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"缓存写入失败: {e}")
    
    # ==================== JSON 格式方法 ====================
    
    def get_json(self, key: str, subdir: str = None) -> Optional[Dict]:
        """
        获取 JSON 格式的缓存数据
        
        Args:
            key: 缓存键
            subdir: 子目录（如 'ai_analysis'）
        
        Returns:
            缓存的字典数据，如果不存在或已过期则返回 None
        """
        if not self.enabled:
            return None
        
        # 确定缓存路径
        if subdir:
            cache_dir = self.cache_dir / subdir
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            cache_dir = self.cache_dir
        
        cache_path = cache_dir / f"{key}.json"
        
        if not cache_path.exists():
            return None
        
        # 检查过期时间
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        ttl = self._get_ttl_for_key(key)
        if datetime.now() - modified_time > ttl:
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def set_json(self, key: str, value: Dict, subdir: str = None) -> None:
        """
        设置 JSON 格式的缓存数据
        
        Args:
            key: 缓存键
            value: 要缓存的字典数据
            subdir: 子目录（如 'ai_analysis'）
        """
        if not self.enabled:
            return
        
        # 确定缓存路径
        if subdir:
            cache_dir = self.cache_dir / subdir
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            cache_dir = self.cache_dir
        
        cache_path = cache_dir / f"{key}.json"
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"JSON 缓存写入失败: {e}")
    
    # ==================== 缓存键生成方法 ====================
    
    def generate_key(self, *args, **kwargs) -> str:
        """根据输入参数生成缓存键"""
        data = {
            'args': args,
            'kwargs': kwargs
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def generate_ai_cache_key(
        self,
        symbol: str,
        strategies: list,
        market: str,
        interval: str,
        model: str,
        hist_hash: str = None,
        info_hash: str = None
    ) -> str:
        """
        生成 AI 分析结果的缓存键
        
        使用数据的哈希值而非日期，确保数据不变时缓存可复用
        
        Args:
            symbol: 股票代码
            strategies: 符合的策略列表
            market: 市场代码
            interval: 数据时段
            model: AI 模型名称
            hist_hash: 历史数据的哈希值
            info_hash: 基本面信息的哈希值
        
        Returns:
            缓存键字符串
        """
        cache_content = {
            'symbol': symbol,
            'strategies': sorted(strategies) if strategies else [],
            'market': market,
            'interval': interval,
            'model': model,
        }
        
        # 使用数据哈希而非日期，确保数据不变时缓存有效
        if hist_hash:
            cache_content['hist_hash'] = hist_hash
        if info_hash:
            cache_content['info_hash'] = info_hash
        
        cache_str = json.dumps(cache_content, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    @staticmethod
    def compute_data_hash(data: Any) -> str:
        """
        计算数据的哈希值，用于缓存键
        
        Args:
            data: 任意可序列化的数据
        
        Returns:
            哈希字符串
        """
        try:
            if hasattr(data, 'to_json'):
                # DataFrame 或类似对象
                data_str = data.to_json()
            else:
                data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()[:16]
        except Exception:
            # 如果无法序列化，使用时间戳作为后备
            return datetime.now().strftime('%Y%m%d')
    
    # ==================== 缓存管理方法 ====================
    
    def clear_expired(self) -> int:
        """清理所有过期的缓存文件，返回清理的数量"""
        count = 0
        if not self.cache_dir.exists():
            return count
        
        for cache_file in self.cache_dir.rglob("*.cache"):
            try:
                modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - modified_time > timedelta(days=self.ttl_days):
                    cache_file.unlink()
                    count += 1
            except Exception:
                continue
        
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - modified_time > timedelta(days=self.ttl_days):
                    cache_file.unlink()
                    count += 1
            except Exception:
                continue
        
        return count
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        cache_path = self._get_cache_path(key, ".cache")
        if cache_path.exists():
            modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            ttl = self._get_ttl_for_key(key)
            return datetime.now() - modified_time <= ttl
        return False
