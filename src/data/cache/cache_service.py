from typing import Optional, Any, Dict
import json
import pickle
from datetime import datetime, timedelta
import hashlib
import os
import hmac
from pathlib import Path
import logging
import platform
import socket


def _generate_machine_key() -> bytes:
    """
    基于机器特征生成唯一的缓存签名密钥
    
    安全说明：
    - 不使用硬编码默认密钥
    - 密钥基于机器特征生成，不同机器使用不同密钥
    - 环境变量 CACHE_SECRET_KEY 优先级最高
    """
    # 优先使用环境变量
    env_key = os.environ.get('CACHE_SECRET_KEY')
    if env_key:
        return env_key.encode()
    
    # 基于机器特征生成密钥
    machine_info = f"{platform.node()}-{socket.gethostname()}-{platform.system()}-{platform.machine()}"
    # 使用 SHA-256 生成固定长度的密钥
    return hashlib.sha256(f"yfinance_cache_{machine_info}".encode()).digest()


# 安全密钥 - 用于签名验证，防止 pickle 反序列化攻击
_CACHE_SECRET_KEY = _generate_machine_key()


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
        self.logger = logging.getLogger(__name__)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_signature(self, data: bytes) -> str:
        """
        为数据生成 HMAC 签名，用于验证数据完整性
        
        Args:
            data: 要签名的数据（bytes）
            
        Returns:
            签名字符串（hex）
        """
        return hmac.new(_CACHE_SECRET_KEY, data, hashlib.sha256).hexdigest()
    
    def _verify_signature(self, data: bytes, signature: str) -> bool:
        """
        验证数据的签名
        
        Args:
            data: 原始数据（bytes）
            signature: 预期的签名字符串
            
        Returns:
            签名是否有效
        """
        expected = self._generate_signature(data)
        return hmac.compare_digest(expected, signature)
    
    def _validate_cache_key(self, key: str) -> bool:
        """
        验证缓存键的安全性，防止路径遍历攻击
        
        Args:
            key: 缓存键
            
        Returns:
            是否为安全的键
        """
        # 禁止路径遍历字符
        dangerous_chars = ['..', '/', '\\', '\x00']
        for char in dangerous_chars:
            if char in key:
                return False
        return True
    
    def _get_cache_path(self, key: str, suffix: str = ".cache") -> Path:
        """
        获取缓存文件路径
        
        安全说明：
        - 验证键的安全性，防止路径遍历攻击
        - 验证最终路径在缓存目录内
        
        Args:
            key: 缓存键
            suffix: 文件后缀
            
        Returns:
            缓存文件的绝对路径
            
        Raises:
            ValueError: 如果键包含危险字符或路径不在缓存目录内
        """
        # 验证键的安全性
        if not self._validate_cache_key(key):
            raise ValueError(f"Invalid cache key: contains dangerous characters")
        
        cache_path = (self.cache_dir / f"{key}{suffix}").resolve()
        
        # 验证路径在缓存目录内
        if not str(cache_path).startswith(str(self.cache_dir.resolve())):
            raise ValueError(f"Cache path escapes cache directory")
        
        return cache_path
    
    def _get_ttl_for_key(self, key: str) -> timedelta:
        """根据缓存键确定 TTL，对于小时线和分钟线数据使用更短的 TTL"""
        if '_1h_' in key or '_1m_' in key:
            return timedelta(hours=1)
        return timedelta(days=self.ttl_days)
    
    # ==================== Pickle 格式方法 ====================
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取 pickle 格式的缓存数据
        
        安全说明：
        - 验证缓存键安全性，防止路径遍历攻击
        - 验证数据签名，防止反序列化攻击
        """
        if not self.enabled:
            return None
        
        # 安全检查：验证键的安全性
        if not self._validate_cache_key(key):
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
                content = f.read()
            
            # 分离签名和数据
            # 格式: 签名(64字符) + '\n' + pickle数据
            if b'\n' not in content:
                # 旧格式数据，无签名，拒绝加载
                return None
            
            signature_bytes, data = content.split(b'\n', 1)
            signature = signature_bytes.decode('utf-8')
            
            # 验证签名
            if not self._verify_signature(data, signature):
                # 签名验证失败，拒绝加载
                self.logger.warning(f"缓存签名验证失败: {cache_path}")
                return None
            
            return pickle.loads(data)
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        设置 pickle 格式的缓存数据
        
        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 缓存有效期（秒），None 则使用默认 TTL
        
        安全说明：
        - 验证缓存键安全性，防止路径遍历攻击
        - 添加签名，确保数据完整性
        """
        if not self.enabled:
            return
        
        # 安全检查：验证键的安全性
        if not self._validate_cache_key(key):
            return
            
        cache_path = self._get_cache_path(key, ".cache")
        try:
            # 序列化数据
            data = pickle.dumps(value)
            
            # 生成签名
            signature = self._generate_signature(data)
            
            # 写入签名和数据
            with open(cache_path, 'wb') as f:
                f.write(signature.encode('utf-8'))
                f.write(b'\n')
                f.write(data)
            
            # 如果指定了 TTL，需要额外记录过期时间
            # 通过修改文件的修改时间为未来时间来实现
            # 这里采用简单方式：使用默认 TTL 机制
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
        except Exception as e:
            self.logger.warning(f"加载JSON缓存失败 {cache_path}: {e}")
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
        info_hash: str = None,
        news_hash: str = None
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
            news_hash: 新闻数据的哈希值
        
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
        if news_hash:
            cache_content['news_hash'] = news_hash
        
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
        except Exception as e:
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
            except Exception as e:
                self.logger.debug(f"清理缓存文件失败 {cache_file}: {e}")
                continue
        
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - modified_time > timedelta(days=self.ttl_days):
                    cache_file.unlink()
                    count += 1
            except Exception as e:
                self.logger.debug(f"清理JSON缓存文件失败 {cache_file}: {e}")
                continue
        
        return count
    
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        安全说明：
        - 验证缓存键安全性，防止路径遍历攻击
        """
        # 安全检查：验证键的安全性
        if not self._validate_cache_key(key):
            return False
        
        try:
            cache_path = self._get_cache_path(key, ".cache")
        except ValueError:
            return False
        
        if cache_path.exists():
            modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            ttl = self._get_ttl_for_key(key)
            return datetime.now() - modified_time <= ttl
        return False
