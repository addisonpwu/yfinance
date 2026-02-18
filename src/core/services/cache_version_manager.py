"""
缓存版本管理器

负责缓存版本检查和增量更新逻辑
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class CacheVersionManager:
    """缓存版本管理器"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
    
    def get_version_file(self, market: str) -> Path:
        """获取版本文件路径"""
        return self.cache_dir / market.upper() / "version.txt"
    
    def check_version(self, market: str, force_fast_mode: bool = False, interval: str = '1d') -> Tuple[bool, str]:
        """
        检查缓存版本
        
        Args:
            market: 市场代码
            force_fast_mode: 是否强制快速模式
            interval: 数据时段类型
        
        Returns:
            (是否需要同步, 状态消息)
        """
        
        today_str = datetime.now().date().isoformat()
        version_file = self.get_version_file(market)
        
        if force_fast_mode:
            return False, "--- 强制快速模式：跳过缓存更新检查 ---"
        
        try:
            with open(version_file, 'r') as f:
                last_sync_date = f.read().strip()
            
            if last_sync_date == today_str:
                # 对于小时线或分钟线，即使当天已有缓存，也应检查是否需要更新
                if interval in ['1h', '1m']:
                    return True, f"--- 使用 {interval} 间隔，将执行数据同步以获取最新数据 ---"
                else:
                    return False, f"--- 数据缓存已是最新 ({today_str})，将以快速模式运行 ---"
            else:
                return True, f"--- 数据缓存不是最新 (版本: {last_sync_date})，将执行增量同步 ---"
                
        except FileNotFoundError:
            return True, "--- 未找到缓存版本文件，将执行首次同步 ---"
    
    def update_version(self, market: str) -> None:
        """更新缓存版本"""
        version_file = self.get_version_file(market)
        version_file.parent.mkdir(parents=True, exist_ok=True)
        
        today_str = datetime.now().date().isoformat()
        with open(version_file, 'w') as f:
            f.write(today_str)
        
        print(f"--- 更新缓存版本至 {today_str} ---")
    
    def get_last_sync_date(self, market: str) -> Optional[str]:
        """获取上次同步日期"""
        version_file = self.get_version_file(market)
        
        try:
            with open(version_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
