"""
进度追踪器

负责进度显示和预估时间计算
"""
import time
from typing import Optional


class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, total: int):
        """
        初始化进度追踪器
        
        Args:
            total: 总任务数
        """
        self.total = total
        self.completed = 0
        self.successful = 0
        self.start_time = time.time()
    
    def update(self, success: bool = True) -> None:
        """
        更新进度
        
        Args:
            success: 是否成功完成
        """
        self.completed += 1
        if success:
            self.successful += 1
    
    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> Optional[int]:
        """
        获取预估剩余时间（分钟）
        
        Returns:
            剩余分钟数，如果无法预估返回 None
        """
        if self.completed == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        avg_time = elapsed / self.completed
        remaining = (self.total - self.completed) * avg_time
        return max(0, int(remaining / 60))
    
    def get_progress_bar(self, width: int = 20) -> str:
        """
        获取进度条字符串
        
        Args:
            width: 进度条宽度
        
        Returns:
            进度条字符串
        """
        progress = self.completed / self.total if self.total > 0 else 0
        filled = int(progress * width)
        empty = width - filled
        return '[' + '#' * filled + '-' * empty + ']'
    
    def format_status(self) -> str:
        """
        格式化状态字符串
        
        Returns:
            状态字符串
        """
        progress_bar = self.get_progress_bar()
        remaining = self.get_remaining_time()
        
        if remaining is not None:
            return (f"\r分析进度: {progress_bar} {self.completed}/{self.total} 已分析, "
                   f"{self.successful} 符合条件, 预估剩余: {remaining} 分钟")
        else:
            return (f"\r分析进度: {progress_bar} {self.completed}/{self.total} 已分析, "
                   f"{self.successful} 符合条件")
    
    def get_summary(self) -> str:
        """
        获取摘要信息
        
        Returns:
            摘要字符串
        """
        elapsed = self.get_elapsed_time()
        minutes = int(elapsed / 60)
        seconds = int(elapsed % 60)
        
        return (f"\n--- 分析完成！成功分析 {self.completed}/{self.total} 支股票，"
               f"找到 {self.successful} 支符合条件的股票 ---\n"
               f"--- 总耗时: {minutes} 分钟 {seconds} 秒 ---")
