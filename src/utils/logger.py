import logging
from typing import Optional
import os
from datetime import datetime

class LoggerManager:
    _instances = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        if name not in cls._instances:
            logger = logging.getLogger(name)
            if not logger.handlers:
                cls._setup_logger(logger, name)
            cls._instances[name] = logger
        return cls._instances[name]
    
    @classmethod
    def _setup_logger(cls, logger: logging.Logger, name: str):
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

# 为不同模块提供专门的logger
def get_core_logger():
    return LoggerManager.get_logger("core")

def get_data_logger():
    return LoggerManager.get_logger("data")

def get_ai_logger():
    return LoggerManager.get_logger("ai")

def get_analysis_logger():
    return LoggerManager.get_logger("analysis")

def get_app_logger():
    return LoggerManager.get_logger("app")