import logging
from typing import Optional

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
        
        # 只保留控制台处理器，不生成日志文件
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)

        # File logging with rotation
        try:
            import os
            from logging.handlers import RotatingFileHandler
            
            os.makedirs("logs", exist_ok=True)
            file_handler = RotatingFileHandler(
                "logs/app.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")

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