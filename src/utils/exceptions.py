class StockAnalysisException(Exception):
    """基础异常类"""
    pass

class DataFetchException(StockAnalysisException):
    """数据获取异常"""
    pass

class AnalysisException(StockAnalysisException):
    """分析过程异常"""
    pass

class CacheException(StockAnalysisException):
    """缓存相关异常"""
    pass

class ConfigException(StockAnalysisException):
    """配置相关异常"""
    pass

class AIException(StockAnalysisException):
    """AI分析相关异常"""
    pass