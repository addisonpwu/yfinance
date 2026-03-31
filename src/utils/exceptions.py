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


class DatabaseException(StockAnalysisException):
    """数据库操作异常基类"""

    pass


class ConnectionException(DatabaseException):
    """数据库连接异常"""

    pass


class RecordNotFoundException(DatabaseException):
    """记录未找到异常"""

    pass


class StockNotFoundException(RecordNotFoundException):
    """股票未找到异常"""

    pass


class NewsNotFoundException(RecordNotFoundException):
    """新闻未找到异常"""

    pass


class DuplicateRecordException(DatabaseException):
    """重复记录异常"""

    pass


class DatabaseOperationException(DatabaseException):
    """数据库操作异常"""

    pass
