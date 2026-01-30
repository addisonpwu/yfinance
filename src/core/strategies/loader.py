import pkgutil
import importlib
import inspect
from typing import List
from src.core.strategies.strategy import BaseStrategy

def get_strategies() -> List[BaseStrategy]:
    """
    動態從 strategies 模組加載所有策略類別的實例。
    """
    strategies = []
    
    # 导入strategies模块
    import src.strategies as strategies_module
    strategy_path = strategies_module.__path__

    for _, name, _ in pkgutil.iter_modules(strategy_path):
        if name != 'base_strategy':
            try:
                module = importlib.import_module(f"src.strategies.{name}")
                for item_name, item in inspect.getmembers(module, inspect.isclass):
                    if issubclass(item, BaseStrategy) and item is not BaseStrategy:
                        strategies.append(item())
            except ImportError as e:
                print(f"无法导入策略模块 {name}: {e}")
            except Exception as e:
                print(f"加载策略 {name} 时出错: {e}")
                
    return strategies