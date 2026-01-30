#!/usr/bin/env python3
"""
最終測試：驗證禁用緩存選項是否正常工作
"""
import json
import os
from src.config.settings import config_manager

def test_config_with_cache_disabled():
    """測試配置系統與禁用緩存選項"""
    # 創建一個臨時配置文件，禁用緩存
    temp_config = {
        "api": {
            "base_delay": 0.5,
            "max_delay": 2.0,
            "min_delay": 0.1,
            "retry_attempts": 3,
            "max_workers": 4
        },
        "data": {
            "max_cache_days": 7,
            "float_dtype": "float32",
            "data_download_period": {
                "1m": "7d",
                "1h": "730d",
                "1d": "max"
            },
            "enable_cache": False  # 禁用緩存
        }
    }
    
    # 保存到臨時文件
    temp_config_file = "temp_test_config.json"
    with open(temp_config_file, 'w', encoding='utf-8') as f:
        json.dump(temp_config, f, ensure_ascii=False, indent=2)
    
    try:
        # 加載配置
        config = config_manager.load_config(temp_config_file)
        
        print(f"緩存是否啟用: {config.data.enable_cache}")
        
        if not config.data.enable_cache:
            print("✓ 配置系統正確讀取了禁用緩存設置")
        else:
            print("✗ 配置系統未能正確讀取禁用緩存設置")
        
        # 測試創建禁用緩存的實例
        from src.data.cache.cache_service import OptimizedCache
        cache = OptimizedCache(enabled=config.data.enable_cache)
        
        # 測試設置和獲取值
        cache.set("test", "value")
        result = cache.get("test")
        
        if result is None:
            print("✓ 緩存禁用功能正常工作：get 返回 None")
        else:
            print("✗ 緩存禁用功能異常：get 返回了值")
            
    finally:
        # 清理臨時文件
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)

def test_config_with_cache_enabled():
    """測試配置系統與啟用緩存選項"""
    # 創建一個完整的臨時配置文件，啟用緩存
    temp_config = {
        "api": {
            "base_delay": 0.5,
            "max_delay": 2.0,
            "min_delay": 0.1,
            "retry_attempts": 3,
            "max_workers": 4
        },
        "data": {
            "max_cache_days": 7,
            "float_dtype": "float32",
            "data_download_period": {
                "1m": "7d",
                "1h": "730d",
                "1d": "max"
            },
            "enable_cache": True  # 啟用緩存
        },
        "analysis": {
            "enable_realtime_output": True,
            "enable_data_preprocessing": True,
            "min_volume_threshold": 100000,
            "min_data_points_threshold": 20
        },
        "technical_indicators": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std_dev": 2,
            "atr_period": 14,
            "ma_periods": [5, 10, 20, 50, 200]
        },
        "strategies": {
            "vcp_pocket_pivot": {
                "ma_periods": [50, 150, 200],
                "volatility_windows": [50, 20, 10],
                "volume_avg_period": 50,
                "pp_lookback_period": 10,
                "pp_max_bias_ratio": 0.08
            },
            "bollinger_squeeze": {
                "bb_period": 20,
                "squeeze_lookback": 100,
                "squeeze_percentile": 0.10,
                "prolonged_squeeze_period": 5,
                "long_trend_period": 200,
                "ma_slope_period": 5,
                "volume_period": 50
            }
        },
        "news": {
            "timeout": 60000,
            "max_news_items": 5
        },
        "ai": {
            "api_timeout": 30,
            "model": "deepseek-v3.2",
            "max_data_points": 100
        }
    }
    
    # 保存到臨時文件
    temp_config_file = "temp_test_config2.json"
    with open(temp_config_file, 'w', encoding='utf-8') as f:
        json.dump(temp_config, f, ensure_ascii=False, indent=2)
    
    try:
        # 加載配置
        config = config_manager.load_config(temp_config_file)
        
        print(f"\n緩存是否啟用: {config.data.enable_cache}")
        
        if config.data.enable_cache:
            print("✓ 配置系統正確讀取了啟用緩存設置")
        else:
            print("✗ 配置系統未能正確讀取啟用緩存設置")
        
        # 測試創建啟用緩存的實例
        from src.data.cache.cache_service import OptimizedCache
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = OptimizedCache(cache_dir=temp_dir, enabled=config.data.enable_cache)
            
            # 測試設置和獲取值
            cache.set("test", "value")
            result = cache.get("test")
            
            if result == "value":
                print("✓ 緩存啟用功能正常工作：get 返回正確值")
            else:
                print("✗ 緩存啟用功能異常：get 未返回正確值")
                
    finally:
        # 清理臨時文件
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)

if __name__ == "__main__":
    test_config_with_cache_disabled()
    test_config_with_cache_enabled()
    print("\n禁用緩存選項測試完成!")
