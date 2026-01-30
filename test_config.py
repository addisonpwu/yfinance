#!/usr/bin/env python3
"""
測試配置系統是否正常工作
"""
from src.config.settings import config_manager

def test_config():
    """測試配置系統"""
    print("\u6e2c試配置系統...")
    
    # 加載配置
    config = config_manager.get_config()
    
    print(f"API \u914d置:")
    print(f"  base_delay: {config.api.base_delay}")
    print(f"  max_delay: {config.api.max_delay}")
    print(f"  min_delay: {config.api.min_delay}")
    print(f"  retry_attempts: {config.api.retry_attempts}")
    print(f"  max_workers: {config.api.max_workers}")
    
    print(f"\n數據配置:")
    print(f"  max_cache_days: {config.data.max_cache_days}")
    print(f"  float_dtype: {config.data.float_dtype}")
    print(f"  data_download_period.1m: {config.data.data_download_period.m1}")
    print(f"  data_download_period.1h: {config.data.data_download_period.h1}")
    print(f"  data_download_period.1d: {config.data.data_download_period.d1}")
    
    print(f"\n分析配置:")
    print(f"  enable_realtime_output: {config.analysis.enable_realtime_output}")
    print(f"  enable_data_preprocessing: {config.analysis.enable_data_preprocessing}")
    print(f"  min_volume_threshold: {config.analysis.min_volume_threshold}")
    print(f"  min_data_points_threshold: {config.analysis.min_data_points_threshold}")
    
    print(f"\n技術指標配置:")
    print(f"  rsi_period: {config.technical_indicators.rsi_period}")
    print(f"  macd_fast: {config.technical_indicators.macd_fast}")
    print(f"  macd_slow: {config.technical_indicators.macd_slow}")
    print(f"  macd_signal: {config.technical_indicators.macd_signal}")
    print(f"  bb_period: {config.technical_indicators.bb_period}")
    print(f"  bb_std_dev: {config.technical_indicators.bb_std_dev}")
    print(f"  atr_period: {config.technical_indicators.atr_period}")
    print(f"  ma_periods: {config.technical_indicators.ma_periods}")
    
    print(f"\n策略配置 (VCP):")
    print(f"  vcp_ma_periods: {config.strategies.vcp_pocket_pivot.ma_periods}")
    print(f"  vcp_volatility_windows: {config.strategies.vcp_pocket_pivot.volatility_windows}")
    print(f"  vcp_volume_avg_period: {config.strategies.vcp_pocket_pivot.volume_avg_period}")
    
    print(f"\n策略配置 (布林帶擠壓):")
    print(f"  bb_period: {config.strategies.bollinger_squeeze.bb_period}")
    print(f"  squeeze_percentile: {config.strategies.bollinger_squeeze.squeeze_percentile}")
    
    print(f"\n新聞配置:")
    print(f"  timeout: {config.news.timeout}")
    print(f"  max_news_items: {config.news.max_news_items}")
    
    print(f"\nAI配置:")
    print(f"  api_timeout: {config.ai.api_timeout}")
    print(f"  model: {config.ai.model}")
    print(f"  max_data_points: {config.ai.max_data_points}")
    
    print("\n配置系統測試完成!")

if __name__ == "__main__":
    test_config()