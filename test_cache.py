#!/usr/bin/env python3
"""
测试缓存功能是否按预期工作
"""
import pandas as pd
from datetime import datetime, timedelta
from src.data.cache.cache_service import OptimizedCache

def test_cache_ttl():
    """测试不同时间间隔数据的TTL策略"""
    cache = OptimizedCache()
    
    # 测试日线数据缓存键（应该使用7天TTL）
    daily_key = "0001.HK_1d_HK"
    # 测试小时线数据缓存键（应该使用1小时TTL）
    hourly_key = "0001.HK_1h_HK"
    # 测试分钟线数据缓存键（应该使用1小时TTL）
    minute_key = "0001.HK_1m_HK"
    
    # 验证TTL函数
    print("测试TTL策略:")
    print(f"日线缓存TTL: {cache._get_ttl_for_key(daily_key)}")
    print(f"小时线缓存TTL: {cache._get_ttl_for_key(hourly_key)}")
    print(f"分钟线缓存TTL: {cache._get_ttl_for_key(minute_key)}")
    
    # 创建一些测试数据
    test_data = {'test': 'data', 'timestamp': datetime.now().isoformat()}
    
    # 测试缓存设置和获取
    print("\n测试缓存设置和获取...")
    cache.set(daily_key, test_data)
    cached_data = cache.get(daily_key)
    
    if cached_data and cached_data['test'] == 'data':
        print("✓ 缓存设置和获取功能正常")
    else:
        print("✗ 缓存设置和获取功能异常")

if __name__ == "__main__":
    test_cache_ttl()