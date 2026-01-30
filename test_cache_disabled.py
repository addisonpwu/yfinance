#!/usr/bin/env python3
"""
測試禁用緩存功能
"""
import os
import tempfile
from src.data.cache.cache_service import OptimizedCache

def test_cache_disabled():
    """測試緩存禁用功能"""
    print("\u6e2c\u8a66\u7de9\u5b58\u7981\u7528\u529f\u80fd...")
    
    # 創建一個禁用的緩存實例
    cache = OptimizedCache(enabled=False)
    
    # 測試 set 方法 - 應該不保存任何內容
    cache.set("test_key", "test_value")
    
    # 測試 get 方法 - 應該返回 None
    result = cache.get("test_key")
    
    print(f"\u7de9\u5b58\u7981\u7528\u6642 set \u5f8c get \u7d50\u679c: {result}")
    print(f"\u9810\u671f\u7d50\u679c: None (\u56e0\u70ba\u7de9\u5b58\u88ab\u7981\u7528)")
    
    if result is None:
        print("\u2713 \u7de9\u5b58\u7981\u7528\u529f\u80fd\u6b63\u5e38\u5de5\u4f5c")
    else:
        print("\u2717 \u7de9\u5b58\u7981\u7528\u529f\u80fd\u7570\u5e38")
    
    # 檢查是否沒有創建任何 .cache 文件
    import glob
    cache_files = glob.glob("*.cache")
    print(f"\u7576\u524d\u76ee\u9304\u4e2d\u7684 .cache \u6587\u4ef6: {cache_files}")
    
    if not cache_files:
        print("\u2713 \u7de9\u5b58\u7981\u7528\u6642\u6c92\u6709\u751f\u6210 .cache \u6587\u4ef6")
    else:
        print("\u2717 \u7de9\u5b58\u7981\u7528\u6642\u4ecd\u7136\u751f\u6210\u4e86 .cache \u6587\u4ef6")

def test_cache_enabled():
    """測試緩存啟用功能"""
    print("\n\u6e2c\u8a66\u7de9\u5b58\u555f\u7528\u529f\u80fd...")
    
    # 創建一個臨時目錄用於測試
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = OptimizedCache(cache_dir=temp_dir, enabled=True)
        
        # 測試 set 方法
        cache.set("test_key", "test_value")
        
        # 測試 get 方法
        result = cache.get("test_key")
        
        print(f"\u7de9\u5b58\u555f\u7528\u6642 set \u5f8c get \u7d50\u679c: {result}")
        print(f"\u9810\u671f\u7d50\u679c: test_value (\u56e0\u70ba\u7de9\u5b58\u88ab\u555f\u7528)")
        
        if result == "test_value":
            print("\u2713 \u7de9\u5b58\u555f\u7528\u529f\u80fd\u6b63\u5e38\u5de5\u4f5c")
        else:
            print("\u2717 \u7de9\u5b58\u555f\u7528\u529f\u80fd\u7570\u5e38")

if __name__ == "__main__":
    test_cache_disabled()
    test_cache_enabled()
    print("\n\u7de9\u5b58\u529f\u80fd\u6e2c\u8a66\u5b8c\u6210!")
