# 新闻服务扩展 PRD - 产品需求规格

## 1. 项目概述

### 问题陈述
当前系统仅支持 Yahoo Finance RSS 新闻源，缺少对 Google News 的支持。需要扩展新闻服务架构，支持多新闻源切换和聚合。

### 计划概述
1. 创建 `GoogleNewsRepository` 类获取 Google News RSS
2. 创建 `NewsService` 统一接口支持 Yahoo/Google/both
3. 配置驱动，可在 config.json 中切换新闻源

---

## 2. 详细验收标准

### 2.1 功能验收

| 序号 | 验收项 | 验收条件 | 测试方法 |
|------|--------|----------|----------|
| F-01 | Google News RSS URL 构建 | URL 格式为 `https://news.google.com/rss/search?q={symbol}+when:{days_back}d&hl={lang}&gl={region}` | 单元测试验证 URL 生成 |
| F-02 | 搜索运算符支持 | 支持 OR、intitle:、when: 等运算符 | 验证 URL 包含正确运算符 |
| F-03 | 语言参数 hl | 支持 zh-TW、en-US 等语言代码 | 配置不同语言，验证 URL 参数 |
| F-04 | 地区参数 gl | 支持 TW、HK、CN 等地区代码 | 配置不同地区，验证 URL 参数 |
| F-05 | RSS 解析 | 正确解析 title、link、published、description | 解析实际 RSS 响应，验证字段 |
| F-06 | 时间过滤 | 按 days_back 参数过滤过期新闻 | 设置 days_back=1，验证过滤效果 |
| F-07 | 数量限制 | 返回新闻数量不超过 max_items | 设置 max_items=3，验证返回数量 |

### 2.2 配置验收

| 序号 | 验收项 | 验收条件 | 测试方法 |
|------|--------|----------|----------|
| C-01 | 新闻源配置 | config.json 可配置 source: yahoo/google/both | 修改配置，重启程序验证 |
| C-02 | 语言默认值 | 支持自定义默认语言 (default_lang) | 配置 default_lang: "en-US" |
| C-03 | 地区默认值 | 支持自定义默认地区 (default_region) | 配置 default_region: "HK" |
| C-04 | days_back 生效 | 配置 days_back 传递给新闻源 | 配置不同值，验证过滤效果 |
| C-05 | max_news_items 生效 | 配置 max_news_items 限制返回数量 | 配置不同值，验证返回数量 |
| C-06 | 缓存 TTL 生效 | cache_ttl_hours 配置生效 | 检查缓存文件生命周期 |

### 2.3 集成验收

| 序号 | 验收项 | 验收条件 | 测试方法 |
|------|--------|----------|----------|
| I-01 | Yahoo 源切换 | 配置 source: "yahoo" 只用 Yahoo | 配置后运行，验证使用 Yahoo |
| I-02 | Google 源切换 | 配置 source: "google" 只用 Google | 配置后运行，验证使用 Google |
| I-03 | both 模式 | source: "both" 同时获取两个源 | 配置后运行，验证返回合并结果 |
| I-04 | 向后兼容 | 现有 get_news() 调用方无需修改 | 现有代码直接运行 |
| I-05 | 回退机制 | Google 失败时自动使用 Yahoo | 模拟 Google 失败，验证回退 |
| I-06 | 源优先级 | both 模式按配置优先级合并 | 验证重复新闻去重逻辑 |

### 2.4 质量验收

| 序号 | 验收项 | 验收条件 | 测试方法 |
|------|--------|----------|----------|
| Q-01 | 解析容错 | RSS 解析失败不崩溃，记录日志 | 发送畸形 RSS，验证日志 |
| Q-02 | 网络容错 | 网络超时不崩溃，返回空列表 | 模拟网络超时 |
| Q-03 | 空结果处理 | 无新闻时返回空列表 [] | 测试无新闻股票 |
| Q-04 | 缓存一致性 | 缓存键包含新闻源标识 | 检查缓存文件名/键 |
| Q-05 | 日志记录 | 记录新闻源、获取数量、耗时 | 查看日志输出 |
| Q-06 | 类型安全 | 返回 List[Dict]，字段类型正确 | 单元测试类型检查 |

---

## 3. 可测试检查清单

### 3.1 单元测试检查清单

```
[ ] GoogleNewsRepository 类存在且可实例化
[ ] GoogleNewsRepository.get_news() 返回 List[Dict]
[ ] get_news() 包含 title, link, published, summary, source 字段
[ ] symbol 参数正确处理大小写
[ ] market 参数正确映射到 hl/gl
[ ] days_back 参数正确构建 URL
[ ] 空 symbol 时抛出异常或返回空
[ ] 网络错误时返回空列表而非崩溃
[ ] 缓存键包含 source 标识 (google_news)
```

### 3.2 集成测试检查清单

```
[ ] NewsService 存在且可实例化
[ ] source="yahoo" 时调用 YahooFinanceRepository
[ ] source="google" 时调用 GoogleNewsRepository
[ ] source="both" 时调用两个源并合并
[ ] 合并结果按时间降序排列
[ ] 重复新闻（相同 title/link）去重
[ ] 回退机制：Google 异常时只用 Yahoo
[ ] 回退机制：Yahoo 异常时只用 Google
[ ] 回退机制：两者都失败时返回空列表
```

### 3.3 配置测试检查清单

```
[ ] config.json 存在 news.source 字段
[ ] news.source: "yahoo" 配置生效
[ ] news.source: "google" 配置生效
[ ] news.source: "both" 配置生效
[ ] news.default_lang 配置生效
[ ] news.default_region 配置生效
[ ] news.days_back 配置生效
[ ] news.max_news_items 配置生效
[ ] news.cache_ttl_hours 配置生效
[ ] 无效 source 值时默认使用 "yahoo"
```

### 3.4 端到端测试检查清单

```
[ ] main.py 运行 --market HK 使用配置的新闻源
[ ] main.py 运行 --market US 使用配置的新闻源
[ ] HTML 报告中显示新闻来源标识
[ ] TXT 报告中显示新闻来源标识
[ ] 日志中显示新闻获取详情
[ ] 多次运行使用缓存（验证 cache_ttl）
```

---

## 4. 技术规格

### 4.1 新增文件结构

```
src/
├── data/
│   ├── loaders/
│   │   ├── google_news_loader.py    # 新增：Google News RSS 加载器
│   │   └── news_service.py          # 新增：统一新闻服务
```

### 4.2 GoogleNewsRepository 类设计

```python
class GoogleNewsRepository:
    def __init__(self):
        # 初始化配置和缓存
        pass
    
    def get_news(
        self, 
        symbol: str, 
        market: str = "HK", 
        days_back: int = None, 
        max_items: int = None,
        lang: str = None,
        region: str = None
    ) -> List[Dict]:
        """
        从 Google News RSS 获取股票相关新闻
        
        Args:
            symbol: 股票代码
            market: 市场代码 (HK/US)
            days_back: 回溯天数
            max_items: 最大新闻数
            lang: 语言代码 (覆盖默认)
            region: 地区代码 (覆盖默认)
            
        Returns:
            List[Dict]: 新闻列表
        """
        pass
    
    def _build_rss_url(...) -> str:
        """构建 Google News RSS URL"""
        pass
    
    def _parse_rss(...) -> List[Dict]:
        """解析 RSS 响应"""
        pass
```

### 4.3 NewsService 类设计

```python
class NewsService:
    def __init__(self, source: str = "yahoo"):
        """
        统一新闻服务接口
        
        Args:
            source: 新闻源 (yahoo/google/both)
        """
        pass
    
    def get_news(
        self, 
        symbol: str, 
        market: str = "HK", 
        days_back: int = None, 
        max_items: int = None
    ) -> List[Dict]:
        """获取新闻的统一接口"""
        pass
    
    def _fetch_yahoo(...) -> List[Dict]:
        """获取 Yahoo 新闻"""
        pass
    
    def _fetch_google(...) -> List[Dict]:
        """获取 Google 新闻"""
        pass
    
    def _merge_results(...) -> List[Dict]:
        """合并多源结果"""
        pass
```

### 4.4 配置扩展

```json
{
  "news": {
    "source": "both",
    "default_lang": "zh-TW",
    "default_region": "HK",
    "timeout": 60000,
    "max_news_items": 10,
    "days_back": 14,
    "cache_ttl_hours": 6,
    "fallback_order": ["google", "yahoo"]
  }
}
```

---

## 5. 实现任务分解

### 任务 1: GoogleNewsRepository 实现
- [ ] 创建 `src/data/loaders/google_news_loader.py`
- [ ] 实现 `_build_rss_url()` 方法
- [ ] 实现 `_parse_rss()` 方法  
- [ ] 实现 `get_news()` 主方法
- [ ] 添加缓存支持
- [ ] 添加错误处理和日志

### 任务 2: NewsService 实现
- [ ] 创建 `src/data/loaders/news_service.py`
- [ ] 实现 Yahoo/Google 源切换逻辑
- [ ] 实现 both 模式合并逻辑
- [ ] 实现回退机制
- [ ] 添加缓存协调

### 任务 3: 配置集成
- [ ] 扩展 config.json 添加 news 配置项
- [ ] 扩展 settings.py 的 NewsConfig
- [ ] 修改 StockRepository 接口说明
- [ ] 更新 StockAnalyzer 使用 NewsService

### 任务 4: 测试验证
- [ ] 编写 GoogleNewsRepository 单元测试
- [ ] 编写 NewsService 单元测试
- [ ] 编写集成测试
- [ ] 端到端手动测试

---

## 6. 回退和错误处理策略

| 场景 | 处理策略 |
|------|----------|
| Google RSS 解析失败 | 记录日志，返回空列表，尝试回退到 Yahoo |
| Yahoo RSS 解析失败 | 记录日志，返回空列表，尝试回退到 Google |
| 网络超时 | 记录日志，返回空列表，触发回退 |
| both 模式一个源失败 | 记录日志，使用成功的源继续 |
| 两个源都失败 | 记录错误日志，返回空列表 |
| 配置 source 无效 | 默认使用 "yahoo"，记录警告 |

---

## 7. 日志规范

```python
# 示例日志输出
INFO - [NewsService] 初始化新闻服务，source=both
INFO - [GoogleNews] 获取 0700.HK 新闻，days_back=14, max_items=10
INFO - [GoogleNews] 成功获取 5 条新闻，耗时 0.32s
INFO - [YahooNews] 获取 0700.HK 新闻，成功 8 条
INFO - [NewsService] 合并结果：Google 5条 + Yahoo 8条 = 去重后 10条
WARNING - [GoogleNews] 获取 AAPL 新闻失败，回退到 Yahoo
ERROR - [NewsService] 所有新闻源失败，symbol=AAPL
```

---

## 8. 关键约束

1. **向后兼容**: 现有调用 `get_news()` 的代码无需修改
2. **无外部依赖**: Google News 使用标准 RSS，无需额外 SDK
3. **缓存兼容**: 缓存键需包含源标识，避免 Yahoo/Google 缓存冲突
4. **性能优先**: both 模式应并行获取，减少总耗时
5. **配置驱动**: 所有行为可通过 config.json 控制
