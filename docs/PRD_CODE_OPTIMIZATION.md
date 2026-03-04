# PRD: 代码质量优化与安全加固

**文档版本**: v1.0  
**创建日期**: 2026-03-03  
**状态**: 待评审  

---

## 一、问题陈述

项目存在三类主要问题，影响安全性、可维护性和性能：

| 问题类型 | 严重程度 | 影响范围 |
|---------|---------|---------|
| 安全漏洞 | 🔴 高 | 生产环境数据安全 |
| 异常处理缺陷 | 🟡 中 | 故障排查效率 |
| 代码结构问题 | 🟡 中 | 可维护性 |

### 1.1 安全漏洞详情

| 漏洞 | 位置 | 风险描述 | CVSS评分 |
|-----|------|---------|---------|
| Pickle 反序列化 | `cache_service.py:55` | 恶意缓存文件可执行任意代码 | 9.8 (Critical) |
| SSL 证书禁用 | `yahoo_loader.py:26-27` | 中间人攻击风险 | 7.5 (High) |
| 路径遍历 | `cache_service.py:25` | 缓存键未验证可访问任意文件 | 6.5 (Medium) |

### 1.2 异常处理问题统计

```
静默异常 (except Exception:): 11 处
裸异常 (except:): 2 处
合计: 13 处
```

**影响**：异常被静默吞掉，日志无记录，故障难以定位。

### 1.3 代码结构问题

| 文件 | 行数 | 问题 |
|-----|------|-----|
| `iflow_analyzer.py` | 1485 | 单文件职责过多 |
| `report_writer.py` | 1112 | HTML 模板内嵌 |

---

## 二、功能范围与非目标

### 2.1 功能范围 (In Scope)

#### 第一阶段：安全漏洞修复 (P0)

| 任务 | 描述 | 验收标准 |
|-----|------|---------|
| PICKLE-001 | 替换 pickle 为 JSON/MessagePack | 所有缓存使用安全序列化 |
| SSL-001 | 移除全局 SSL 禁用 | 仅在必要时处理证书错误 |
| PATH-001 | 验证缓存键合法性 | 拒绝包含 `..` 或绝对路径的键 |

#### 第二阶段：异常处理优化 (P1)

| 任务 | 描述 | 验收标准 |
|-----|------|---------|
| EXCEPT-001 | 静默异常添加日志 | 100% 覆盖，每处至少记录异常类型和消息 |
| EXCEPT-002 | 裸异常改为具体类型 | 无裸异常，使用 `Exception` 或具体子类 |

#### 第三阶段：代码重构 (P2)

| 任务 | 描述 | 验收标准 |
|-----|------|---------|
| REFACTOR-001 | 拆分 iflow_analyzer.py | 单文件 < 500 行 |
| REFACTOR-002 | 拆分 report_writer.py | 模板与逻辑分离 |

#### 第四阶段：性能优化 (P2)

| 任务 | 描述 | 验收标准 |
|-----|------|---------|
| PERF-001 | 启用缓存系统 | 数据缓存命中率 > 50% |
| PERF-002 | 配置缓存 TTL | 不同数据类型使用不同 TTL |

### 2.2 非目标 (Out of Scope)

| 项目 | 原因 |
|-----|------|
| 新增策略功能 | 本次仅优化，不添加新功能 |
| API 接口变更 | 保持向后兼容 |
| 数据库迁移 | 继续使用文件缓存 |
| 单元测试框架搭建 | 独立任务处理 |

---

## 三、验收标准 (Acceptance Criteria)

### 3.1 安全验收标准

```gherkin
Feature: 安全漏洞修复

Scenario: Pickle 反序列化漏洞修复
  Given 系统使用 OptimizedCache 类进行缓存
  When 调用 cache.get() 或 cache.set() 方法
  Then 不使用 pickle 模块进行序列化
  And 使用 JSON 或 MessagePack 进行序列化
  And 缓存文件扩展名为 .json 或 .msgpack

Scenario: SSL 证书验证恢复
  Given 系统通过 feedparser 获取新闻数据
  When 初始化 feedparser 模块
  Then 不全局禁用 SSL 证书验证
  And 仅在必要时捕获 ssl.SSLError 异常
  And 日志记录证书错误详情

Scenario: 缓存键路径验证
  Given OptimizedCache._get_cache_path() 方法
  When 传入缓存键参数
  Then 验证键不包含 ".." 字符
  And 验证键不包含路径分隔符
  And 验证键不以 "/" 开头
  And 非法键抛出 ValueError 异常
```

### 3.2 异常处理验收标准

```gherkin
Feature: 异常处理优化

Scenario: 静默异常日志记录
  Given 代码中存在 except Exception: 块
  When 异常发生时
  Then 使用 logger 记录异常类型
  And 使用 logger 记录异常消息
  And 使用 logger 记录堆栈跟踪 (可选)

Scenario: 裸异常消除
  Given 代码中存在 except: 块
  When 进行代码审查
  Then 将裸异常改为 except Exception as e:
  And 添加日志记录
```

### 3.3 代码质量验收标准

```gherkin
Feature: 代码重构

Scenario: 文件行数限制
  Given 项目中的 Python 文件
  When 进行代码审查
  Then 单个文件行数不超过 500 行
  And 单个函数行数不超过 50 行
  And 循环复杂度不超过 10
```

### 3.4 性能验收标准

```gherkin
Feature: 性能优化

Scenario: 缓存系统启用
  Given 配置文件中 enable_cache = true
  When 运行股票筛选程序
  Then 历史数据从缓存读取
  And 缓存命中率 > 50%
  And 总运行时间减少 > 30%

Scenario: 缓存 TTL 配置
  Given 不同类型的数据缓存
  When 设置缓存 TTL
  Then 日线数据 TTL = 7 天
  And 小时线数据 TTL = 1 小时
  And AI 分析结果 TTL = 24 小时
```

---

## 四、测试用例

### 4.1 安全测试用例

| ID | 测试场景 | 输入 | 预期结果 |
|----|---------|------|---------|
| SEC-001 | 缓存键路径遍历攻击 | `key="../../../etc/passwd"` | 抛出 ValueError |
| SEC-002 | 缓存键包含绝对路径 | `key="/etc/passwd"` | 抛出 ValueError |
| SEC-003 | 恶意 pickle 文件 | 构造恶意 .cache 文件 | 不执行代码，返回错误 |
| SEC-004 | SSL 证书验证 | 访问 HTTPS 资源 | 验证证书，记录错误 |

### 4.2 功能测试用例

| ID | 测试场景 | 前置条件 | 操作步骤 | 预期结果 |
|----|---------|---------|---------|---------|
| FUNC-001 | 缓存正常读写 | 启用缓存 | 1. 写入数据<br>2. 读取数据 | 数据一致 |
| FUNC-002 | 缓存过期清理 | TTL=1秒 | 1. 写入数据<br>2. 等待2秒<br>3. 读取数据 | 返回 None |
| FUNC-003 | 异常日志记录 | 模拟异常 | 1. 触发异常<br>2. 检查日志 | 日志包含异常信息 |
| FUNC-004 | AI 分析正常执行 | 配置 API Key | 1. 运行分析<br>2. 检查结果 | 返回分析结果 |

### 4.3 性能测试用例

| ID | 测试场景 | 基准 | 目标 |
|----|---------|------|------|
| PERF-001 | 100 只股票筛选时间 | 120s (无缓存) | < 84s (有缓存) |
| PERF-002 | 内存使用峰值 | 800MB | < 600MB |
| PERF-003 | 缓存命中率 | 0% | > 50% |

---

## 五、约束与依赖

### 5.1 技术约束

| 约束类型 | 描述 |
|---------|------|
| Python 版本 | >= 3.8 |
| 向后兼容 | 不改变公共 API 签名 |
| 依赖最小化 | 不引入新的重量级依赖 |

### 5.2 时间约束

| 阶段 | 预计工时 | 优先级 |
|-----|---------|--------|
| 第一阶段：安全修复 | 4h | P0 |
| 第二阶段：异常处理 | 2h | P1 |
| 第三阶段：代码重构 | 4h | P2 |
| 第四阶段：性能优化 | 2h | P2 |
| **合计** | **12h** | - |

### 5.3 依赖关系

```
第一阶段 (安全修复)
    ↓
第二阶段 (异常处理) ← 可并行
    ↓
第三阶段 (代码重构)
    ↓
第四阶段 (性能优化)
```

---

## 六、回滚方案

### 6.1 快速回滚

| 场景 | 回滚操作 | 命令 |
|-----|---------|------|
| 安全修复失败 | 恢复原文件 | `git checkout HEAD~1 -- src/data/cache/cache_service.py` |
| 异常处理问题 | 回退单个提交 | `git revert <commit-hash>` |
| 全局回滚 | 回退到优化前 | `git checkout <pre-optimization-tag>` |

### 6.2 数据兼容性

| 变更 | 旧缓存处理 | 新缓存格式 |
|-----|-----------|-----------|
| Pickle → JSON | 旧缓存自动失效，重新生成 | `.json` 文件 |
| 缓存键变更 | 不兼容，需清空缓存目录 | `rm -rf data_cache/` |

### 6.3 功能降级

```python
# 安全序列化降级方案
try:
    import msgpack  # 优先使用 MessagePack (更高效)
    SERIALIZE_FORMAT = "msgpack"
except ImportError:
    SERIALIZE_FORMAT = "json"  # 降级到 JSON

# 缓存禁用降级
if not ENABLE_SAFE_CACHE:
    logger.warning("安全缓存不可用，禁用缓存功能")
    cache_service.enabled = False
```

---

## 七、交付清单

### 7.1 代码变更

- [ ] `src/data/cache/cache_service.py` - 安全序列化
- [ ] `src/data/loaders/yahoo_loader.py` - SSL 修复
- [ ] `src/ai/analyzer/iflow_analyzer.py` - 异常处理 + 拆分
- [ ] `src/ai/analyzer/nvidia_analyzer.py` - 异常处理
- [ ] `src/ai/analyzer/gemini_analyzer.py` - 异常处理
- [ ] `src/core/services/report_writer.py` - 异常处理 + 拆分

### 7.2 配置变更

- [ ] `config.json` - 启用缓存，配置 TTL

### 7.3 文档更新

- [ ] 更新 AGENTS.md 记录变更
- [ ] 更新 README.md 安全说明

---

## 八、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| 序列化格式变更导致缓存失效 | 高 | 低 | 清空缓存目录即可恢复 |
| SSL 修复导致部分 HTTPS 访问失败 | 低 | 中 | 添加证书错误回退逻辑 |
| 重构引入新 Bug | 中 | 中 | 分阶段提交，每阶段测试 |

---

## 九、审批记录

| 角色 | 姓名 | 日期 | 状态 |
|-----|------|------|------|
| 产品经理 | - | 2026-03-03 | 待审批 |
| 技术负责人 | - | - | 待审批 |
| 安全负责人 | - | - | 待审批 |

---

**文档结束**
