# Stock Analysis Database API Skill

當用戶需要操作股票數據或新聞數據時，使用此 Skill 調用 FastAPI REST API。

## API 基礎信息

- **Base URL**: `http://localhost:8000`
- **健康檢查**: `GET /health`

## 數據模型

### Stock (股票)

| 字段 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `symbol` | string(20) | ✅ | 股票代碼，唯一鍵，**必須為4位數字+.HK** (例: `0700.HK`, `1234.HK`) |
| `name` | string(255) | ✅ | 股票名稱 |
| `market` | string(10) | ✅ | 市場代碼 (`US` 或 `HK`) |
| `id` | integer | - | 主鍵 (自動生成) |
| `created_at` | datetime | - | 創建時間 (自動) |
| `updated_at` | datetime | - | 更新時間 (自動) |

### News (新聞)

| 字段 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `stock_symbol` | string(20) | ✅ | 關聯的股票代碼，**必須為4位數字+.HK** (例: `0700.HK`, `1234.HK`) |
| `title` | text | ✅ | 新聞標題 |
| `content` | text | ✅ | 新聞內容/摘要 |
| `publish_time` | datetime | ✅ | 發布時間 |
| `url` | string(512) | ✅ | 新聞連結，唯一鍵 |
| `id` | integer | - | 主鍵 (自動生成) |
| `stock_id` | integer | - | 外鍵 → stocks.id (自動填充) |
| `created_at` | datetime | - | 創建時間 (自動) |

**關係**: News → Stock (多對一，一條新聞屬於一個股票)

---

## API 端點

### 股票 API (`/api/v1/stocks`)

#### 創建股票

```http
POST /api/v1/stocks/
Content-Type: application/json

{
  "symbol": "0700.HK",
  "name": "騰訊控股",
  "market": "HK"
}
```

**響應 (201 Created)**:
```json
{
  "id": 1,
  "symbol": "0700.HK",
  "name": "騰訊控股",
  "market": "HK",
  "created_at": "2026-03-31T15:00:00Z",
  "updated_at": "2026-03-31T15:00:00Z"
}
```

**錯誤 (409 Conflict)**: 股票已存在

---

#### 查詢股票列表

```http
GET /api/v1/stocks/?market=HK&skip=0&limit=100
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `market` | string | ❌ | 按市場篩選 (`US`/`HK`) |
| `skip` | int | ❌ | 跳過記錄數 (默認 0) |
| `limit` | int | ❌ | 返回記錄數 (默認 100) |

**響應 (200 OK)**:
```json
{
  "items": [
    {
      "id": 1,
      "symbol": "0700.HK",
      "name": "騰訊控股",
      "market": "HK",
      "created_at": "2026-03-31T15:00:00Z",
      "updated_at": "2026-03-31T15:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 100
}
```

---

#### 查詢單個股票

```http
GET /api/v1/stocks/{symbol}
```

**響應 (200 OK)**:
```json
{
  "id": 1,
  "symbol": "0700.HK",
  "name": "騰訊控股",
  "market": "HK",
  "created_at": "2026-03-31T15:00:00Z",
  "updated_at": "2026-03-31T15:00:00Z"
}
```

**錯誤 (404 Not Found)**: 股票不存在

---

### 新聞 API (`/api/v1/news`)

#### 創建新聞

```http
POST /api/v1/news/
Content-Type: application/json

{
  "stock_symbol": "0700.HK",
  "title": "騰訊發布2026年Q1財報",
  "content": "騰訊今日發布2026年第一季度財報...",
  "publish_time": "2026-03-31T10:30:00+08:00",
  "url": "https://example.com/news/001"
}
```

**參數說明**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `stock_symbol` | string | ✅ | 關聯的股票代碼 (必須已存在) |
| `title` | string | ✅ | 新聞標題 |
| `content` | string | ✅ | 新聞內容/摘要 |
| `publish_time` | datetime | ✅ | 發布時間 (ISO 8601) |
| `url` | string | ✅ | 新聞連結 (唯一) |

**響應 (201 Created)**:
```json
{
  "id": 1,
  "stock_id": 1,
  "stock_symbol": "0700.HK",
  "title": "騰訊發布2026年Q1財報",
  "content": "騰訊今日發布2026年第一季度財報...",
  "publish_time": "2026-03-31T10:30:00+08:00",
  "url": "https://example.com/news/001",
  "created_at": "2026-03-31T15:00:00Z"
}
```

**錯誤**:
- `404 Not Found`: 股票不存在
- `409 Conflict`: URL 已存在

---

#### 查詢新聞列表

```http
GET /api/v1/news/?skip=0&limit=100&stock_symbol=0700.HK&start_time=2026-03-01&end_time=2026-03-31
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `skip` | int | ❌ | 跳過記錄數 (默認 0) |
| `limit` | int | ❌ | 返回記錄數 (默認 100) |
| `stock_symbol` | string | ❌ | 按股票代碼篩選 |
| `start_time` | datetime | ❌ | 發布時間起始 (ISO 8601) |
| `end_time` | datetime | ❌ | 發布時間結束 (ISO 8601) |

**響應 (200 OK)**:
```json
{
  "items": [
    {
      "id": 1,
      "stock_id": 1,
      "stock_symbol": "0700.HK",
      "title": "騰訊發布2026年Q1財報",
      "content": "騰訊今日發布2026年第一季度財報...",
      "publish_time": "2026-03-31T10:30:00+08:00",
      "url": "https://example.com/news/001",
      "created_at": "2026-03-31T15:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 100
}
```

---

#### 查詢單個新聞

```http
GET /api/v1/news/{news_id}
```

**響應 (200 OK)**: 返回新聞對象

**錯誤 (404 Not Found)**: 新聞不存在

---

## 錯誤響應格式

所有錯誤響應格式一致：

```json
{
  "detail": "Stock not found: 0700.HK"
}
```

**HTTP 狀態碼**:
| 狀態碼 | 說明 |
|--------|------|
| 200 | 成功 (GET/PUT) |
| 201 | 創建成功 (POST) |
| 204 | 無內容 (DELETE) |
| 404 | 資源不存在 |
| 409 | 衝突 (重複資源) |
| 422 | 驗證失敗 (請求格式錯誤) |
| 500 | 服務器錯誤 |

**校驗失敗示例 (422)**:
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "symbol"],
      "msg": "Value error, 股票代码格式错误: 必须为4位数字+.HK (例如: 0700.HK, 1234.HK)",
      "input": "700.HK"
    }
  ]
}
```

---
## cURL 示例

```bash
# 健康檢查
curl http://localhost:8000/health

# 創建股票
curl -X POST http://localhost:8000/api/v1/stocks/ \
  -H "Content-Type: application/json" \
  -d '{"symbol": "0700.HK", "name": "騰訊控股", "market": "HK"}'

# 查詢港股列表
curl "http://localhost:8000/api/v1/stocks/?market=HK"

# 查詢單個股票
curl http://localhost:8000/api/v1/stocks/0700.HK

# 創建新聞 (需要指定 stock_symbol)
curl -X POST http://localhost:8000/api/v1/news/ \
  -H "Content-Type: application/json" \
  -d '{"stock_symbol": "0700.HK", "title": "騰訊發布財報", "content": "騰訊今日發布財報，營收增長...", "publish_time": "2026-03-31T10:30:00+08:00", "url": "https://example.com/news/001"}'

# 查詢特定股票的新聞
curl "http://localhost:8000/api/v1/news/?stock_symbol=0700.HK"
```

---

## 注意事項

1. **股票代碼格式**: 港股使用 `4位數字.HK` (例: `0700.HK`, `1234.HK`)
2. **時間格式**: ISO 8601 格式 (例: `2026-03-31T10:30:00+08:00`)
3. **唯一性**: `symbol` 和 `url` 字段必須唯一
4. **新聞關聯**: 創建新聞時必須指定 `stock_symbol`，股票必須已存在
5. **級聯刪除**: 刪除股票時會自動刪除相關的新聞
6. **分頁**: 默認 `limit=100`，最大建議不超過 1000