## 任務目標
讀取指定路徑下的新聞 JSON 文件，精準識別涉及**港股**的利好與利空消息，並嚴格依照分類規則調用 `stock-api-skill` 同步數據至本地 API 服務。

## 輸入文件
- `~/Documents/temp/stock_news.json`
- `~/Documents/temp/aastocks_news.json`
- `~/Documents/temp/eastmoney_hk_news.json`

## 新聞分類標準
| **正面消息** | 盈喜、預增、業績預告、內幕消息、正面盈喜、業績大增、純利增長、盈利上升、正面評級（強烈推薦／買入／增持／上調評級）等 | 直接新增股票與新聞 |
| **負面消息** | 盈警、虧損、下調/降低評級、違約、訴訟、盈利倒退、退市風險等具負面含義內容 | 需先驗證股票是否存在，再決定是否新增新聞 |

## 核心規範（必須嚴格遵守）
1. **港股代碼格式**：統一轉換為標準 `XXXX.HK`（固定 4 位數字）。
   - 處理規則：去除前導零至剩餘 4 位；若不足 4 位則前面補零。
   - 範例：`01941.HK` → `1941.HK` ｜ `00117.HK` → `0117.HK` ｜ `5.HK` → `0005.HK`
2. **市場過濾**：僅處理明確標示或可推斷為**港股市場**的新聞，A 股、美股及其他市場新聞一律跳過。
3. **原始文件保護**：`stock_news.json`、`aastocks_news.json`、`eastmoney_hk_news.json` **絕對不可刪除、移動或修改**。
4. **僅處理港股相關新聞**。

##正面消息處理流程
1. 使用 stock-api-skill 調用本地服務接口，新增股票及新聞至系統
2. 清理執行過程中生成的臨時檔案（注意：保留原始文件 stock_news.json、aastocks_news.json、eastmoney_hk_news.json）

##負面消息處理流程
1. 查詢股票是否存在：調用 GET /api/v1/stocks/{symbol} 檢查該股票是否已存在於系統中
- 若返回錯誤（股票不存在）：停止處理，不創建該股票的新聞及代碼
- 若返回成功（股票已存在）：繼續下一步
2. 使用 stock-api-skill 調用本地服務接口，新增該股票的新聞
3. 清理執行過程中生成的臨時檔案（注意：保留原始文件 stock_news.json、aastocks_news.json、eastmoney_hk_news.json）

