##任務
讀取~/Documents/temp中stock_news.json、aastocks_news.json、eastmoney_hk_news.json，當發現有利好消息的**港股**，使用~/.qclaw/skills/stock-api-skill新增至股票及新聞。

##目標文件
~/Documents/temp/stock_news.json
~/Documents/temp/aastocks_news.json
~/Documents/temp/eastmoney_hk_news.json

##新聞分類
1. 正面消息：例如盈喜｜預增｜業績預告｜內幕消息｜正面盈喜｜業績大增｜純利增長｜盈利上升｜正面評級（強烈推薦｜買入｜增持｜上調評級）......
2. 負面消息：盈警、虧損、下調或降低行業評級等**有負面含意**新聞

##核心要求（必須遵守）
### 1. 股票代碼格式
- **必須轉為4位數字+.HK**
- 5位數→4位數轉換：去掉前導零
- 示例：01941.HK → 1941.HK, 00117.HK → 0117.HK
- 必須要是港股相關新聞

##正面消息處理流程
1. 使用stock-api-skill調用本地服務接口新增股票及新聞
2. **把執行任務時生成的檔案移除（stock_news.json、aastocks_news.json、eastmoney_hk_news.json不要刪除）**

##負面消息處理流程
1. 先使用stock-api-skill調用本地服務接口，查看是否存在這個股票（GET /api/v1/stocks/{symbol}），如返回**錯誤**，側**停止創建該股票的新聞**:
2. 如查看是否存在這個股票（GET /api/v1/stocks/{symbol}）返回有這個股票，側使用stock-api-skill調用本地服務接口新增該股票的新聞
3. **把執行任務時生成的檔案移除（stock_news.json、aastocks_news.json、eastmoney_hk_news.json不要刪除）**