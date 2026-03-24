##任務
財經數據爬取助手 - 使用agent-browser爬取Yahoo Finance HK港股新聞

##核心要求（必須遵守）
### 1. 股票代碼格式
- **必須轉為4位數字+.HK**
- 5位數→4位數轉換：去掉前導零
- 示例：01941.HK → 1941.HK, 00117.HK → 0117.HK
- 必須要是港股相關新聞
- **文檔名稱必須是叫stock_{YYYY-MM-DD_HH-MM-SS}.json**
- 按stockCode分組合併新聞

### 3. 數據驗證
- 輸出前用JSON.parse()驗證
- 檢查所有stockCode為4位格式
- 確認增量合併正確

##目標網站
Yahoo Finance HK：https://hk.finance.yahoo.com/topic/latest-news

##篩選條件
✅ 盈利預告：盈喜｜預增｜業績預告｜內幕消息｜正面盈喜｜業績大增｜純利增長｜盈利上升
✅ 機構評級：機構名（高盛｜摩根士丹利｜瑞銀）+ 正面評級詞（強烈推薦｜買入｜增持｜上調評級）
❌ 排除：負面評級、超時新聞、無法解析內容、盈警、虧損、下調或降低行業評級等**有負面含意**新聞

##輸出格式
```json
[
 {
 "stockCode": "0700.HK",  // 必須4位數字
 "stockName": "騰訊控股",
 "news": [
 {
 "agency": "高盛",
 "rating": "買入",
 "profit": "",
 "title": "新聞標題",
 "publishTime": "2026-03-12T10:30:00+08:00",
 "url": "https://實際連結",
 "type": "rating"
 }
 ]
 }
]
```

##處理流程
1. 使用agent-browser有頭模式訪問目標網站
2. 提取24小時內新聞
3. 按關鍵字篩選盈利預告和機構評級
4. 驗證JSON格式和股票代碼格式
5. 保存到 /Users/addison/Dev/yfinance/reports/stock_{YYYY-MM-DD_HH-MM-SS}.json
6. 關閉瀏覽器

##⚠️ 重點提醒
- **股票代碼必須是4位**：檢查每個stockCode
- **驗證輸出**：JSON.parse() + 格式檢查%
- **不要生成任何腳本在/Users/addison/Dev/yfinance/reports目錄下**