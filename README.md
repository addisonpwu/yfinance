# 靈活的股票篩選器 (Flexible Stock Screener)

這是一個基於 Python 的股票篩選專案，採用模組化設計，旨在根據多種可擴充的策略，從美國和香港股市中找出符合特定條件的股票。

## 專案架構

專案已重構成模組化架構，將數據獲取、策略定義和分析引擎完全分離，以實現最大的靈活性和可擴充性。

-   `main.py`: 專案的主入口，負責解析參數、調度分析並輸出結果。
-   `data_loader/`: 負責從不同來源獲取股票代碼列表。
-   `strategies/`: 存放所有選股策略。您可以輕鬆地在此處添加自己的策略。
-   `analysis/`: 核心分析引擎，負責執行所有策略並篩選股票。

## 內建策略

1.  **價漲量增 (MomentumStrategy)**
    -   **中期趨勢向上**：股價高於50日或100日移動平均線。
    -   **成交量顯著放大**：當日成交量超過過去20日平均成交量的3倍。
    -   **價格溫和上漲**：當日為紅K棒，且漲幅介於 2% 至 4% 之間。

2.  **優化價漲量增 (OptimizedMomentumStrategy)**
    -   包含上述所有基本條件。
    -   **相對強度過濾**：股價漲幅必須跑贏大盤指數 (美股: S&P 500, 港股: HSI) 至少1%。
    -   **突破訊號**：股價需突破過去10日的最高價。
    -   **資金流向**：能量潮指標 (OBV) 的5日趨勢必須向上。

## 功能

-   **模組化設計**：輕鬆添加新策略，而無需修改核心程式碼。
-   **雙市場支援**：透過命令列參數即可切換美國 (`US`) 和香港 (`HK`) 市場。
-   **自動化列表獲取**：
    -   美股：自動從 `statementdog.com` 獲取一個廣泛的美股列表。
    -   港股：自動從港交所(HKEX)網站下載最新的證券列表。
-   **動態策略加載**：分析引擎會自動偵測並執行 `strategies` 文件夾中的所有策略。
-   **清晰的結果輸出**：結果會儲存到對應市場的 `.txt` 檔案中，並標明股票通過了哪些策略。

## 安裝與設定

1.  **建立虛擬環境** (建議)
    ```bash
    python3 -m venv venv
    ```

2.  **啟用虛擬環境**
    -   在 macOS / Linux 上:
        ```bash
        source venv/bin/activate
        ```
    -   在 Windows 上:
        ```bash
        .\venv\Scripts\activate
        ```

3.  **安裝所需套件**
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

所有操作都透過 `main.py` 執行，並使用 `--market` 參數指定要分析的市場。

### 篩選美股

```bash
python3 main.py --market US
```
-   **過程**：腳本會自動獲取美股列表，然後使用所有可用策略進行分析。
-   **輸出**：符合條件的股票列表將儲存到 `us_stocks.txt` 檔案中。

### 篩選港股

```bash
python3 main.py --market HK
```
-   **過程**：腳本會自動下載港交所的股票列表，然後使用所有可用策略進行分析。
-   **輸出**：符合條件的股票列表將儲存到 `hk_stocks.txt` 檔案中。

## 如何擴充 (增加新策略)

這是此架構的核心優勢。若要增加一個新的選股策略，只需：

1.  在 `strategies/` 資料夾中建立一個新的 Python 檔案 (例如 `my_new_strategy.py`)。
2.  在該檔案中，建立一個繼承自 `BaseStrategy` 的新策略類別。
3.  實現 `name` 屬性 (策略名稱) 和 `run` 方法 (策略邏輯)。

**範例 `strategies/my_new_strategy.py`:**
```python
import pandas as pd
from .base_strategy import BaseStrategy

class MyNewStrategy(BaseStrategy):
    @property
    def name(self):
        return "我的新策略"

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # 在此實現您的策略邏輯
        # 例如，尋找收盤價高於 200 日均線的股票
        if len(hist) < 200:
            return False
        
        hist['MA200'] = hist['Close'].rolling(window=200).mean()
        latest_close = hist['Close'].iloc[-1]
        latest_ma200 = hist['MA200'].iloc[-1]

        if latest_close > latest_ma200:
            return True
        
        return False
```
完成後，分析引擎會自動偵測並執行您的新策略，無需修改任何其他程式碼！

---

## 開發日誌

-   **2025-10-04**:
    -   **重構**: 專案進行了重大重構，引入了模組化架構。
    -   **分離**: 將數據下載 (`data_loader`)、策略定義 (`strategies`) 和分析引擎 (`analysis`) 完全分離。
    -   **新增**: 建立了 `main.py` 作為統一的程式入口，並支援 `--market` 參數。
    -   **優化**: 實現了策略的動態加載，使擴充新策略變得極為簡單。
    -   **清理**: 移除了舊的 `momentum_screener.py` 和 `hk_stock_screener.py` 腳本。