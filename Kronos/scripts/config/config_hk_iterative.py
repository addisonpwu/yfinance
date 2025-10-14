TICKER = "1810.HK"  # Default ticker
MODEL_ID = "NeoQuasar/Kronos-base"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
CONTEXT_LENGTH = 512
LOOKBACK = 400
INTERVAL = "1h"  # Data interval: "1d", "1h", "5m", etc.
DEVICE = "cpu"

# --- Iterative Prediction Parameters ---
# Total length of the forecast
TOTAL_PRED_LEN = 121
# In each step, predict `SHORT_STEP` candles
SHORT_STEP = 20
# Number of samples for probabilistic forecasting in each step
SAMPLE_COUNT = 2
