WATCH_LIST = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'BNB/USDT',
    'PEPE/USDT',
    'ARB/USDT',
    'AVAX/USDT',
    'DOGE/USDT',
    'SUI/USDT',
    'INJ/USDT', 
    'TIA/USDT', 
    'WIF/USDT',
]

# Exchange + candle settings
EXCHANGE = 'binance'
TIMEFRAME = '5m'
LOOKBACK = 500   # candles for training
LIVE_LOOKBACK = 60   # candles for live scoring

# Detection settings
ALERT_THRESHOLD = 0.93
POLL_INTERVAL = 300   # seconds between runs
MAX_ALERTS_PER_RUN = 3   # max alerts per detection cycle
BTC_MOVE_FILTER = 0.02   # suppress if BTC moves > 2%

# Model settings
N_ESTIMATORS = 200
CONTAMINATION = 0.05