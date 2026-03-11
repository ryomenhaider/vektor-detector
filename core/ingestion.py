import ccxt
import pandas as pd
import time
from datetime import datetime
from config import TIMEFRAME, EXCHANGE, LIVE_LOOKBACK, LOOKBACK, ALERT_THRESHOLD, POLL_INTERVAL

class DataIngestion:

    def __init__(self, exchange_id=EXCHANGE):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit' : True,
            'timeout': 30000
        })
    
    def fetch_ohlcv(self, symbol, timeframe=TIMEFRAME, limit=LOOKBACK, since=None):
        try:
            since_ms = None
            if since:
                since_ms = int(since.timestamp() * 1000)
            
            raw = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_ms
            )

            df = pd.DataFrame(raw, columns=[
                'timestamps',
                'open',
                'close',
                'low',
                'high',
                'volume'
            ])

            df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms')

            df = df.set_index('timestamps')

            return df
 
        except ccxt.NetworkError as e:
            print(f'network Error for {symbol}: {e}')
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            print(f'Exchange Error for {symbol}: {e}')


    def fetch_multi(self, symbols, timeframe=TIMEFRAME, limit=LOOKBACK):

        results = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                results[symbol] = df
            time.sleep(0.5)
        return results


