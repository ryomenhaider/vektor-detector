import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pandas as pd
import numpy as np

class FeatureEngine:

  def compute(self, df: pd.DataFrame) -> pd.DataFrame:
    
    
    f = df.copy()
    print('Calculating basic features like mean, std, delta, etc....')
    range = (f['high'] - f['low']).replace(0, np.nan)
    body = (f['close'] - f['open']).abs()
    vol_mean = f['volume'].rolling(20).mean()
    vol_std = f['volume'].rolling(20).std()
    delta = f['close'].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(lower=0)).ewm(span=14, adjust=False).mean()

    print('Calculating Z Score...')
    f['vol_zscore'] = (f['volume'] - vol_mean ) / vol_std

    print('Calculating Wick Ratio...')
    f['wick_ratio'] = 1 - (body / range)

    print('calculating pv_divergence...')
    f['pv_divergence'] = (f['volume'].pct_change()).abs() - (f['close'].pct_change()).abs()

    print('Calculating Volume Per Move ...')
    f['vol_per_move'] = f['volume'] / range

    print('Calculating RSI...')
    f['rsi'] = 100 - (100 / 1 + (gain/loss))

    print('Calculating Vol_accel')
    f['vol_accel'] = f['volume'].pct_change(3)

    print('Calculating Spread proxy....')
    f['spread_proxy'] = range / f['close']

    cols = ['vol_zscore','wick_ratio','pv_divergence',
            'vol_per_move','rsi','vol_accel','spread_proxy']

    df =  f[cols].replace([np.inf, -np.inf], np.nan).dropna()
    df.to_csv('data/features/enrichdata.csv', index=False)

    return df
  
featureengineering = FeatureEngine()

df = pd.read_csv('data/dataofbtc.csv')
featureengineering.compute(df)