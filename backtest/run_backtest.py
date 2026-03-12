import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ingestion import DataIngestion
from core.features  import FeatureEngine
from core.detector  import ManipulationDetector
from datetime import datetime
import numpy as np
EVENTS = [
    ('LUNC/USDT', datetime(2022,4,1),  datetime(2022,5,9),  "LUNA Collapse", 0.70),
    ('FTT/USDT',  datetime(2022,10,1), datetime(2022,11,8),  "FTX Collapse",  0.68),
    ('BUSD/USDT', datetime(2023,1,1),  datetime(2023,2,13), "BUSD Depeg",    0.65),
    ('BTC/USDT', datetime(2022,10,1), datetime(2023,1,15), 'Normal day', 0.0),
]

ingestion = DataIngestion()
engine    = FeatureEngine()
detector  = ManipulationDetector()

for symbol, train_start, crash_dt, label, min_score in EVENTS:
    print(f"\n{'─'*50}\n{label} | {symbol}")
    df_train = ingestion.fetch_ohlcv(symbol, limit=500, since=train_start)
    if df_train.empty: print("No data"); continue
    detector.fit(engine.compute(df_train).values)

    df_crash = ingestion.fetch_ohlcv(symbol, limit=100, since=crash_dt)
    if df_crash.empty: print("No data"); continue
    X      = engine.compute(df_crash).values
    scores = detector.score(X)

    peak = scores.max()
    fired = (scores > 0.65).any()
    status = "✓ PASS" if (label == "Normal day" and not fired) \
             or (label != "Normal day" and peak >= min_score) \
             else "✗ FAIL — fix features"

    print(f"Peak score:  {peak:.3f}")
    print(f"Status:      {status}")
    if fired:
        first_idx = (scores > 0.65).argmax()
        print(f"First alert: {df_crash.index[first_idx]}")